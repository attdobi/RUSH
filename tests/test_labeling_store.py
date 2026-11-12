"""Data-integrity tests for the additive labeling lifecycle, without providers."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from labeling import Actor, LabelingError, LabelingStore
from labeling.policy_graph import GraphValidationError, graph_digest, load_version_directory, normalize_policy_graph, validate_route_trace


ROOT = Path(__file__).resolve().parents[1]


class LabelingStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.db = Path(self.temp.name) / "labels.sqlite3"
        self.store = LabelingStore(self.db)
        self.fixture = json.loads((ROOT / "web/labeling/fixtures.json").read_text())
        self.store.seed_fixture(self.fixture)
        self.policy = Actor(**self.fixture["viewer"])
        self.sme = Actor("sme-test", "SME reviewer", "sme")
        self.ops = Actor("ops-test", "Ops reviewer", "ops")
        self.blind_id = next(i["id"] for i in self.fixture["items"] if i["type"] == "assistant" and not i.get("seed_independent_labels"))
        self.ready_id = next(i["id"] for i in self.fixture["items"] if i.get("seed_independent_labels"))

    def assert_error(self, code, call):
        with self.assertRaises(LabelingError) as caught:
            call()
        self.assertEqual(code, caught.exception.code)

    def payload(self, item, *, phase="independent", decision="save", labels=None):
        return {"base_revision": item["revision"], "phase": phase, "decision": decision,
                "labels": labels or {t["id"]: t["choices"][0] for t in item["targets"]},
                "rationale": "Original evidence reviewed for each target under the current policy.", "policy_refs": []}

    def approve_ready(self):
        item = self.store.get_item(self.ready_id, self.policy)
        return self.store.review_item(self.ready_id, self.policy, self.payload(item, phase="adjudication", decision="approve"))

    def test_first_pass_is_actor_specific_and_tracks_separate_targets(self):
        item = self.store.get_item(self.blind_id, self.policy)
        self.assertFalse(item["guidance_visible"])
        self.assertTrue(all(t["ai"] is None and t["gold"] is None for t in item["targets"]))
        labels = {"input": "Violation", "output": "No violation"}
        reviewed = self.store.review_item(self.blind_id, self.policy, self.payload(item, labels=labels))
        self.assertEqual(labels, reviewed["independent_labels"])
        self.assertTrue(all(t["ai"] for t in reviewed["targets"]))
        self.assertIsNone(reviewed["assignee"], "Completed Ops work must be available for expert handoff")
        other = self.store.get_item(self.blind_id, self.ops)
        self.assertFalse(other["guidance_visible"])
        self.assertTrue(all(t["gold"] is None for t in other["targets"]))
        event = reviewed["events"][0]
        self.assertFalse(event["ai_visible"])
        self.assertEqual({"input", "output"}, {t["id"] for t in event["targets"]})
        self.assertTrue(all(t["category"] and "before" in t and "after" in t for t in event["targets"]))

    def test_summary_and_activity_do_not_leak_other_reviewers_labels(self):
        self.approve_ready()
        result = self.store.get_item(self.ready_id, self.ops)
        for target in result["targets"]:
            self.assertIsNone(target["ai"]); self.assertIsNone(target["gold"]); self.assertIsNone(target["staged_label"])
        for event in result["events"] + self.store.events(self.ops)["events"]:
            self.assertNotIn("labels", event)
            self.assertNotIn("rationale", event)
            for target in event.get("targets", []):
                self.assertNotIn("before", target); self.assertNotIn("after", target)
        for summary in self.store.list_items(self.ops)["items"]:
            self.assertNotIn("targets", summary); self.assertNotIn("events", summary)

    def test_idempotency_replays_a_committed_result_and_rejects_new_body(self):
        item = self.store.get_item(self.blind_id, self.policy)
        body = self.payload(item)
        first = self.store.review_item(item["id"], self.policy, body, idempotency_key="first-pass")
        again = self.store.review_item(item["id"], self.policy, body, idempotency_key="first-pass")
        self.assertEqual(first, again)
        self.assertEqual(1, len(first["events"]))
        self.assert_error("idempotency_conflict", lambda: self.store.review_item(item["id"], self.policy, {**body, "rationale": "Different request"}, idempotency_key="first-pass"))
        self.assert_error("revision_conflict", lambda: self.store.review_item(item["id"], self.policy, body, idempotency_key="new-key"))

    def test_concurrent_writers_cannot_silently_overwrite_a_revision(self):
        item = self.store.get_item(self.blind_id, self.policy)
        body = self.payload(item)
        def write(actor):
            try:
                return self.store.review_item(item["id"], actor, body)["revision"]
            except LabelingError as exc:
                return exc.code
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(write, [self.policy, self.sme]))
        self.assertCountEqual([2, "revision_conflict"], results)
        self.assertEqual(1, len(self.store.get_item(item["id"], self.policy)["events"]))

    def test_ops_cannot_approve_publish_or_request_reruns(self):
        item = self.store.get_item(self.blind_id, self.ops)
        saved = self.store.review_item(item["id"], self.ops, self.payload(item))
        self.assert_error("forbidden", lambda: self.store.review_item(item["id"], self.ops, self.payload(saved, phase="adjudication", decision="approve")))
        epoch = self.store.bootstrap(self.ops)["epoch"]
        self.assert_error("forbidden", lambda: self.store.publish_epoch(self.ops, {"expected_epoch": epoch, "note": "Unauthorized"}))
        self.assert_error("forbidden", lambda: self.store.request_rerun(self.ops, {"epoch": epoch, "note": "Unauthorized"}))
        self.assert_error("forbidden", lambda: self.store.export_epoch(epoch, self.ops))

    def test_handoff_allows_ops_review_then_independent_expert_approval(self):
        item = self.store.get_item(self.blind_id, self.ops)
        saved = self.store.review_item(item["id"], self.ops, self.payload(item))
        expert = self.store.get_item(item["id"], self.sme)
        self.assertFalse(expert["guidance_visible"])
        self.assert_error("first_pass_required", lambda: self.store.review_item(item["id"], self.sme, self.payload(expert, phase="adjudication", decision="approve")))
        reviewed = self.store.review_item(item["id"], self.sme, self.payload(expert))
        approved = self.store.review_item(item["id"], self.sme, self.payload(reviewed, phase="adjudication", decision="approve"))
        self.assertEqual("approved", approved["status"])
        self.assertEqual(saved["revision"] + 2, approved["revision"])

    def test_labels_require_exact_targets_and_no_uncertain_approval(self):
        item = self.store.get_item(self.ready_id, self.policy)
        body = self.payload(item, phase="adjudication", decision="approve")
        self.assert_error("validation_error", lambda: self.store.review_item(item["id"], self.policy, {**body, "labels": {"input": "Violation"}}))
        self.assert_error("validation_error", lambda: self.store.review_item(item["id"], self.policy, {**body, "labels": {**body["labels"], "other": "Violation"}}))
        self.assert_error("unresolved_label", lambda: self.store.review_item(item["id"], self.policy, {**body, "labels": {"input": "Uncertain", "output": "No violation"}}))
        self.assertEqual(item["revision"], self.store.get_item(item["id"], self.policy)["revision"])

    def test_approval_stages_before_publication_and_old_epoch_stays_immutable(self):
        epoch = self.store.bootstrap(self.policy)["epoch"]
        original = self.store.export_epoch(epoch)
        before = self.store.get_item(self.ready_id, self.policy)
        approved = self.approve_ready()
        self.assertEqual([t["gold"] for t in before["targets"]], [t["gold"] for t in approved["targets"]])
        self.assertTrue(all(t["staged_label"] is not None for t in approved["targets"]))
        new_epoch = self.store.publish_epoch(self.policy, {"expected_epoch": epoch, "note": "Reviewed label decisions"}, idempotency_key="publish")
        self.assertEqual(epoch + 1, new_epoch["epoch"])
        self.assertEqual(original, self.store.export_epoch(epoch))
        published = self.store.get_item(self.ready_id, self.policy)
        self.assertEqual("published", published["status"])
        self.assertTrue(all(t["staged_label"] is None for t in published["targets"]))
        self.assertEqual(new_epoch["digest"], self.store.export_epoch(new_epoch["epoch"])["digest"])
        self.assert_error("epoch_conflict", lambda: self.store.publish_epoch(self.policy, {"expected_epoch": epoch, "note": "Stale epoch"}))
        self.assert_error("no_changes", lambda: self.store.publish_epoch(self.policy, {"expected_epoch": epoch+1, "note": "Nothing staged"}))

    def test_holdout_cannot_enter_optimization_and_outbox_pins_snapshot(self):
        protected = next(i for i in self.fixture["items"] if i["split"] == "holdout")
        item = self.store.get_item(protected["id"], self.policy)
        saved = self.store.review_item(item["id"], self.policy, self.payload(item))
        self.assert_error("protected_holdout", lambda: self.store.review_item(item["id"], self.policy, self.payload(saved, phase="adjudication", decision="approve")))
        epoch = self.store.bootstrap(self.policy)["epoch"]
        export = self.store.export_epoch(epoch)
        self.assertNotIn(item["id"], {i["id"] for i in export["items"]})
        run = self.store.request_rerun(self.policy, {"epoch": epoch, "note": "Explicit connector handoff"})
        self.assertEqual("awaiting_connector", run["status"])
        self.assertEqual(export["digest"], run["digest"])
        restarted = LabelingStore(self.db)
        self.assertEqual(run, restarted.bootstrap(self.policy)["runs"][0])

    def test_escalation_preserves_gold_and_route_change_invalidates_staged_approval(self):
        approved = self.approve_ready()
        routed = self.store.route_item(approved["id"], self.policy, {"base_revision": approved["revision"], "nodes": [], "reason": "Needs broad policy context"})
        self.assertTrue(routed["route"]["fallback"])
        self.assertEqual("open", routed["status"])
        self.assertTrue(all(t["staged_label"] is None for t in routed["targets"]))
        escalated = self.store.review_item(routed["id"], self.policy, self.payload(routed, phase="adjudication", decision="escalate"))
        self.assertEqual("escalated", escalated["status"])
        self.assertTrue(escalated["escalation_owner"])
        self.assertEqual([t["gold"] for t in approved["targets"]], [t["gold"] for t in escalated["targets"]])

    def test_import_sanitizes_state_and_never_overwrites_review_history(self):
        self.approve_ready()
        old = self.store.get_item(self.ready_id, self.policy)
        raw = copy.deepcopy(self.fixture["items"][0]); raw["id"] = "imported-17"
        raw.update({"revision": 88, "status": "published", "events": [{"actor": "forged", "labels": {"input": "Violation"}}], "assignee": "forged", "guidance_visible": True})
        for target in raw["targets"]: target["staged_label"] = target["choices"][0]
        result = self.store.import_items(self.policy, {"items": [raw, self.fixture["items"][0]]})
        self.assertEqual({"imported": 1, "skipped": 1}, result)
        candidate = self.store.get_item("imported-17", self.policy)
        self.assertEqual("open", candidate["status"]); self.assertEqual(1, candidate["revision"])
        self.assertIsNone(candidate["assignee"]); self.assertFalse(candidate["guidance_visible"])
        self.assertTrue(all(t["gold"] is None and t["staged_label"] is None for t in candidate["targets"]))
        self.assertEqual(["import"], [e["type"] for e in candidate["events"]])
        self.assertEqual(old, self.store.get_item(self.ready_id, self.policy))

    def test_import_rejects_duplicate_targets_unsafe_images_and_split_leakage(self):
        raw = copy.deepcopy(self.fixture["items"][0]); raw["id"] = "invalid-targets"
        raw["targets"][1]["id"] = raw["targets"][0]["id"]
        self.assert_error("validation_error", lambda: self.store.import_items(self.policy, {"items": [raw]}))
        raw = copy.deepcopy(next(i for i in self.fixture["items"] if i["type"] == "image")); raw["id"] = "invalid-image"
        raw["content"]["image"] = "../fixtures.json"
        self.assert_error("validation_error", lambda: self.store.import_items(self.policy, {"items": [raw]}))
        raw = copy.deepcopy(self.fixture["items"][0]); raw["id"] = "group-a"; raw["source"]["conversation_id"] = "group-locked"
        second = copy.deepcopy(raw); second["id"] = "group-b"; second["split"] = "holdout"
        self.assert_error("split_conflict", lambda: self.store.import_items(self.policy, {"items": [raw, second]}))
        self.assert_error("not_found", lambda: self.store.get_item("group-a", self.policy))

    def test_relevance_pins_keep_ordinal_schema_and_seed_is_repeat_safe(self):
        relevance = next(i for i in self.fixture["items"] if i["targets"][0]["id"] == "relevance")
        item = self.store.get_item(relevance["id"], self.policy)
        self.assertEqual("pin", item["type"])
        reviewed = self.store.review_item(item["id"], self.policy, self.payload(item, labels={"relevance": "4"}))
        self.assertEqual({"relevance": "4"}, reviewed["independent_labels"])
        self.assertFalse(self.store.seed_fixture(self.fixture))
        self.assertEqual(reviewed, self.store.get_item(item["id"], self.policy))

    def test_sqlite_enforces_append_only_events_and_immutable_epochs(self):
        with sqlite3.connect(self.db) as conn:
            with self.assertRaises(sqlite3.IntegrityError): conn.execute("DELETE FROM events")
            with self.assertRaises(sqlite3.IntegrityError): conn.execute("UPDATE epochs SET manifest='{}'")

    def test_new_adjudication_draft_revokes_previous_staged_approval(self):
        approved = self.approve_ready()
        draft = self.store.review_item(approved["id"], self.policy,
            self.payload(approved, phase="adjudication", decision="save", labels={"input": "Violation", "output": "No violation"}))
        self.assertEqual("open", draft["status"])
        self.assertFalse(draft["gold_eligible"])
        self.assertTrue(all(t["staged_label"] is None for t in draft["targets"]))
        self.assert_error("no_changes", lambda: self.store.publish_epoch(self.policy,
            {"expected_epoch": self.fixture["initial_epoch"], "note": "Old approval must not publish"}))

    def test_disputed_and_route_stale_gold_are_quarantined_from_new_epochs(self):
        epoch = self.fixture["initial_epoch"]
        original = self.store.export_epoch(epoch)
        ready = self.store.get_item(self.ready_id, self.policy)
        disputed = self.store.review_item(ready["id"], self.policy,
            self.payload(ready, phase="adjudication", decision="escalate", labels={"input": "Uncertain", "output": "No violation"}))
        stale_id = next(i["id"] for i in self.fixture["items"] if i["type"] == "pin")
        stale = self.store.get_item(stale_id, self.policy)
        self.store.route_item(stale_id, self.policy, {"base_revision": stale["revision"], "nodes": [], "reason": "The route did not cover all applicable policies"})
        publish_id = next(i["id"] for i in self.fixture["items"] if i["type"] == "pin" and i["id"] != stale_id)
        other = self.store.get_item(publish_id, self.policy)
        other = self.store.review_item(publish_id, self.policy, self.payload(other))
        self.store.review_item(publish_id, self.policy, self.payload(other, phase="adjudication", decision="approve"))
        newer = self.store.publish_epoch(self.policy, {"expected_epoch": epoch, "note": "Publish unrelated approved labels"})
        ids = {i["id"] for i in self.store.export_epoch(newer["epoch"])["items"]}
        self.assertNotIn(disputed["id"], ids); self.assertNotIn(stale_id, ids)
        self.assertIn(publish_id, ids)
        self.assertEqual(original, self.store.export_epoch(epoch))
        self.assertIn(disputed["id"], {i["id"] for i in original["items"]})
        # Explicit resolution and a new publication are both necessary to restore eligibility.
        resolved = self.store.review_item(disputed["id"], self.policy,
            self.payload(disputed, phase="adjudication", decision="approve"))
        self.assertFalse(resolved["gold_eligible"])
        newest = self.store.publish_epoch(self.policy, {"expected_epoch": newer["epoch"], "note": "Policy resolved the boundary"})
        self.assertIn(disputed["id"], {i["id"] for i in self.store.export_epoch(newest["epoch"])["items"]})

    def test_import_rejects_unknown_policy_versions_and_nested_unsafe_media(self):
        raw = copy.deepcopy(self.fixture["items"][0]); raw["id"] = "unknown-policy"
        raw["policy_version"] = "never-existed"
        self.assert_error("policy_version_mismatch", lambda: self.store.import_items(self.policy, {"items": [raw]}))
        raw = copy.deepcopy(self.fixture["items"][0]); raw["id"] = "unsafe-message-image"
        raw["content"]["messages"][0]["image"] = "file:///etc/passwd"
        self.assert_error("validation_error", lambda: self.store.import_items(self.policy, {"items": [raw]}))
        raw["content"]["messages"][0]["image"] = "assets/conversation.svg"
        raw["content"]["messages"][0]["attachments"] = [{"image": "file:///etc/passwd"}]
        self.assert_error("validation_error", lambda: self.store.import_items(self.policy, {"items": [raw]}))
        self.assert_error("not_found", lambda: self.store.get_item("unsafe-message-image", self.policy))

    def test_recorded_trace_is_blind_then_visible_without_summary_leakage(self):
        ready = self.store.get_item(self.ready_id, self.policy)
        recorded = next(i for i in self.fixture["items"] if i["id"] == self.ready_id)["route"]["trace"]
        expected = {**recorded, "selected_not_served_ids": [], "unserved_cited_ids": [], "missing_pinned_ids": []}
        self.assertEqual(expected, ready["route"]["trace"])
        other = self.store.get_item(self.ready_id, self.ops)
        self.assertNotIn("trace", other["route"])
        visible = self.store.review_item(self.ready_id, self.ops, self.payload(other))
        self.assertEqual(expected, visible["route"]["trace"])
        self.assertTrue(all("trace" not in item["route"] for item in self.store.list_items(self.policy)["items"]))

    def test_route_correction_preserves_historical_trace_and_graph_in_epochs(self):
        initial_epoch = self.fixture["initial_epoch"]
        original = self.store.export_epoch(initial_epoch)
        self.assertEqual(self.fixture["graph"], self.store.bootstrap(self.policy)["graph"])
        self.assertEqual(graph_digest(self.fixture["graph"]), original["graph_hash"])
        ready = self.store.get_item(self.ready_id, self.policy)
        historical = copy.deepcopy(ready["route"]["trace"])
        corrected = self.store.route_item(self.ready_id, self.policy, {"base_revision": ready["revision"], "nodes": ["HAR-ROOT", "HAR-3"], "reason": "The quoted-content boundary also applies"})
        self.assertEqual(historical, corrected["route"]["trace"])
        self.assertTrue(corrected["route"]["trace_stale"])
        self.assertTrue(corrected["route"]["current_route_changed"])
        self.assertEqual(["HAR-ROOT", "HAR-3"], corrected["route"]["nodes"])
        hidden = self.store.get_item(self.ready_id, self.ops)
        self.assertNotIn("trace", hidden["route"])
        self.assertNotIn("before", hidden["events"][0])
        self.approve_ready()
        published = self.store.publish_epoch(self.policy, {"expected_epoch": initial_epoch, "note": "Policy confirmed the wider route"})
        manifest = self.store.export_epoch(published["epoch"])
        self.assertEqual(self.fixture["graph"], manifest["graph"])
        self.assertEqual(self.fixture["graph"]["version"], manifest["graph_version"])
        row = next(i for i in manifest["items"] if i["id"] == self.ready_id)
        self.assertEqual(historical, row["route"]["trace"])
        self.assertTrue(row["route"]["trace_stale"])
        self.assertEqual(original, self.store.export_epoch(initial_epoch))

    def test_candidate_trace_rejects_wrong_graph_version_references_and_probabilities(self):
        source = next(i for i in self.fixture["items"] if i["id"] == self.ready_id)
        changes = [
            ("graph_hash", "0" * 64), ("policy_version", "7.1"),
            ("router_version", "unrelated-router"), ("cited_ids", ["not-a-node"]),
            ("served_ids", ["not-a-node"]), ("retained_score_mass", 1.5),
        ]
        for field, value in changes:
            with self.subTest(field=field):
                raw = copy.deepcopy(source); raw["id"] = "invalid-trace"
                raw["route"]["trace"][field] = value
                self.assert_error("route_trace_invalid", lambda: self.store.import_items(self.policy, {"items": [raw]}))
        raw = copy.deepcopy(source); raw["id"] = "valid-trace"
        result = self.store.import_items(self.policy, {"graph": self.fixture["graph"], "items": [raw]})
        self.assertEqual(1, result["imported"])
        hidden = self.store.get_item(raw["id"], self.policy)
        self.assertNotIn("trace", hidden["route"])
        altered = copy.deepcopy(self.fixture["graph"])
        altered["nodes"][0]["body"] += " changed"
        self.assert_error("graph_validation_error", lambda: self.store.import_items(self.policy, {"graph": altered, "items": [raw]}))

    def test_shadow_selection_differs_from_served_context_without_losing_evidence(self):
        raw = copy.deepcopy(next(i for i in self.fixture["items"] if i["id"] == self.ready_id))
        raw["id"] = "shadow-context-difference"
        trace = raw["route"]["trace"]
        trace["served_ids"] = ["HAR-ROOT", "HAR-3"]
        trace["cited_ids"] = ["HAR-3"]
        self.store.import_items(self.policy, {"items": [raw]})
        hidden = self.store.get_item(raw["id"], self.policy)
        visible = self.store.review_item(raw["id"], self.policy, self.payload(hidden))
        result = visible["route"]["trace"]
        self.assertEqual(trace["selected_ids"], result["selected_ids"])
        self.assertEqual(trace["served_ids"], result["served_ids"])
        self.assertEqual(["HAR-2"], result["selected_not_served_ids"])
        self.assertEqual([], result["unserved_cited_ids"])
        self.assertEqual([], result["missing_pinned_ids"])

    def test_unserved_citations_and_missing_pins_are_preserved_and_flagged(self):
        raw = copy.deepcopy(next(i for i in self.fixture["items"] if i["id"] == self.ready_id))
        raw["id"] = "routing-audit-anomalies"
        trace = raw["route"]["trace"]
        trace["served_ids"] = ["HAR-3"]
        # Imported diagnostics are not authoritative; derive them from the sets.
        trace["unserved_cited_ids"] = []
        trace["missing_pinned_ids"] = []
        self.store.import_items(self.policy, {"items": [raw]})
        hidden = self.store.get_item(raw["id"], self.policy)
        self.assertNotIn("trace", hidden["route"])
        visible = self.store.review_item(raw["id"], self.policy, self.payload(hidden))
        result = visible["route"]["trace"]
        self.assertEqual(trace["cited_ids"], result["cited_ids"])
        self.assertEqual(trace["pinned_ids"], result["pinned_ids"])
        self.assertEqual(["HAR-2"], result["unserved_cited_ids"])
        self.assertEqual(["HAR-ROOT"], result["missing_pinned_ids"])
        self.assertEqual(["HAR-2", "HAR-ROOT"], result["selected_not_served_ids"])

    def test_graphless_fixture_remains_supported_but_trace_requires_graph(self):
        fixture = copy.deepcopy(self.fixture)
        fixture.pop("graph")
        for item in fixture["items"]: item["route"].pop("trace", None)
        store = LabelingStore(Path(self.temp.name) / "no-graph.sqlite3")
        store.seed_fixture(fixture)
        self.assertIsNone(store.bootstrap(self.policy)["graph"])
        self.assertNotIn("graph", store.export_epoch(fixture["initial_epoch"]))
        raw = copy.deepcopy(self.fixture["items"][0]); raw["id"] = "needs-graph"
        self.assert_error("route_trace_invalid", lambda: store.import_items(self.policy, {"items": [raw]}))


class PolicyGraphAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.version_dir = self.root / "v61"
        self.version_dir.mkdir()
        self.version = "DeciderAgent.9ea09b9as.v61"
        self.overlay = self.root / "_code" / "abc123"
        self.overlay.mkdir(parents=True)
        self.manifest = {"schema": 1, "agent_type": "DeciderAgent", "config_hash": "9ea09b9as", "version": 61,
            "root_id": "DA.root", "compile_order": {"strategy_directives": ["DA.rule"]},
            "source_sha256": {"strategy_directives": hashlib.sha256(b"Rule\n").hexdigest(), "memory": None},
            "code": {"sha": "abc123", "dir": "../_code/abc123"}}
        self.write_manifest()
        self.write_node(self.version_dir, "DA.root", "root", None, "Root\n")
        self.write_node(self.version_dir, "DA.rule", "rule", "DA.root", "Rule\n", authored=True)
        self.write_node(self.overlay, "DA.code.guard", "code", "DA.root", "raise RuntimeError('This text must never execute')\n", version="code@abc123")
        edges = [{"source_node_id": "DA.rule", "target_node_id": "DA.root", "edge_type": "subtype_of", "confidence": 1.0, "provenance": "derived:hierarchy", "version": self.version}]
        (self.version_dir / "edges.json").write_text(json.dumps(edges), encoding="utf-8")

    def write_manifest(self):
        (self.version_dir / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def write_node(self, directory, node_id, kind, parent, body, *, version=None, authored=False):
        fields = {"id": node_id, "version": version or self.version, "agent": "DeciderAgent", "title": node_id,
            "node_type": kind, "parent": parent, "locked": kind == "code", "provenance": "test",
            "sep_before": "", "sep_after": "\n\n", "body_sha256": hashlib.sha256(body.encode()).hexdigest(), "tags": []}
        lines = [f"{key}: {json.dumps(value)}" for key, value in fields.items()]
        if authored:
            lines += ["edges:", '  - {type: clarifies, to: DA.root, via: "reason, with a comma"}']
        (directory / f"{node_id}.md").write_bytes(("---\n" + "\n".join(lines) + "\n---\n" + body).encode())

    def test_native_frontmatter_typed_edges_and_overlays_keep_version_provenance(self):
        graph = load_version_directory(self.version_dir)
        self.assertEqual(self.version, graph["version"])
        self.assertEqual(self.manifest["source_sha256"], graph["upstream_source_sha256"])
        self.assertEqual(3, len(graph["nodes"]))
        code = next(n for n in graph["nodes"] if n["node_type"] == "code")
        self.assertEqual("code@abc123", code["upstream_node_version"])
        self.assertEqual(self.version, code["version"])
        self.assertEqual("raise RuntimeError('This text must never execute')\n", code["body"])
        self.assertEqual({"subtype_of", "clarifies"}, {e["edge_type"] for e in graph["edges"]})
        self.assertEqual("reason, with a comma", next(e for e in graph["edges"] if e["edge_type"] == "clarifies")["via"])
        self.assertEqual(graph_digest(graph), graph["source_sha256"])

    def test_native_adapter_rejects_overlay_escape_and_corrupted_body(self):
        self.manifest["code"]["dir"] = str(self.overlay)
        self.write_manifest()
        with self.assertRaisesRegex(GraphValidationError, "Absolute"):
            load_version_directory(self.version_dir)
        self.manifest["code"]["dir"] = "../_code/abc123"
        self.write_manifest()
        file = self.version_dir / "DA.rule.md"
        file.write_bytes(file.read_bytes() + b"tampered")
        with self.assertRaisesRegex(GraphValidationError, "body_sha256"):
            load_version_directory(self.version_dir)

    def test_graph_contract_rejects_hash_unknown_edge_and_version_mismatch(self):
        graph = load_version_directory(self.version_dir)
        for mutate in (
            lambda g: g.update(source_sha256="0" * 64),
            lambda g: g["edges"][0].update(target_node_id="missing"),
            lambda g: g["nodes"][0].update(version="other"),
        ):
            candidate = copy.deepcopy(graph); mutate(candidate)
            with self.assertRaises(GraphValidationError): normalize_policy_graph(candidate)


if __name__ == "__main__":
    unittest.main()
