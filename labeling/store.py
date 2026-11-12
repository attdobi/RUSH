"""Transactional review history, local gold snapshots and an APO handoff outbox.

The local database is an executable connector boundary, not a production GDS.
No method in this module invokes a model, spends money or promotes a prompt.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sqlite3
from typing import Any, Callable
from urllib.parse import urlsplit
import uuid

from .policy_graph import GraphValidationError, normalize_policy_graph, validate_route_trace


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def clone(value: Any) -> Any:
    return json.loads(canonical(value))


@dataclass(frozen=True)
class Actor:
    id: str
    name: str
    role: str

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id or not isinstance(self.name, str) or not self.name or not isinstance(self.role, str) or self.role not in {"policy", "sme", "bpo", "ops"}:
            raise ValueError("An actor needs an id, name and policy/sme/bpo/ops role")


class LabelingError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message

    def response(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": self.message}}


def require(condition: Any, status: int, code: str, message: str) -> None:
    if not condition:
        raise LabelingError(status, code, message)


def nonempty(value: Any, field: str, *, max_length: int = 4000) -> str:
    require(isinstance(value, str) and bool(value.strip()), 422, "validation_error", f"{field} is required")
    require(len(value) <= max_length, 422, "validation_error", f"{field} is too long")
    return value.strip()


def safe_image_reference(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    if parsed.scheme:
        return parsed.scheme == "https" and bool(parsed.hostname) and not parsed.username and not parsed.password
    if parsed.netloc or parsed.query or parsed.fragment or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and len(path.parts) > 1 and path.parts[0] == "assets" and ".." not in path.parts


SCHEMA = """
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS items (id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS independent_reviews (
    item_id TEXT NOT NULL REFERENCES items(id), actor_id TEXT NOT NULL,
    labels TEXT NOT NULL, created_at TEXT NOT NULL, PRIMARY KEY (item_id, actor_id)
);
CREATE TABLE IF NOT EXISTS events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT, item_id TEXT REFERENCES items(id),
    actor_id TEXT NOT NULL, body TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'review events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'review events are append-only'); END;
CREATE TABLE IF NOT EXISTS epochs (
    epoch INTEGER PRIMARY KEY, record TEXT NOT NULL, manifest TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS epochs_no_update BEFORE UPDATE ON epochs
BEGIN SELECT RAISE(ABORT, 'gold epochs are immutable'); END;
CREATE TRIGGER IF NOT EXISTS epochs_no_delete BEFORE DELETE ON epochs
BEGIN SELECT RAISE(ABORT, 'gold epochs are immutable'); END;
CREATE TABLE IF NOT EXISTS outbox (id TEXT PRIMARY KEY, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS idempotency (
    actor_id TEXT NOT NULL, key TEXT NOT NULL, operation TEXT NOT NULL,
    request_hash TEXT NOT NULL, response TEXT NOT NULL, PRIMARY KEY (actor_id, key)
);
"""


class LabelingStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._memory_anchor = None
        self._uri = self.db_path == ":memory:"
        if self._uri:
            self.db_path = f"file:rush-labeling-{uuid.uuid4().hex}?mode=memory&cache=shared"
            self._memory_anchor = sqlite3.connect(self.db_path, uri=True)
        else:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connection(self):
        conn = sqlite3.connect(self.db_path, timeout=15, uri=self._uri)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        if not self._uri:
            conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _meta(conn, key, default=None):
        row = conn.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    @staticmethod
    def _set_meta(conn, key, value):
        conn.execute("INSERT INTO metadata(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, canonical(value)))

    @staticmethod
    def _item(conn, item_id: str) -> dict[str, Any]:
        row = conn.execute("SELECT body FROM items WHERE id=?", (item_id,)).fetchone()
        require(row is not None, 404, "not_found", "Review item not found")
        return json.loads(row[0])

    @staticmethod
    def _save_item(conn, item):
        conn.execute("UPDATE items SET body=? WHERE id=?", (canonical(item), item["id"]))

    @staticmethod
    def _all_items(conn):
        return [json.loads(row[0]) for row in conn.execute("SELECT body FROM items ORDER BY id")]

    @staticmethod
    def _check_revision(item, body):
        revision = body.get("base_revision")
        require(isinstance(revision, int) and not isinstance(revision, bool), 422, "validation_error", "base_revision must be an integer")
        require(revision == item["revision"], 409, "revision_conflict", "This item changed. Refresh it before saving your review.")

    @staticmethod
    def _check_assignee(item, actor):
        require(item.get("assignee") in (None, actor.id), 409, "already_claimed", "This item is assigned to another reviewer.")

    @staticmethod
    def _check_role(actor, roles):
        require(actor.role in roles, 403, "forbidden", "Your reviewer role cannot perform this action.")

    def _mutate(self, actor: Actor, operation: str, body: dict, key: str | None, action: Callable) -> Any:
        require(isinstance(body, dict), 422, "validation_error", "Request body must be an object")
        if key is not None:
            nonempty(key, "Idempotency-Key", max_length=200)
        digest = hashlib.sha256(canonical(body).encode()).hexdigest()
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if key:
                    previous = conn.execute("SELECT operation,request_hash,response FROM idempotency WHERE actor_id=? AND key=?", (actor.id, key)).fetchone()
                    if previous:
                        require(previous[0] == operation and previous[1] == digest, 409, "idempotency_conflict", "This idempotency key was used for another request.")
                        conn.commit()
                        return json.loads(previous[2])
                result = action(conn)
                if key:
                    conn.execute("INSERT INTO idempotency VALUES(?,?,?,?,?)", (actor.id, key, operation, digest, canonical(result)))
                conn.commit()
                return result
            except Exception:
                conn.rollback()
                raise

    @staticmethod
    def _event(conn, item_id, actor: Actor, event_type: str, **fields):
        event = {"item_id": item_id, "type": event_type, "action": event_type,
                 "actor": asdict(actor), "actor_id": actor.id, "role": actor.role,
                 "created_at": utcnow(), **fields}
        event["timestamp"] = event["created_at"]
        cursor = conn.execute("INSERT INTO events(item_id,actor_id,body) VALUES(?,?,?)", (item_id, actor.id, canonical(event)))
        event["id"] = f"evt-{cursor.lastrowid:06d}"
        return event

    @staticmethod
    def _read_events(conn, item_id=None):
        if item_id is None:
            rows = conn.execute("SELECT event_id,body FROM events ORDER BY event_id DESC LIMIT 1000")
        else:
            rows = conn.execute("SELECT event_id,body FROM events WHERE item_id=? ORDER BY event_id DESC", (item_id,))
        result = []
        for row in rows:
            event = json.loads(row[1]); event["id"] = f"evt-{row[0]:06d}"; result.append(event)
        return result

    @staticmethod
    def _redact_event(event):
        safe = clone(event)
        for key in ("labels", "before", "after", "rationale", "comment", "policy_refs"):
            safe.pop(key, None)
        if "targets" in safe:
            safe["targets"] = [{"id": x["id"], "category": x["category"]} for x in safe["targets"]]
        return safe

    def _view(self, conn, item, actor):
        result = clone(item)
        row = conn.execute("SELECT labels FROM independent_reviews WHERE item_id=? AND actor_id=?", (item["id"], actor.id)).fetchone()
        labels = json.loads(row[0]) if row else None
        result["guidance_visible"] = labels is not None
        result["independent_labels"] = labels
        result["events"] = self._read_events(conn, item["id"])
        result.pop("seed_independent_labels", None)
        if labels is None:
            result["route"].pop("override_reason", None)
            result["route"].pop("trace", None)
            for target in result["targets"]:
                target["ai"] = None
                target["gold"] = None
                target["staged_label"] = None
            result["events"] = [self._redact_event(x) for x in result["events"]]
        return result

    def _known_policies(self, conn):
        return {p["id"]: p for p in self._meta(conn, "policies", [])}

    def _validate_graph(self, conn, graph):
        try:
            graph = normalize_policy_graph(graph)
        except GraphValidationError as exc:
            raise LabelingError(422, "graph_validation_error", str(exc)) from exc
        catalog = {node["id"]: node for node in graph["nodes"]}
        policies = self._known_policies(conn)
        require(set(policies).issubset(catalog), 422, "graph_validation_error", "Configured policy references must exist in the graph")
        require(all(catalog[node_id]["version"] == str(policy.get("version")) and catalog[node_id]["body"] == policy.get("body") for node_id, policy in policies.items()), 422, "graph_validation_error", "Graph node content must match the configured policy catalog")
        return graph

    def _validate_item(self, conn, raw, *, trusted=False):
        require(isinstance(raw, dict), 422, "validation_error", "Each imported item must be an object")
        item_id = nonempty(raw.get("id"), "id", max_length=120)
        require(bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", item_id)), 422, "validation_error", "Item id must use letters, digits, dots, underscores or hyphens")
        kind = raw.get("type")
        require(isinstance(kind, str) and kind in {"image", "pin", "assistant", "search"}, 422, "validation_error", "Unsupported item type")
        categories = {c["id"] for c in self._meta(conn, "categories", [])}
        policies = self._known_policies(conn)
        category = raw.get("category")
        require(isinstance(category, str) and category in categories, 422, "validation_error", "Unknown item category")
        project = nonempty(raw.get("project"), "project", max_length=100)
        require(project in self._meta(conn, "projects", []), 422, "validation_error", "Unknown project")
        split = raw.get("split", "train")
        require(isinstance(split, str) and split in {"train", "holdout"}, 422, "validation_error", "split must be train or holdout")
        policy_version = nonempty(raw.get("policy_version"), "policy_version", max_length=100)
        require(any(str(p.get("version")) == policy_version and p.get("category") == category for p in policies.values()), 422, "policy_version_mismatch", "Item policy version must match the configured category policy")
        content = raw.get("content")
        require(isinstance(content, dict), 422, "validation_error", "content must be an object")
        allowed_content = {k: clone(v) for k, v in content.items() if k in {"image", "title", "description", "messages", "query", "url", "language"}}
        require(not any(k in content for k in ("attachments", "images")), 422, "validation_error", "Use content.image or message.image for supported image attachments")
        for key in ("title", "description", "query", "language"):
            require(key not in content or isinstance(content[key], str), 422, "validation_error", f"content.{key} must be text")
        if content.get("url") is not None:
            require(isinstance(content["url"], str) and safe_image_reference(content["url"]) and urlsplit(content["url"]).scheme == "https", 422, "validation_error", "Content links must use HTTPS")
        if content.get("image") is not None:
            require(safe_image_reference(content["image"]), 422, "validation_error", "Images must use a safe local assets path or HTTPS URL")
        if kind in {"image", "pin"}:
            require(bool(content.get("image")), 422, "validation_error", "Image and pin items require an image")
        if kind == "assistant":
            messages = content.get("messages")
            require(isinstance(messages, list) and bool(messages), 422, "validation_error", "Assistant items need original messages")
            require(all(isinstance(m, dict) and isinstance(m.get("role"), str) and m.get("role") in {"user", "assistant", "system"} and isinstance(m.get("text"), str) for m in messages), 422, "validation_error", "Invalid conversation message")
            require({"user", "assistant"}.issubset({m["role"] for m in messages}), 422, "validation_error", "Assistant items need user and assistant messages")
            for message in messages:
                require(set(message).issubset({"role", "text", "image"}), 422, "validation_error", "Messages support role, text and one optional image reference")
                if message.get("image") is not None:
                    require(safe_image_reference(message["image"]), 422, "validation_error", "Message images must use a safe local assets path or HTTPS URL")
            allowed_content["messages"] = clone(messages)
        elif content.get("messages") is not None:
            raise LabelingError(422, "validation_error", "Messages belong only to assistant items")
        targets = raw.get("targets")
        require(isinstance(targets, list) and bool(targets), 422, "validation_error", "Each item needs review targets")
        ids = [t.get("id") if isinstance(t, dict) else None for t in targets]
        require(all(isinstance(value, str) for value in ids), 422, "validation_error", "Every target needs a string id")
        relevance = kind == "search" or (kind in {"image", "pin"} and ids == ["relevance"])
        expected = {"input", "output"} if kind == "assistant" else ({"relevance"} if relevance else {"content"})
        require(len(ids) == len(set(ids)) and set(ids) == expected, 422, "validation_error", "All required review targets must be present exactly once")
        clean_targets = []
        for target in targets:
            choices = target.get("choices")
            require(isinstance(choices, list) and len(choices) >= 2 and all(isinstance(c, str) and c for c in choices), 422, "validation_error", "Targets need valid label choices")
            require(len(set(choices)) == len(choices), 422, "validation_error", "Label choices cannot repeat")
            require(isinstance(target.get("category"), str) and target.get("category") in categories, 422, "validation_error", "Unknown target category")
            require(any(str(p.get("version")) == policy_version and p.get("category") == target["category"] for p in policies.values()), 422, "policy_version_mismatch", "Target policy version does not match the configured catalog")
            if target["id"] == "relevance":
                require(set(choices) == {"1", "2", "3", "4", "5", "Uncertain"}, 422, "validation_error", "Search relevance requires choices 1 through 5 and Uncertain")
            ai = target.get("ai")
            if ai is not None:
                require(isinstance(ai, dict) and ai.get("label") in choices, 422, "validation_error", "AI decision must use the target label schema")
                confidence = ai.get("confidence")
                require(confidence is None or (isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and math.isfinite(confidence) and 0 <= confidence <= 1), 422, "validation_error", "AI confidence must be between zero and one")
                refs = ai.get("policy_refs", [])
                require(isinstance(refs, list) and all(isinstance(x, str) and x in policies for x in refs), 422, "validation_error", "AI rationale has unknown policy references")
                require(all(str(policies[x].get("version")) == policy_version for x in refs), 422, "policy_version_mismatch", "AI evidence cites a different policy version")
                ai = {k: clone(v) for k,v in ai.items() if k in {"label", "confidence", "confidence_kind", "model", "prompt_version", "rationale", "policy_refs", "evidence"}}
            gold = target.get("gold") if trusted else None
            if gold is not None:
                require(isinstance(gold, dict) and gold.get("label") in choices, 422, "validation_error", "Invalid seed gold label")
                gold = {"label": gold["label"], "revision": int(gold.get("revision", 1)), "policy_version": policy_version}
            clean_targets.append({"id": target["id"], "name": nonempty(target.get("name", target["id"]), "target name", max_length=100), "category": target["category"], "choices": choices, "ai": ai, "gold": gold, "staged_label": None})
        route = raw.get("route") or {}
        require(isinstance(route, dict), 422, "validation_error", "route must be an object")
        nodes = route.get("nodes", [])
        require(isinstance(nodes, list) and all(isinstance(x, str) and x in policies for x in nodes), 422, "validation_error", "Unknown routed policy node")
        require(all(str(policies[x].get("version")) == policy_version for x in nodes), 422, "policy_version_mismatch", "Routed nodes must match the item policy version")
        confidence = route.get("confidence")
        require(confidence is None or (isinstance(confidence, (int, float)) and not isinstance(confidence, bool) and math.isfinite(confidence) and 0 <= confidence <= 1), 422, "validation_error", "Route confidence must be between zero and one")
        clean_route = {"nodes": list(dict.fromkeys(nodes)), "version": str(route.get("version", "unrouted")), "confidence": route.get("confidence"), "fallback": bool(route.get("fallback", not nodes))}
        if route.get("trace") is not None:
            try:
                clean_route["trace"] = validate_route_trace(route["trace"], self._meta(conn, "graph"), policy_version=policy_version, router_version=clean_route["version"])
            except GraphValidationError as exc:
                raise LabelingError(422, "route_trace_invalid", str(exc)) from exc
        priority = raw.get("priority") or {}
        require(isinstance(priority, dict), 422, "validation_error", "priority must be an object")
        for name in ("score", "gradient", "disagreement", "consensus"):
            value = priority.get(name, 0)
            require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0, 422, "validation_error", "Ranking signals must be finite nonnegative numbers")
        source = raw.get("source") or {"name": "Imported candidate", "sampling": "manual_import"}
        require(isinstance(source, dict), 422, "validation_error", "source must be an object")
        source = {k: clone(v) for k,v in source.items() if k in {"name", "sampling", "run_id", "conversation_id", "group_id", "sampled_at", "item_id", "source_uri", "source_version"}}
        for key in ("conversation_id", "group_id"):
            if source.get(key) is not None:
                nonempty(source[key], key, max_length=200)
        return {"id": item_id, "project": project, "type": kind,
                "title": nonempty(raw.get("title"), "title", max_length=300),
                "description": str(raw.get("description", ""))[:4000], "category": category,
                "policy_version": policy_version, "revision": 1, "status": "open", "split": split,
                "assignee": None, "created_at": raw.get("created_at", utcnow()) if trusted else utcnow(),
                "source": clone(source), "priority": clone(priority),
                "route": clean_route,
                "content": allowed_content, "content_version": str(raw.get("content_version") or hashlib.sha256(canonical(allowed_content).encode()).hexdigest()),
                "targets": clean_targets, "gold_eligible": trusted and split == "train" and all(t.get("gold") and t["gold"]["label"].casefold() != "uncertain" for t in clean_targets)}

    def _check_group_split(self, conn, item):
        groups = {key: item["source"].get(key) for key in ("conversation_id", "group_id") if item["source"].get(key)}
        if not groups:
            return
        for existing in self._all_items(conn):
            same_group = any(existing.get("source", {}).get(key) == value for key,value in groups.items())
            require(not same_group or existing["split"] == item["split"], 422, "split_conflict", "Related items must remain together in one train or holdout split")

    @staticmethod
    def _validate_labels(item, labels):
        require(isinstance(labels, dict), 422, "validation_error", "labels must be an object")
        require(set(labels) == {t["id"] for t in item["targets"]}, 422, "validation_error", "Submit a decision for every required target")
        require(all(labels[t["id"]] in t["choices"] for t in item["targets"]), 422, "validation_error", "A submitted label is not valid for its target")
        return clone(labels)

    def seed_fixture(self, fixture: dict) -> bool:
        """Initialize one isolated demo database once; never overwrite existing work."""
        require(isinstance(fixture, dict), 422, "validation_error", "Fixture must be an object")
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if self._meta(conn, "initialized", False):
                    conn.commit(); return False
                require(conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 0, 409, "already_initialized", "Cannot seed a nonempty database")
                for key in ("categories", "projects", "policies"):
                    require(isinstance(fixture.get(key), list) and bool(fixture[key]), 422, "validation_error", f"Fixture needs {key}")
                    self._set_meta(conn, key, fixture[key])
                if fixture.get("graph") is not None:
                    self._set_meta(conn, "graph", self._validate_graph(conn, fixture["graph"]))
                viewer = fixture.get("viewer", {"id": "demo-policy", "name": "Local demo reviewer", "role": "policy"})
                actor = Actor(**viewer)
                self._set_meta(conn, "viewer", asdict(actor))
                self._set_meta(conn, "policy_owner", actor.id)
                items = fixture.get("items", [])
                require(isinstance(items, list), 422, "validation_error", "Fixture items must be a list")
                for raw in items:
                    item = self._validate_item(conn, raw, trusted=True)
                    self._check_group_split(conn, item)
                    conn.execute("INSERT INTO items VALUES(?,?)", (item["id"], canonical(item)))
                    seed_labels = raw.get("seed_independent_labels")
                    if seed_labels is not None:
                        labels = self._validate_labels(item, seed_labels)
                        conn.execute("INSERT INTO independent_reviews VALUES(?,?,?,?)", (item["id"], actor.id, canonical(labels), utcnow()))
                        self._event(conn, item["id"], actor, "independent_review", phase="independent", ai_visible=False, labels=labels,
                                    targets=[{"id": t["id"], "category": t["category"], "before": (t.get("gold") or {}).get("label"), "after": labels[t["id"]]} for t in item["targets"]],
                                    policy_version=item["policy_version"], summary="Synthetic first pass provided for this example", seeded=True)
                epoch = fixture.get("initial_epoch", 42)
                require(isinstance(epoch, int) and not isinstance(epoch, bool) and epoch >= 1, 422, "validation_error", "Invalid initial epoch")
                self._set_meta(conn, "epoch", epoch)
                self._make_epoch(conn, epoch, actor, 0, "Initial synthetic gold snapshot")
                self._set_meta(conn, "initialized", True)
                conn.commit(); return True
            except Exception:
                conn.rollback(); raise

    def demo_actor(self) -> Actor:
        with self._connection() as conn:
            return Actor(**self._meta(conn, "viewer", {"id": "demo-policy", "name": "Local demo reviewer", "role": "policy"}))

    def bootstrap(self, actor: Actor, *, mode="local-demo"):
        with self._connection() as conn:
            items = self._all_items(conn)
            stats = {"total": len(items), **{s: sum(i["status"] == s for i in items) for s in ("open", "escalated", "approved", "published")}}
            stats["gold_items"] = sum(self._gold_eligible(i) for i in items)
            return {"viewer": asdict(actor), "mode": mode, "epoch": self._meta(conn, "epoch", 0), "stats": stats,
                    **{k: self._meta(conn, k, []) for k in ("categories", "projects", "policies")},
                    "graph": self._meta(conn, "graph"),
                    "epochs": [json.loads(r[0]) for r in conn.execute("SELECT record FROM epochs ORDER BY epoch DESC")],
                    "runs": [json.loads(r[0]) for r in conn.execute("SELECT body FROM outbox ORDER BY rowid DESC")],
                    "integrations": {"gds": "local", "apo": "outbox"}}

    def list_items(self, actor: Actor, filters: dict | None = None):
        filters = filters or {}
        with self._connection() as conn:
            items = self._all_items(conn)
        for key in ("type", "category", "status", "project"):
            if filters.get(key) and filters[key] != "all":
                items = [i for i in items if i.get(key) == filters[key]]
        if filters.get("q"):
            q = str(filters["q"]).casefold()
            items = [i for i in items if q in " ".join(str(i.get(k, "")) for k in ("id", "title", "description", "project", "category")).casefold()]
        items.sort(key=lambda i: (-float(i.get("priority", {}).get("score", 0)), i["id"]))
        keys = {"id", "title", "description", "project", "type", "category", "policy_version", "revision", "status", "split", "assignee", "created_at", "priority", "source", "route", "escalation_owner", "gold_eligible"}
        summaries = [{**{k: clone(v) for k,v in i.items() if k in keys}, "content": {k: i["content"][k] for k in ("image", "title") if k in i["content"]}, "target_count": len(i["targets"])} for i in items]
        for summary in summaries:
            summary["route"].pop("override_reason", None)
            summary["route"].pop("trace", None)
        return {"items": summaries, "total": len(summaries)}

    def get_item(self, item_id: str, actor: Actor):
        with self._connection() as conn:
            return self._view(conn, self._item(conn, item_id), actor)

    def claim_item(self, item_id: str, actor: Actor, body: dict, *, idempotency_key=None):
        def action(conn):
            item = self._item(conn, item_id); self._check_revision(item, body); self._check_assignee(item, actor)
            item["assignee"] = actor.id; item["revision"] += 1
            self._save_item(conn, item)
            self._event(conn, item_id, actor, "claim", summary=f"Claimed by {actor.name}", revision=item["revision"])
            return self._view(conn, item, actor)
        return self._mutate(actor, f"claim:{item_id}", body, idempotency_key, action)

    def review_item(self, item_id: str, actor: Actor, body: dict, *, idempotency_key=None):
        def action(conn):
            item = self._item(conn, item_id); self._check_revision(item, body); self._check_assignee(item, actor)
            phase, decision = body.get("phase"), body.get("decision")
            require(isinstance(phase, str) and phase in {"independent", "adjudication"}, 422, "validation_error", "Invalid review phase")
            require(isinstance(decision, str) and decision in {"save", "approve", "escalate"}, 422, "validation_error", "Invalid review decision")
            labels = self._validate_labels(item, body.get("labels"))
            rationale = nonempty(body.get("rationale"), "rationale")
            refs = body.get("policy_refs", [])
            require(isinstance(refs, list) and all(isinstance(x, str) and x in self._known_policies(conn) for x in refs), 422, "validation_error", "Unknown policy reference")
            require(all(str(self._known_policies(conn)[x].get("version")) == item["policy_version"] for x in refs), 422, "policy_version_mismatch", "Review references must match the item policy version")
            own_pass = conn.execute("SELECT labels FROM independent_reviews WHERE item_id=? AND actor_id=?", (item_id, actor.id)).fetchone()
            if phase == "independent":
                require(decision == "save", 422, "validation_error", "Independent reviews must be saved before adjudication")
                require(own_pass is None, 409, "first_pass_exists", "Your independent first pass is already recorded")
                conn.execute("INSERT INTO independent_reviews VALUES(?,?,?,?)", (item_id, actor.id, canonical(labels), utcnow()))
            else:
                self._check_role(actor, {"sme", "policy"})
                require(own_pass is not None, 409, "first_pass_required", "Submit your independent review before viewing and adjudicating AI guidance")
                if decision == "approve":
                    require(item["split"] == "train", 403, "protected_holdout", "Protected evaluation items cannot be published through the optimization workflow")
                    require(all(v.casefold() != "uncertain" for v in labels.values()), 422, "unresolved_label", "Resolve every uncertain target before approving gold")
                    for target in item["targets"]:
                        target["staged_label"] = labels[target["id"]]
                    item["status"] = "approved"; item.pop("escalation_owner", None)
                elif decision == "escalate":
                    item["status"] = "escalated"
                    item["gold_eligible"] = False
                    item["escalation_owner"] = self._meta(conn, "policy_owner", "policy")
                    for target in item["targets"]: target["staged_label"] = None
                    item["assignee"] = None
                else:
                    item["status"] = "open"
                    item["gold_eligible"] = False
                    for target in item["targets"]: target["staged_label"] = None
            item["revision"] += 1
            # Submission hands work back to the pool so Ops can hand an item
            # to SME/Policy without a privileged reassignment side channel.
            item["assignee"] = actor.id if phase == "adjudication" and decision == "save" else None
            self._save_item(conn, item)
            self._event(conn, item_id, actor, "independent_review" if phase == "independent" else decision,
                        phase=phase, decision=decision, ai_visible=phase == "adjudication", labels=labels,
                        targets=[{"id": t["id"], "category": t["category"], "before": (t.get("gold") or {}).get("label"), "after": labels[t["id"]]} for t in item["targets"]],
                        rationale=rationale, policy_refs=refs, policy_version=item["policy_version"], revision=item["revision"],
                        summary={"approve": "Approved for the next local gold epoch", "escalate": "Escalated for Policy clarification", "save": "Independent review recorded" if phase == "independent" else "Adjudication saved"}[decision])
            return self._view(conn, item, actor)
        return self._mutate(actor, f"review:{item_id}", body, idempotency_key, action)

    def route_item(self, item_id: str, actor: Actor, body: dict, *, idempotency_key=None):
        def action(conn):
            item = self._item(conn, item_id); self._check_revision(item, body); self._check_assignee(item, actor)
            nodes = body.get("nodes")
            require(isinstance(nodes, list) and all(isinstance(x, str) and x in self._known_policies(conn) for x in nodes) and len(nodes) == len(set(nodes)), 422, "validation_error", "Route nodes must identify known policies")
            require(all(str(self._known_policies(conn)[x].get("version")) == item["policy_version"] for x in nodes), 422, "policy_version_mismatch", "Route references must match the item policy version")
            reason = nonempty(body.get("reason"), "reason")
            before = clone(item["route"])
            item["route"].update({"nodes": nodes, "fallback": not bool(nodes), "overridden_by": actor.id, "override_reason": reason,
                                  "current_route_changed": True, "trace_stale": True})
            item["status"] = "open"
            item["gold_eligible"] = False
            for target in item["targets"]: target["staged_label"] = None
            item["revision"] += 1; self._save_item(conn, item)
            self._event(conn, item_id, actor, "route", before=before, after=item["route"], rationale=reason,
                        policy_version=item["policy_version"], revision=item["revision"], summary="Policy route corrected" if nodes else "Broader Policy review requested")
            return self._view(conn, item, actor)
        return self._mutate(actor, f"route:{item_id}", body, idempotency_key, action)

    @staticmethod
    def _gold_eligible(item):
        return item["split"] == "train" and item.get("gold_eligible", True) and item["status"] != "escalated" and all(t.get("gold") and t["gold"].get("label") in t["choices"] and t["gold"]["label"].casefold() != "uncertain" and t["gold"].get("policy_version", item["policy_version"]) == item["policy_version"] for t in item["targets"])

    def _make_epoch(self, conn, epoch, actor, changed_count, note):
        created_at = utcnow(); records = []
        for item in self._all_items(conn):
            if not self._gold_eligible(item): continue
            records.append({"id": item["id"], "project": item["project"], "type": item["type"], "category": item["category"],
                            "policy_version": item["policy_version"], "content_version": item["content_version"], "revision": item["revision"],
                            "content": item["content"], "source": item["source"], "split": "train", "route": item["route"],
                            "labels": {t["id"]: t["gold"]["label"] for t in item["targets"]},
                            "targets": [{"id": t["id"], "category": t["category"], "label": t["gold"]["label"], "gold_revision": t["gold"]["revision"]} for t in item["targets"]]})
        manifest = {"epoch": epoch, "policy_versions": sorted({i["policy_version"] for i in records}), "created_at": created_at, "items": records}
        graph = self._meta(conn, "graph")
        if graph is not None:
            manifest.update({"graph": graph, "graph_id": graph["graph_id"], "graph_version": graph["version"], "graph_hash": graph["source_sha256"]})
        digest = hashlib.sha256(canonical({k:v for k,v in manifest.items() if k != "created_at"}).encode()).hexdigest()
        manifest["digest"] = digest
        record = {"epoch": epoch, "created_at": created_at, "created_by": actor.id, "item_count": len(records), "changed_count": changed_count, "digest": digest, "status": "local_published", "note": note}
        if graph is not None:
            record.update({key: manifest[key] for key in ("graph_id", "graph_version", "graph_hash")})
        conn.execute("INSERT INTO epochs VALUES(?,?,?)", (epoch, canonical(record), canonical(manifest)))
        return record

    def publish_epoch(self, actor: Actor, body: dict, *, idempotency_key=None):
        self._check_role(actor, {"policy"})
        def action(conn):
            expected = body.get("expected_epoch")
            require(isinstance(expected, int) and not isinstance(expected, bool), 422, "validation_error", "expected_epoch must be an integer")
            require(expected == self._meta(conn, "epoch", 0), 409, "epoch_conflict", "Gold changed. Refresh the current epoch before publishing.")
            note = nonempty(body.get("note"), "note")
            approved = [i for i in self._all_items(conn) if i["status"] == "approved" and i["split"] == "train"]
            require(bool(approved), 409, "no_changes", "No approved training labels are ready to publish")
            epoch = expected + 1
            for item in approved:
                require(all(t.get("staged_label") in t["choices"] and t["staged_label"].casefold() != "uncertain" for t in item["targets"]), 409, "incomplete_approval", "An approval is incomplete")
                changes = []
                for target in item["targets"]:
                    old = target.get("gold") or {}
                    changes.append({"id": target["id"], "category": target["category"], "before": old.get("label"), "after": target["staged_label"]})
                    target["gold"] = {"label": target["staged_label"], "revision": int(old.get("revision", 0)) + 1, "policy_version": item["policy_version"], "epoch": epoch}
                    target["staged_label"] = None
                item["status"] = "published"; item["revision"] += 1; item["assignee"] = None; item["gold_eligible"] = True
                self._save_item(conn, item)
                self._event(conn, item["id"], actor, "publish", targets=changes, epoch=epoch, policy_version=item["policy_version"], revision=item["revision"], summary=f"Published in local gold epoch {epoch}")
            self._set_meta(conn, "epoch", epoch)
            record = self._make_epoch(conn, epoch, actor, len(approved), note)
            self._event(conn, None, actor, "epoch_published", epoch=epoch, digest=record["digest"], summary=f"Published local gold epoch {epoch}")
            return record
        return self._mutate(actor, "publish_epoch", body, idempotency_key, action)

    def export_epoch(self, epoch: int, actor: Actor | None = None):
        if actor is not None: self._check_role(actor, {"policy"})
        with self._connection() as conn:
            row = conn.execute("SELECT manifest FROM epochs WHERE epoch=?", (epoch,)).fetchone()
            require(row is not None, 404, "not_found", "Gold epoch not found")
            return json.loads(row[0])

    def request_rerun(self, actor: Actor, body: dict, *, idempotency_key=None):
        self._check_role(actor, {"policy"})
        def action(conn):
            epoch = body.get("epoch")
            require(isinstance(epoch, int) and not isinstance(epoch, bool), 422, "validation_error", "epoch must be an integer")
            note = nonempty(body.get("note"), "note")
            row = conn.execute("SELECT manifest FROM epochs WHERE epoch=?", (epoch,)).fetchone()
            require(row is not None, 404, "not_found", "Gold epoch not found")
            manifest = json.loads(row[0])
            require(bool(manifest["items"]), 409, "empty_epoch", "The epoch has no optimization-eligible gold items")
            request = {"id": f"apo-{uuid.uuid4().hex[:12]}", "epoch": epoch, "digest": manifest["digest"], "created_at": utcnow(), "created_by": actor.id, "status": "awaiting_connector", "note": note}
            conn.execute("INSERT INTO outbox VALUES(?,?)", (request["id"], canonical(request)))
            self._event(conn, None, actor, "apo_requested", epoch=epoch, digest=request["digest"], run_id=request["id"], summary=f"Queued APO handoff for epoch {epoch}; awaiting connector")
            return request
        return self._mutate(actor, "request_rerun", body, idempotency_key, action)

    def events(self, actor: Actor):
        with self._connection() as conn:
            return {"events": [self._redact_event(e) for e in self._read_events(conn)]}

    def import_items(self, actor: Actor, body: dict, *, idempotency_key=None):
        self._check_role(actor, {"policy"})
        def action(conn):
            raw_items = body.get("items")
            require(isinstance(raw_items, list) and 1 <= len(raw_items) <= 500, 422, "validation_error", "Import between 1 and 500 items")
            if body.get("graph") is not None:
                imported_graph = self._validate_graph(conn, body["graph"])
                configured_graph = self._meta(conn, "graph") or {}
                require(imported_graph["source_sha256"] == configured_graph.get("source_sha256"), 422, "graph_validation_error", "Candidate imports cannot replace the configured policy graph")
            candidates = [self._validate_item(conn, raw) for raw in raw_items]
            imported = skipped = 0
            for item in candidates:
                if conn.execute("SELECT 1 FROM items WHERE id=?", (item["id"],)).fetchone():
                    skipped += 1; continue
                self._check_group_split(conn, item)
                conn.execute("INSERT INTO items VALUES(?,?)", (item["id"], canonical(item))); imported += 1
                self._event(conn, item["id"], actor, "import", summary="Imported candidate for independent review", policy_version=item["policy_version"])
            return {"imported": imported, "skipped": skipped}
        return self._mutate(actor, "import_items", body, idempotency_key, action)
