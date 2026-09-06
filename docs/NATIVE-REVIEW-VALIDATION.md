# Native review validation follow-up

The first full-application CI run, 34011623998, passed native graph/API and restored-About checks for both demos. It exercised recorded MNIST run exp-20260707T232538-8b0f8f; the checkout had a GenAI graph but no selected GenAI experiment. This was not a test of the private production host or its SQL connection.

Reading the actual HTTP logs uncovered a gap: native thumbnail requests returned a same-origin 302, which the preview blocked. The bridge now relays only validated same-origin static-evidence redirects through the preview. External destinations, credentials, traversal, and redirects to arbitrary API actions remain rejected. The documentation's earlier statement that all redirects are rejected is superseded by this narrower media rule.

The full-app test now verifies a committed image is served byte-identically by the native endpoint and the proxy after its real 302 redirect. Graph-only screenshots are captured separately. Genuinely missing image files display an explicit unavailable state. Evidence request failures clear the additive panel's old measurements and highlights without clearing the working native graph.

Local revalidation: 15 Python bridge/configuration tests and 13 JavaScript contracts passed. The workflow re-runs those, 13 existing graph/render contracts, and the full native application. Inspect its actual result before claiming all stages passed. No new provider calls, schema initialization, private data migration, or live SQL connection are part of these tests.
