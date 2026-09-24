---
name: verify
description: Drive Classifast's web pages and classification flow in an isolated browser run, and keep evidence of user-visible behavior.
---

# Verify Classifast

Read [the feature map](features/README.md) and the feature file for the path you are checking. The primary product is the web UI. The FastAPI HTTP routes are useful for checking downloads and response status. `app/api.py` is the separate RapidAPI integration and is outside these recipes.

## Launch

From the repo root, build frontend assets and use the existing Python environment:

```bash
npm run build
source .venv/bin/activate
sudo docker run --rm -d --name classifast-verify-redis -p 127.0.0.1:16379:6379 redis:7.2 redis-server --save "" --appendonly no
mkdir -p .cache/verify
RUN_DIR=$(mktemp -d "$PWD/.cache/verify/run-XXXXXX")
.agents/skills/verify/scripts/control.py launch "$RUN_DIR"
```

If the dedicated Redis container is already running, skip the `docker run` command after checking that it is the intended disposable instance. The verifier requires a Redis `PING` at `127.0.0.1:16379` and points the app there, even if `.env` names a different Redis host. Docker may require an interactive sudo password; never copy credentials into a tool call. Keep the container for this run and remove it after the run if you started it.

The command prints the instance URL and `run_dir`. Copy the printed `run_dir` into `RUN_DIR` at the start of every later shell tool call; shell variables do not persist between calls. The server binds to a free `127.0.0.1` port and writes `instance.json` under `RUN_DIR`. The default full mode loads `.env` for the configured on-premises Qdrant and HF/OpenRouter credentials. Startup validates Qdrant collections read-only. `health_gate_passed: true` and `local_redis_ready: true` are required before a lookup; only a completed lookup proves the external classification path. Do not run `utilities/qdrant_config.py` or `utilities/sync_payload_indexes.py apply` as part of verification. Use `npm run dev` for interactive development and this script for an isolated, recorded run. Each instance has its own app port, but concurrent full-mode runs can share Redis quota keys and external service credentials.

For page and download checks without live classification, launch a fresh run directory with `--mode public`. Public mode disables FastAPI lifespan and does not connect to Qdrant, Redis, or embedding services. It cannot prove classification results or checkout. Do not drive another instance or use the deployed site as a substitute.

## Doctor

Run this read-only check before a browser session and whenever a page looks wrong:

```bash
.agents/skills/verify/scripts/control.py doctor "$RUN_DIR" | tee "$RUN_DIR/evidence/doctor.json"
```

It checks the recorded process identity, the mapping page, and `/health`. Full mode also requires a local Redis `PING`, `/health` 200, and `health_gate_passed: true`. Public mode expects the mapping page to work and reports `/health` as 503 because no clients started. If the process is gone or the port shows the wrong app, launch a new run rather than driving an existing process.

## Drive

Use Codex's collaborative browser preview tools for the real UI. Pass the exact `url` printed by launch, plus `/`, to `mcp__t3_code__preview_open`. Inspect it with `mcp__t3_code__preview_snapshot({"includeImage":false})`, then act with `preview_click`, `preview_type`, and `preview_wait_for`. Keep the returned `tabId` and pass it to later calls. For example, the homepage mapping entry is `role=link[name='Mapping Files']`, the catalog product is `role=link[name='UNSPSC to CPV Mapping']`, and the classifier input is `role=textbox[name='Product description']`. The [feature files](features/README.md) give each exact route, selector, and expected state. Navigate the browser to the instance port, never to `classifast.com`.

If preview tools are unavailable, use a browser with Playwright selectors against this same local URL. If `preview_snapshot` fails while other preview actions work, use read-only `preview_evaluate` calls to record the current URL, submitted input value, and visible results before and after the action. Note the missing screenshot in the evidence. A `curl` response alone proves an HTTP route, not the browser interaction. For sample downloads, request the exact visible link's `href` with `curl -fSL` and save the response as a CSV artifact.

The homepage and classifier pages load production Clerk scripts and Google Analytics. Clerk rejects a localhost origin and logs a browser error; that does not prevent public navigation or mapping downloads. Do not count a local sign-in check as passed on this setup, and expect outbound analytics traffic unless the browser blocks it.

## Evidence

Put proof under `$RUN_DIR/evidence/`; cleanup preserves it. Record the feature ID and entry point in `evidence/notes.md`. Capture a browser snapshot before the action and after the result, using `preview_snapshot({"save":true,"includeImage":false})` and copy each returned `screenshotPath` into the evidence directory when snapshots work. Record the browser URL and visible text after each action. Save HTTP headers or downloaded files for HTTP side effects, such as the sample CSV. Check that the file has a CSV header and data rows. For classification, capture the submitted description, the resulting list or explicit empty state, and the URL change. A loading indicator, mock response, or `/health` result alone is not proof of classification.

Use production user paths. Never set DOM state or call a test-only endpoint to manufacture success. Mock external services only at a boundary already isolated by the app, and say so in the proof. Do not assume a dry-run skips external work: inspect its actual network calls and side effects before relying on it. Payment checkout and sign-in can create external state; verify them only with configured test credentials and inspect the resulting Polar or Clerk state separately.

## Cleanup

Stop only the process this run started:

```bash
.agents/skills/verify/scripts/control.py cleanup "$RUN_DIR"
test -f "$RUN_DIR/evidence/doctor.json"
```

The script checks the recorded PID and process start time, stops that process, and removes `instance.json`. Keep `$RUN_DIR/evidence/` and its screenshots, downloads, and server log. Never use `pkill` or remove another run's directory. After a failed launch, the script stops its own server; inspect `evidence/server.log` before retrying in a new run directory.

If you started the dedicated Redis container above, remove that exact container after cleanup with `sudo docker rm -f classifast-verify-redis`. Leave an existing container running if someone else started it.
