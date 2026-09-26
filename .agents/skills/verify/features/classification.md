# Classify a description

A visitor enters a product description and receives ranked codes for the chosen standard.

## Sub-features

- `classify-form` shows the selected standard, version, description field, and result-count choice.
- `classify-submit` returns code results or an explicit no-match state.
- `classify-share` updates the URL so the lookup can be reopened.
- `classify-beta-enhancement` lets a visitor opt in to a query description for embedding and reranking while keeping the submitted text visible.

## How to get to it (user POV)

- Choose a standard from the homepage, such as `UNSPSC`.
- Open `/UNSPSC/` directly or follow a shared `/UNSPSC/<query>/` URL.
- Enter text in `Product description` and choose `Lookup code for product description`.

## Driving it with browser preview

Preconditions: use the default full-mode launch with the disposable local Redis container. Doctor must report `health_gate_passed: true` and `local_redis_ready: true`. The server reads the configured Qdrant, Hugging Face, and optional OpenRouter targets from `.env`. Use a quota-eligible test identity.

- Navigate to `<instance URL>/UNSPSC/`. Inspect the form. Find `role=textbox[name='Product description']`, `role=combobox[name='Classification version']`, and `role=combobox[name='Number of results to show']`.
- Fill the description with `stainless steel office scissors` using `preview_type({"locator":"role=textbox[name='Product description']","text":"stainless steel office scissors","clear":true})`. Record the typed value before submission using a snapshot or the read-only fallback in the main skill.
- Click `role=button[name='Lookup code for product description']`. Wait for the submitted fragment request to finish and the URL path to contain `/UNSPSC/stainless_steel_office_scissors/`. Then confirm that the updated `#results-container` contains `[role='listitem']` or `No matching classification results found.` Existing example results alone do not prove submission. A quota warning or 503 is a failed precondition, not a classification result.
- For a nonempty result, record a displayed code, name, and score from `#results-container`. Use `role=button[name='Copy link']` if present, then reopen the resulting URL in the same local instance and confirm the query is present. Save the result snapshot and URL.
- To check `classify-beta-enhancement`:
  1. Start at `/UNSPSC/` and confirm `Better results Beta` is off. Enter a short query such as `POS` and submit it. Record the URL, input value, results heading, and first result.
  2. Click the visible `Better results Beta` label and submit the same query again. Confirm the URL includes `?enhance_query=1`, the switch is on, the input and heading still show the original query, and a completed result list appears.
  3. Record the first result. Check `evidence/server.log` for a successful OpenRouter chat completion and rerank request using the expanded query. Live model output can vary, so do not require a particular code or score.
  4. Reopen the enhanced URL and confirm it restores the switch and original query.

## Gotchas

- Explicit `--mode public` renders the form but cannot classify; `/health` is 503 there by design.
- Classification invokes paid or metered external services and may reserve quota in Redis. Use a suitable test setup.
- Opening a classifier page in full mode can classify its example before submission. Check the service targets before navigation, and do not count those example results as the submitted query's result.
- A shared URL may load results asynchronously. Wait for a result or explicit empty state, not a fixed delay.
- The form can return a paywall when quota is exhausted. Record that as a distinct outcome.
