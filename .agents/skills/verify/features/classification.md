# Classify a description

A visitor enters a product description and receives ranked codes for the chosen standard.

## Sub-features

- `classify-form` shows the selected standard, version, description field, and result-count choice.
- `classify-submit` returns code results or an explicit no-match state.
- `classify-share` updates the URL so the lookup can be reopened.
- `classify-beta-enhancement` lets a visitor opt in to a query description for embedding and reranking while keeping the submitted text visible.
- `classify-count` automatically reruns the lookup when the result-count choice changes.
- `classify-copy-code` copies the original classification ID from a result row.

## How to get to it (user POV)

- Choose a standard from the homepage, such as `UNSPSC`.
- Open `/UNSPSC/` directly or follow a shared `/UNSPSC/<query>/` URL.
- Enter text in `Product description` and choose `Lookup code for product description`.

## Driving it with browser preview

Preconditions: use the default full-mode launch with the disposable local Redis server described in the main skill. Doctor must report `health_gate_passed: true` and `local_redis_ready: true`. The server reads the configured Qdrant, Hugging Face, and optional OpenRouter targets from `.env`. Use a quota-eligible test identity.

- Navigate to `<instance URL>/UNSPSC/`. Inspect the form. Find `role=textbox[name='Product description']`, `role=combobox[name='Classification version']`, and `role=combobox[name='Number of results to show']`.
- Fill the description with `stainless steel office scissors` using `preview_type({"locator":"role=textbox[name='Product description']","text":"stainless steel office scissors","clear":true})`. Record the typed value before submission using a snapshot or the read-only fallback in the main skill.
- Click `role=button[name='Lookup code for product description']`. Wait for the submitted fragment request to finish and the URL path to contain `/UNSPSC/stainless_steel_office_scissors/`. Then confirm that the updated `#results-container` contains `[role='listitem']` or `No matching classification results found.` Existing example results alone do not prove submission. A quota warning or 503 is a failed precondition, not a classification result.
- For a nonempty result, record a displayed code, name, and score from `#results-container`. Click the first row's `[data-copy-original-id]` button and confirm the clipboard matches that attribute. Record its `View details` link's destination without leaving the local instance. Grant clipboard permissions in the browser when needed; a visible button alone does not prove copying.
- Change `role=combobox[name='Number of results to show']` to `30`. In Playwright, use `selectOption('30')`. Confirm an automatic fragment request, `top_k=30` in the URL, and the updated result list. No submit click is needed for this control.
- Click `role=button[name='Copy link']` and capture the `Copied!` feedback. Confirm the clipboard equals the current local URL, then reopen that exact URL. Confirm the description, selected count, and completed results. Save the result snapshot and URL. If clipboard reading is unavailable, record sharing as partially verified.
- To check `classify-beta-enhancement`:
  1. Start at `/UNSPSC/` and confirm `Expand query Beta` is off. Enter a short query such as `POS` and submit it. Record the URL, input value, results heading, and first result.
  2. Click the visible `Expand query Beta` label and submit the same query again. Confirm the URL includes `?enhance_query=1`, the switch is on, the input and heading still show the original query, and a completed result list appears.
  3. Record the first result. Check `evidence/server.log` for a successful OpenRouter chat completion and rerank request using the expanded query. Live model output can vary, so do not require a particular code or score.
  4. Reopen the enhanced URL and confirm it restores the switch and original query.

## Quota recovery and paid access

An exhausted trial displays `Free trial limit reached` in `#paywall-warning`. Anonymous visitors see `Sign In`; all visitors see `Upgrade to Pro` and `Try again`. `Try again` resubmits the current form. A successful sign-in also retries the form and uses the authenticated allowance.

Verify this branch only with an exhausted test identity and a Clerk test configuration that accepts the local origin. Record the submitted classifier route and paywall state. Click `Try again` and confirm that the current description is resubmitted. With test authentication available, sign in and confirm a completed lookup after the retry. Production Clerk's localhost rejection makes authenticated recovery unreachable in the default setup.

`Upgrade to Pro` sends a request to `POST /api/create-checkout` after authentication. Successful checkout also needs a dedicated Polar sandbox integration; the current backend uses the production SDK server. Record the missing prerequisite without creating a production checkout. A `checkout=success` URL alone grants no entitlement; subscription activation comes from the verified Polar webhook.

## Gotchas

- Explicit `--mode public` renders the form but cannot classify; `/health` is 503 there by design.
- Classification invokes paid or metered external services and may reserve quota in Redis. Use a suitable test setup.
- Opening a classifier page in full mode can classify its example before submission. Check the service targets before navigation, and do not count those example results as the submitted query's result.
- A shared URL may load results asynchronously. Wait for a result or explicit empty state, not a fixed delay.
- The form can return a paywall when quota is exhausted. Record that as a distinct outcome.
- The description field accepts 4,000 characters, but shared URL slugs retain at most 200 characters and sanitize punctuation. Use a short description for a full round-trip check; a long input is not preserved in full by its share link.
