# Classify a description

A visitor enters a product description and receives ranked codes for the chosen standard.

## Sub-features

- `classify-form` shows the selected standard, version, description field, and result-count choice.
- `classify-submit` returns code results or an explicit no-match state.
- `classify-share` updates the URL so the lookup can be reopened.

## How to get to it (user POV)

- Choose a standard from the homepage, such as `UNSPSC`.
- Open `/UNSPSC/` directly or follow a shared `/UNSPSC/<query>/` URL.
- Enter text in `Product description` and choose `Lookup code for product description`.

## Driving it with browser preview

Preconditions: launch with `--mode full`; doctor must report `health_gate_passed: true`. Use a quota-eligible test identity or a verified local quota setup. Confirm the configured Hugging Face, Qdrant, and optional reranker targets are appropriate before a live request.

- Navigate to `<instance URL>/UNSPSC/`. Snapshot the form. Find `role=textbox[name='Product description']`, `role=combobox[name='Classification version']`, and `role=combobox[name='Number of results to show']`.
- Fill the description with `stainless steel office scissors` using `preview_type({"locator":"role=textbox[name='Product description']","text":"stainless steel office scissors","clear":true})`. Save a snapshot with the typed value before submission.
- Click `role=button[name='Lookup code for product description']`. Wait for the submitted fragment request to finish and the URL path to contain `/UNSPSC/stainless_steel_office_scissors/`. Then confirm that the updated `#results-container` contains `[role='listitem']` or `No matching classification results found.` Existing example results alone do not prove submission. A quota warning or 503 is a failed precondition, not a classification result.
- For a nonempty result, record a displayed code, name, and score from `#results-container`. Use `role=button[name='Copy link']` if present, then reopen the resulting URL in the same local instance and confirm the query is present. Save the result snapshot and URL.

## Gotchas

- Public mode renders the form but cannot classify; `/health` is 503 there by design.
- Classification invokes paid or metered external services and may reserve quota in Redis. Use a suitable test setup.
- Opening a classifier page in full mode can classify its example before submission. Check the service targets before navigation, and do not count those example results as the submitted query's result.
- A shared URL may load results asynchronously. Wait for a result or explicit empty state, not a fixed delay.
- The form can return a paywall when quota is exhausted. Record that as a distinct outcome.
