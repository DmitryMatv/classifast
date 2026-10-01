# Browse mappings and download samples

A visitor browses crosswalk products, reads one product's coverage, and downloads a free sample CSV.

## Sub-features

- `mapping-catalog` lists both crosswalk products with coverage and price.
- `mapping-detail` shows the selected product and its sample action.
- `mapping-sample` returns a CSV sample with header and data rows.

## How to get to it (user POV)

- Choose `Mapping` from the homepage's main navigation, or `Mapping Files` from its mobile menu.
- Open `/mapping/` directly and choose `UNSPSC to CPV Mapping`.
- On the product page, choose `Download sample CSV`. The catalog also has this link.

## Driving it with browser preview

Preconditions: doctor reports `mapping_page_ready: true`. Public mode is sufficient.

- Navigate to `<instance URL>/` and snapshot the page. Click `role=link[name='Mapping Files']`, then wait for text `Mapping tables for cross-referencing`. Save the catalog snapshot and record both product titles, prices, and coverage. Match the heading by this substring; its full text also includes procurement and finance.
- Click `role=link[name='UNSPSC to CPV Mapping']` within `article.listing-row`. Wait for `/mapping/unspsc-to-cpv-mapping/` and the heading `UNSPSC to CPV Mapping`. Save the detail snapshot; confirm `Download sample CSV` is visible.
- Request that visible link's route as a browser download or with `curl -fSL -D "$RUN_DIR/evidence/sample-headers.txt" "<instance URL>/mapping/unspsc-to-cpv-mapping/sample" -o "$RUN_DIR/evidence/unspsc-to-cpv-sample.csv"`. Check the HTTP status, CSV header, and at least one data row. The file itself proves the download.
- Return to the catalog and download the UNSPSC sample from its scoped listing link. Record that entry point separately. Download the reverse sample from the `CPV to UNSPSC Mapping` listing and check its header and data rows too. Retain each CSV separately and confirm `Content-Disposition` names an attachment when recording HTTP headers.

## Paid mapping checkout

`Buy full mapping file` appears in both catalog listings and on the detail page. It sends the selected product to `POST /api/create-mapping-checkout` and redirects to Polar when checkout succeeds. Mapping checkout does not require Clerk sign-in. The button temporarily displays `Error - Try again` after a failed request.

Successful checkout needs full mode with working local Redis, a configured Polar product and access token, and a dedicated sandbox integration. The current payment routes use the production SDK server, so a sandbox token alone is insufficient. Record checkout as unreachable in this setup and capture the visible purchase entry point without pressing it. Product-detail buttons send a `classifast.com` return URL, which must also be allowed by `ALLOWED_REDIRECT_HOSTS` for local checkout verification.

## Gotchas

- The `Buy full mapping file` button starts checkout. Do not press it while checking the free sample.
- The catalog repeats `Download sample CSV` for two products. Scope the link to the UNSPSC to CPV listing.
- `/mapping` and `/mapping/unspsc-to-cpv-mapping` redirect to slash-terminated pages. The sample route has no trailing slash.
