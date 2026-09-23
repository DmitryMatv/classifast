# Browse and download a mapping sample

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

- Navigate to `<instance URL>/` and snapshot the page. Click `role=link[name='Mapping Files']`, then wait for text `Mapping tables for cross-referencing`. Save the catalog snapshot and record both product titles.
- Click `role=link[name='UNSPSC to CPV Mapping']` within `article.listing-row`. Wait for `/mapping/unspsc-to-cpv-mapping/` and the heading `UNSPSC to CPV Mapping`. Save the detail snapshot; confirm `Download sample CSV` is visible.
- Request that visible link's route as a browser download or with `curl -fSL -D "$RUN_DIR/evidence/sample-headers.txt" "<instance URL>/mapping/unspsc-to-cpv-mapping/sample" -o "$RUN_DIR/evidence/unspsc-to-cpv-sample.csv"`. Check the HTTP status, CSV header, and at least one data row. The file itself proves the download.
- Return to the catalog and use its `Download sample CSV` link for the same product if covering that entry point. Record it separately.

## Gotchas

- The `Buy full mapping file` button starts checkout. Do not press it while checking the free sample.
- The catalog repeats `Download sample CSV` for two products. Scope the link to the UNSPSC to CPV listing.
- `/mapping` and `/mapping/unspsc-to-cpv-mapping` redirect to slash-terminated pages. The sample route has no trailing slash.
