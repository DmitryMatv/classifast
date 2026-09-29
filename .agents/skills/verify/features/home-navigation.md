# Find a standard

The homepage lets a visitor choose a classification standard or open the mapping catalog.

## Sub-features

- `home-standards` shows the supported standard links.
- `home-classifier-link` opens a classifier form from the standard list.
- `home-mapping-link` opens the mapping catalog from the main navigation.
- `home-shortcuts` opens UNSPSC from the hero and sample-result card, or scrolls to the standards list.
- `home-popular-lookup` opens a shared query from `Popular code lookups`.
- `home-mobile-menu` opens and closes the menu and reaches classifiers and mapping files at a narrow viewport.

## How to get to it (user POV)

- Open the homepage and choose `UNSPSC` under `12 standards in 1 search box`.
- Open the homepage and choose `Mapping` in the main navigation.
- On a narrow viewport, open `Toggle mobile menu` and choose a standard or `Mapping Files` there.
- Choose `Find UNSPSC code` or `Try it with your own description` to open UNSPSC. Choose `Browse all classifiers` to reach the standards list.
- Choose a query under `Popular code lookups` to open a classifier with that description.

## Driving it with browser preview

Preconditions: doctor reports `mapping_page_ready: true` for this run.

- Navigate to the instance root with `preview_navigate({"url":"<instance URL>/"})`. Snapshot the page and confirm `12 standards in 1 search box` and the Classifast masthead.
- Choose the UNSPSC list link with `preview_click({"selector":"#standards a[href$='/UNSPSC/']"})`. Wait for `urlIncludes: "/UNSPSC/"` and `role=textbox[name='Product description']`. Save the before and after snapshots.
- Return to `/` and choose the desktop mapping link with `preview_click({"locator":"role=link[name='Mapping Files']"})`. Wait for `urlIncludes: "/mapping/"` and text `Mapping tables for cross-referencing`. Save the result snapshot.
- Return to `/` between shortcut checks. Click `role=link[name='Find UNSPSC code']` and `role=link[name='Try it with your own description']` separately. Each opens `/UNSPSC/` with the description textbox. Click `role=link[name='Browse all classifiers']` and confirm the URL hash is `#standards`.
- From `/`, record the first `.popular-grid a` link's name and pathname, then click it. Confirm the same pathname and a populated description textbox. Public mode proves query navigation only; use full mode to confirm its completed classification result.
- To check the homepage mobile path, return to `/` first. Call `preview_resize({"mode":"preset","preset":"iphone-12-pro","orientation":"portrait"})`, then click `role=button[name='Toggle mobile menu']`. Confirm `aria-expanded="true"` and visible mobile navigation. Press Escape and confirm `aria-expanded="false"` and focus on `#mobile-menu-button`.
- Reopen the menu and click `nav[aria-label='Mobile navigation'] a[href$='/UNSPSC/']`. Confirm `/UNSPSC/` and the description textbox. Return home, reopen the menu, and click `nav[aria-label='Mobile navigation'] a[href$='/mapping/']`. Confirm the catalog heading. Record these entry points separately.

## Gotchas

- `/UNSPSC` and `/mapping` redirect to slash-terminated canonical routes.
- The standard list contains several links named by acronym; scope the UNSPSC click to `#standards`.
- The homepage hamburger can open the menu at desktop width too. Resize to test the narrow layout, and scope menu links to `nav[aria-label='Mobile navigation']`.
- Popular lookup links come from the sitemap-filtered list in `app/classifier_page_delivery.py`; record the visible destination rather than hardcoding the first query. Compare URL pathnames because links can be relative or absolute.
