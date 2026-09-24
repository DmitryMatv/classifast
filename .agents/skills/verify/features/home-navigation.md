# Find a standard

The homepage lets a visitor choose a classification standard or open the mapping catalog.

## Sub-features

- `home-standards` shows the supported standard links.
- `home-classifier-link` opens a classifier form from the standard list.
- `home-mapping-link` opens the mapping catalog from the main navigation.

## How to get to it (user POV)

- Open the homepage and choose `UNSPSC` under `12 standards in 1 search box`.
- Open the homepage and choose `Mapping` in the main navigation.
- On a narrow viewport, open `Toggle mobile menu` and choose a standard or `Mapping Files` there.

## Driving it with browser preview

Preconditions: doctor reports `mapping_page_ready: true` for this run.

- Navigate to the instance root with `preview_navigate({"url":"<instance URL>/"})`. Snapshot the page and confirm `12 standards in 1 search box` and the Classifast masthead.
- Choose the UNSPSC list link with `preview_click({"selector":"#standards a[href$='/UNSPSC/']"})`. Wait for `urlIncludes: "/UNSPSC/"` and `role=textbox[name='Product description']`. Save the before and after snapshots.
- Return to `/` and choose the desktop mapping link with `preview_click({"locator":"role=link[name='Mapping Files']"})`. Wait for `urlIncludes: "/mapping/"` and text `Mapping tables for cross-referencing`. Save the result snapshot.
- To check the homepage mobile path, return to `/` first. Call `preview_resize({"mode":"preset","preset":"iphone-12-pro","orientation":"portrait"})`, click `role=button[name='Toggle mobile menu']`, then click `nav[aria-label='Mobile navigation'] a[href$='/mapping/']`. This link displays `Mapping Files` and a description. The catalog heading appears. Record this entry point separately.

## Gotchas

- `/UNSPSC` and `/mapping` redirect to slash-terminated canonical routes.
- The standard list contains several links named by acronym; scope the UNSPSC click to `#standards`.
- The homepage hamburger can open the menu at desktop width too. Resize to test the narrow layout, and scope menu links to `nav[aria-label='Mobile navigation']`.
