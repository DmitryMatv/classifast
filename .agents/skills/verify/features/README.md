# Classifast feature map

This index is the maintained source for browser verification. Read the matching feature file before driving a flow. Start a fresh local instance with `../SKILL.md` and run doctor first. Full mode is the default and uses live classification services plus local Redis. Use explicit `--mode public` for pages and mapping downloads when live classification is unnecessary.

Record each tested feature ID and entry point in `$RUN_DIR/evidence/notes.md`. Capture the user action and resulting page, plus any downloaded file. An untested entry point is still unverified even when a related route works.

## Features

- [Find a standard](home-navigation.md): standard links, homepage shortcuts, popular lookups, and mobile navigation.
- [Classify a description](classification.md): submit a description, change the result count, copy a code, and reopen the shareable URL. The file also records quota recovery and its authentication prerequisites.
- [Browse mappings and download samples](mapping-sample.md): inspect both crosswalk products, download their CSV samples, and identify the prerequisites for paid checkout.

Public mode covers navigation and free mapping samples. Classification needs full mode and available quota. Authenticated quota recovery needs a localhost-compatible Clerk test configuration. Successful paid checkout needs a dedicated Polar sandbox integration, which the current payment routes do not select. Record these restricted paths as unreachable with their entry point and prerequisite; public pages and fallback sign-in links do not prove them.
