# Optional query enhancement

The classifier form sends the original query and `enhance_query=1` when the beta
switch is on. The fragment route reserves usage before classification. It then
passes the beta flag to `ClassificationService.classify`.

The service uses the classification worker to validate the query and search for
an exact `original_id` match. An exact hit returns immediately. After a miss,
the service asks OpenRouter asynchronously for a short description, then resumes
partial ID and semantic search on the classification worker. The prepared
classification state prevents a second exact ID lookup.

The model receives one user message:

```text
<original query>

<instruction mentioning the classification standard>
```

When the model returns a usable description, the semantic text is:

```text
<original query>

<generated description>
```

Only that successful beta path puts the semantic text first in embedding and
reranker inputs, followed by a blank line and the standard-specific instruction.
Ordinary searches and beta fallbacks keep the existing instruction-first format.
An exact match, a code-like query, an empty description, or a provider failure
uses the original query for semantic search. Provider failures and a missing
enhancer set `no-store` on the fragment response. Intentional skips and empty
descriptions use the normal classification cache profile.

The original query always supplies exact and partial ID lookup, result headings,
URLs, and the outcome's `query` field. The share URL retains `enhance_query=1`,
so a refreshed beta search repeats the same flow. Direct and ordinary callers
still use one classification worker call through `perform_classification`.
