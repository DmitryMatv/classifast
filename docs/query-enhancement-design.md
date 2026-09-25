# Optional query enhancement

## Use

The classifier form submits the original description and an `enhance_query=1` flag only when the beta switch is on. The fragment route reserves usage, asks OpenRouter for a short description, and classifies with both the original and the resulting semantic text. The original stays in the input, results context, page title, ID lookup, and URL path. The flag stays in the URL query string to reproduce an enhanced search after refresh.

## Contract

```text
QueryEnhancer.enhance(original: str, classifier_type: str) -> str
ClassificationService.classify(query: str, ..., semantic_query: str | None = None)
perform_classification(query: str, ..., semantic_query: str | None = None)
build_classification_results_context(..., semantic_query: str | None = None)
```

`query` is always the validated user text. `semantic_query` contains that text plus a bounded, generic description when the model supplies one. The search pipeline uses `semantic_query` for embedding and reranking, and `query` for exact and partial ID lookup. An absent semantic query means the existing behavior.

## Decision

An alternative places the model call inside the synchronous classifier after ID lookup. That avoids a call for IDs but occupies the single classification worker during network I/O. The fragment route can call OpenRouter asynchronously after quota approval and leave the worker available. The model has a short timeout, no retry, and a conservative prompt. A failed, empty, or invalid response falls back to the original text.

The switch is unchecked unless the page URL explicitly has `enhance_query=1`. The canonical page URL remains based on the original query. The fragment GET flag separates cached enabled and disabled results.
