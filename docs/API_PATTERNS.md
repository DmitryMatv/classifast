# API Patterns Guide

## API Endpoints

```python
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

router = APIRouter()

class RequestModel(BaseModel):
    code: str = Field(..., description="Classification code")

@router.get("/endpoint")
async def endpoint():
    """
    Multi-line docstring describing the endpoint.
    """
    pass
```

### Required Patterns

- Use `Depends()` for auth and rate limiting injection
- Pydantic models for all request/response validation
- Field descriptions using `Field(..., description="...")`
- Return Pydantic models directly (FastAPI handles serialization)
- Only use `JSONResponse` when you need custom headers or non-model data

## Web Routes (HTML)

```python
from fastapi import Response
from fastapi.responses import HTMLResponse

@router.get("/page", response_class=HTMLResponse)
async def page():
    # Support both GET and HEAD for CDN caching
    # Set Cloudflare-friendly cache headers
    return HTMLResponse(
        content=template.render(),
        headers={"Cache-Control": "public, max-age=3600"}
    )
```

### SEO Requirements

/

- Include canonical URLs
- Add robots meta tags
- Support HEAD method for CDN caching

## Error Handling

```python
from fastapi import HTTPException

# Input validation
if not valid:
    raise HTTPException(status_code=400, detail="Invalid input")

# External service failures
try:
    result = await external_api.call()
except ExternalError as e:
    logger.error("External API failed: %s", e)
    raise HTTPException(status_code=502, detail="External service unavailable")
```

### Patterns

- Always wrap external API calls in try/except with logging
- Log error details with elapsed time
- Raise `HTTPException` with appropriate status codes

## HTTP Client (httpx)

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.get(url)
    response.raise_for_status()
    data = response.json()
```

### Rules

- Always use `async with` context for connection management
- Use httpx for all external HTTP requests
- Add Tenacity retry decorator for external calls (see STYLE.md)

## Response Formatting

```python
# Extract payload from Qdrant results
results = [
    {
        "code": hit.payload["code"],
        "score": hit.score,
        "url": f"{base_url}/{hit.payload['slug']}"
    }
    for hit in search_results
]
```

### Patterns

- Build proper base URLs for API responses
- Extract payload data from Qdrant hits
- Format for consistent API response structure
