# Testing Guide

## Philosophy

Tests in `utilities/` are **integration tests**, not unit tests. They verify external API integrations work correctly.

## Running Tests

```bash
# Run specific test
python utilities/test_<name>.py

# Examples
python utilities/test_rapidapi.py
python utilities/test_embedding_ordering.py
```

## Test Structure

```python
# utilities/test_example.py
import asyncio

async def test_something():
    result = await some_api_call()
    assert result is not None, "API call failed"
    print(f"✓ Test passed: {result}")

def main():
    asyncio.run(test_something())

if __name__ == "__main__":
    main()
```

### Requirements

- Self-contained scripts with `main()` entry point
- Use f-strings for readable output
- Manual assertions (not pytest)
- Test external integrations: RapidAPI, Clerk, Qdrant, Redis

## Health Checks

```bash
# Basic health
curl http://localhost:8001/health

# RapidAPI ping
curl http://localhost:8001/api/v1/rapid/ping
```

## CSS Development

```bash
# Watch mode (during development)
npx @tailwindcss/cli -i ./app/static/css/input.css -o ./app/static/css/styles.css --watch

# Build (production)
npx @tailwindcss/cli -i ./app/static/css/input.css -o ./app/static/css/styles.css
```

## Docker

```bash
# Build and run
docker build -t classifast . && docker run -p 8001:8001 classifast

# Or use docker-compose
docker-compose up -d
```
