# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
- **Development server**: `uvicorn app.main:app --reload --port 8001`
- **Docker**: `docker-compose up` (serves on port 8001)
- **Docker build**: `docker build -t classifast .`

### Environment Setup
- Install dependencies: `pip install -r requirements.txt`
- Environment variables required: `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`
- Use `.env` file for local development

### CSS Development
- **Watch CSS changes**: `npx @tailwindcss/cli -i ./app/static/css/input.css -o ./app/static/css/styles.css --watch`
- Input CSS: `app/static/css/input.css`
- Output CSS: `app/static/css/styles.css`

## Architecture Overview

### Core Application Structure
- **FastAPI backend** (`app/main.py`): Main application with lifespan management, middleware, and routing
- **Classification engine** (`app/classifier.py`): Handles semantic search using Google Gemini embeddings and Qdrant vector database
- **Templates**: Jinja2 templates in `app/templates/` (index.html, classifier_page.html, results.html)
- **Static files**: CSS, JS, images in `app/static/`

### Key Components

#### Classification System
- **Multiple standards supported**: UNSPSC, ETIM, NAICS, ISIC, HS codes
- **Embedding models**: Google Gemini (text-embedding-004, gemini-embedding-exp-03-07)
- **Vector database**: Qdrant for semantic search with async client
- **Batch processing**: `classify_string_batch()` function handles multiple queries efficiently

#### Application Lifecycle
- **Startup**: Initializes embedding client, Qdrant client, validates collections, pre-loads example query results
- **Pre-loading**: Caches results for example queries across all classifiers during startup for fast initial page loads
- **Configuration**: `CLASSIFIER_CONFIG` dict defines all supported classification standards with their versions and settings

#### Performance Features
- **Caching**: Static file caching, pre-loaded example results, CDN-friendly headers
- **Middleware**: Gzip compression, security headers, performance monitoring, bot detection
- **Rate limiting**: 10 requests/minute on classification endpoints using SlowAPI

#### Frontend
- **HTMX**: Dynamic form submission without page refresh
- **Tailwind CSS**: Utility-first CSS framework
- **Responsive design**: Mobile-friendly interface with SEO optimization

### Data Structure
- **Collections**: Each classification standard has versioned Qdrant collections
- **Vector dimensions**: Varies by model (768 for text-embedding-004, 3072 for gemini-embedding-exp-03-07)
- **Payload structure**: Contains original_id, class_name, and classification metadata

### Error Handling
- **Retry logic**: Tenacity-based retries for embedding API calls
- **Graceful degradation**: Fallback to empty results on client failures
- **Health checks**: `/health` endpoint validates both embedding and Qdrant clients

### Security
- **CSP headers**: Content Security Policy for XSS protection
- **Security middleware**: X-Frame-Options, HSTS, X-Content-Type-Options
- **Input validation**: Form data sanitization and rate limiting
- **Non-root containers**: Docker runs as non-root user for security