# AGENTS.md

This file provides guidance to AI agents when working with code in this repository.

## Development Commands

### Running the Application

- **Development server**: `uvicorn app.main:app --reload --port 8001`
- **Docker**: `docker build -t classifast . && docker run -p 8001:8001 classifast`
- **Docker Compose**: `docker-compose up -d` (includes health checks)

### Environment Setup

- Install Python dependencies: `pip install -r requirements.txt`
- Install Node dependencies: `npm install` (for Tailwind CSS development)
- Environment variables required: `GEMINI_API_KEY`, `QDRANT_URL`, `QDRANT_API_KEY`
- Optional: `RAPIDAPI_SECRET` for API authentication
- Use `.env` file for local development

### CSS Development

- **Watch CSS changes**: `npx @tailwindcss/cli -i ./app/static/css/input.css -o ./app/static/css/styles.css --watch`
- Build CSS once: `npx @tailwindcss/cli -i ./app/static/css/input.css -o ./app/static/css/styles.css`
- Input CSS: `app/static/css/input.css`
- Output CSS: `app/static/css/styles.css`

### Testing & Debugging

- **Health check**: `curl http://localhost:8001/health`
- **RapidAPI health**: `curl http://localhost:8001/api/v1/rapid/ping`
- **Manual classification test**: Use web interface or POST to `/{classifier_type}` endpoints
- **Test utilities**: Run scripts directly with `python utilities/test_rapidapi.py` etc.
  - `test_rapidapi.py`: Test RapidAPI endpoints
  - `test_embedding_ordering.py`: Validate embedding generation order
  - `test_cloudflare_headers.py`: Verify CDN header configuration
  - `test_title_functionality.py`: Test embedding with document titles
  - `check_match.py`: Verify classification accuracy for specific queries
  - `count_codes.py`: Analyze collection statistics

## Architecture Overview

### Core Application Structure

- **FastAPI backend** (`app/main.py`): Main application with lifespan management, middleware, routing, and RapidAPI integration
- **Classification engine** (`app/classifier.py`): Handles semantic search using Google Gemini embeddings and Qdrant vector database
- **Shared classification service**: `perform_classification()` function centralizes all classification logic
- **Templates**: Jinja2 templates in `app/templates/` (index.html, classifier_page.html, results.html, rate_limit_warning.html)
- **Static files**: CSS, JS, images in `app/static/`
- **Utilities**: Helper scripts in `utilities/` for testing and configuration

### Key Components

#### Classification System

- **11 classification standards supported**: UNSPSC, ETIM, NAICS, ISIC, HS, CN, NACE, CPV, NSN, HTS, plus test classifier
- **Embedding models**: Google Gemini (text-embedding-004, gemini-embedding-001) with configurable dimensions
- **Vector database**: Qdrant for semantic search with async client and quantization support
- **Batch processing**: `classify_string_batch()` function handles multiple queries efficiently
- **Quantization optimization**: INT8 scalar quantization with rescoring for improved performance

#### Application Lifecycle

- **Startup**: Initializes embedding client, Qdrant client, validates collections, pre-loads example query results
- **Pre-loading cache**: Caches results for example queries across all classifiers during startup for fast initial page loads
- **Configuration**: `CLASSIFIER_CONFIG` dict defines all supported classification standards with their versions and settings
- **Client validation**: Verifies collection existence and vector dimension compatibility on startup

#### RapidAPI Integration

- **API endpoints**: `/api/v1/rapid/classify` and `/api/v1/rapid/standards` for programmatic access
- **Authentication**: Supports both API key and proxy secret authentication
- **Rate limiting**: Separate limiter for API endpoints (600/minute)
- **Response models**: Pydantic models for structured API responses
- **Health monitoring**: Public health check endpoint for API consumers

#### Performance Features

- **Multi-layer caching**: Static file caching, pre-loaded example results, CDN-friendly headers
- **Quantization**: INT8 quantization with rescore=true and oversampling for balance of speed/accuracy
- **Middleware stack**: Gzip compression, security headers, performance monitoring, bot detection
- **Rate limiting**: 20 requests/minute on classification endpoints, 60/minute default using SlowAPI
- **HNSW optimization**: Configurable hnsw_ef parameters for search accuracy/speed tradeoffs

#### Frontend Architecture

- **HTMX**: Dynamic form submission without page refresh for seamless UX
- **Tailwind CSS**: Utility-first CSS framework with JIT compilation
- **SEO optimization**: Clean URLs, structured data, meta tags, and caching headers
- **Responsive design**: Mobile-friendly interface with semantic HTML5 structure

### Data Structure

- **Collections**: Each classification standard has versioned Qdrant collections with specific configurations
- **Vector dimensions**: Configurable per classifier (768 for text-embedding-004, 3072 for gemini-embedding-001)
- **Payload structure**: Contains original_id, class_name, and classification metadata
- **Configuration management**: Version-based configuration with base URLs and tooltips

### Error Handling & Resilience

- **Retry logic**: Tenacity-based retries for embedding API calls with exponential backoff
- **Graceful degradation**: Fallback to empty results on client failures
- **Health checks**: `/health` and `/api/v1/rapid/ping` endpoints validate service availability
- **Input validation**: Query length limits, sanitization, and parameter validation
- **Exception handling**: Centralized error handling with appropriate HTTP status codes

### Security & Compliance

- **CSP headers**: Content Security Policy for XSS protection with Cloudflare optimization
- **Security middleware**: X-Frame-Options, HSTS, X-Content-Type-Options, Permissions-Policy
- **Authentication**: Multi-mode authentication for API endpoints
- **Input sanitization**: Form data sanitization and rate limiting
- **Non-root containers**: Docker runs as non-root user for security
- **Bot detection**: Middleware for logging and analyzing bot traffic patterns

## Important Implementation Details

### URL Structure & SEO

- **Clean URLs**: Supports both `/{classifier}` and `/{classifier}/{query}` patterns
- **Slug generation**: Automatic URL-friendly slug generation for search queries
- **Canonical URLs**: Proper canonical URL generation for SEO
- **HEAD support**: HEAD requests supported for all endpoints with proper headers

### Caching Strategy

- **Static files**: 1-week cache with immutable flag for CSS/JS/images
- **HTML pages**: 1-day cache with stale-while-revalidate for content
- **Pre-loaded results**: Example query results cached at startup for instant page loads
- **CDN optimization**: Cloudflare-specific headers and cache tags

### Search Performance

- **Quantization**: Collections use INT8 scalar quantization with rescore=true and oversampling=2.0
- **HNSW parameters**: Configurable hnsw_ef (currently 256) for search accuracy/responsiveness balance
- **Batch processing**: Efficient batch embedding and search operations
- **Conditional search**: Adapts search parameters based on collection configuration
- **Result filtering**: Returns top_k results with optional rescoring for improved quality

### API Integration Patterns

- **Shared service**: `perform_classification()` centralizes all classification logic
- **Validation layers**: Input validation, client health checks, and error handling
- **Response formatting**: Consistent response structure across web and API endpoints
- **Authentication**: Multi-mode authentication supporting different access patterns
