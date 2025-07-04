# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Classifast is a FastAPI web application that provides semantic classification of text inputs according to international product and service standards (UNSPSC, NAICS, ISIC, ETIM, HS). It uses Google Gemini embeddings and Qdrant vector database for intelligent categorization.

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables (create .env file)
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

### Development Server
```bash
# Run development server with hot reload
uvicorn app.main:app --reload --port 8001

# Run with specific host binding
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

### Docker Development
```bash
# Build and run with Docker Compose
docker-compose up

# Build specific service
docker-compose build classifier-app

# Run in detached mode
docker-compose up -d
```

### Health Check
```bash
# Check application health
curl http://localhost:8001/health
```

## Architecture Overview

### Core Components

**FastAPI Application** (`app/main.py`):
- Async/await FastAPI server with lifecycle management
- Rate limiting (10 requests/minute for classification endpoints)
- GZip compression and performance monitoring middleware
- Bot detection and security headers

**Classification Engine** (`app/classifier.py`):
- Semantic search using Google Gemini embeddings
- Qdrant vector database for similarity search
- Batch processing with retry mechanisms
- Confidence scoring for results

**Frontend** (`app/templates/` + `app/static/`):
- Server-side rendered Jinja2 templates
- HTMX for dynamic interactions
- Tailwind CSS for styling
- SEO-optimized with structured data

### Supported Classification Standards

The application supports multiple international standards:
- **UNSPSC**: United Nations Standard Products and Services Codes
- **ETIM**: European Technical Information Model  
- **NAICS**: North American Industry Classification System
- **ISIC**: International Standard Industrial Classification
- **HS**: Harmonized System for customs/trade

### Key Directories

- `app/`: Main FastAPI application package
- `app/templates/`: Jinja2 HTML templates
- `app/static/`: CSS, JavaScript, and image assets
- `data/`: Classification standard data files (CSV, Excel)
- `utilities/`: Helper scripts for data processing

### API Endpoints

- `GET /` - Homepage
- `GET /{classifier_type}` - Classification interface (etim, unspsc, naics, isic, hs)
- `POST /{classifier_type}` - Submit classification request
- `GET /health` - Health check endpoint

### Dependencies

**Core Runtime**:
- FastAPI with uvicorn ASGI server
- Google GenAI for embeddings
- Qdrant client for vector search
- Jinja2 for templating

**Development**:
- python-dotenv for environment management
- slowapi for rate limiting
- tenacity for retry logic

## Environment Variables

Required environment variables:
- `GEMINI_API_KEY`: Google GenAI API key for embeddings
- `QDRANT_URL`: Qdrant vector database URL
- `QDRANT_API_KEY`: Qdrant authentication key

## Testing

Note: No automated test suite is currently present in the codebase. Consider adding pytest-based tests for:
- API endpoint testing
- Classification accuracy validation
- Vector database operations
- Error handling scenarios

## Performance Considerations

- Vector embeddings are generated in batches for efficiency
- Qdrant provides fast similarity search at scale
- Static assets are optimized with appropriate caching headers
- Rate limiting prevents API abuse
- GZip compression reduces response sizes

## Data Processing

Classification data is stored in the `data/` directory:
- UNSPSC data in Excel format
- ETIM classes in CSV format
- Utility scripts in `utilities/` for data manipulation

The application loads and processes this data into vector embeddings for semantic search capabilities.