# Classifast

## Accurate classifier of UNSPSC, NAICS, HS/CN codes, and more

### Get the right category codes from the most widely used classification standards

Classifast is a web application that provides easy classification of any text input according to international product and service standards like UNSPSC, NAICS, ISIC, ETIM, HS, CN. Built with FastAPI and modern web technologies, it offers fast, accurate semantic search capabilities for automated yet intelligent categorization.

## Features

- **Fast Classification**: Semantic search using advanced embedding models
- **High Accuracy**: Confidence scores for each classification result
- **Multiple Standards**: Support for UNSPSC, ETIM, and NAICS classification standards
- **Modern Interface**: Clean, responsive design built with Tailwind CSS
- **SEO Optimized**: Structured data, meta tags, and performance optimized

## Supported classification standards (Top 3)

### UNSPSC (United Nations Standard Products and Services Codes)

- Global standard for product and service categorization
- Improves spend analytics and procurement processes
- Version: UNv260801 (August 14, 2023)

### ETIM (European Technical Information Model)

- B2B open standard for technical product classification
- Specialized for electrical and technical products
- Version: 10.0 (2024-12-10)

### NAICS 2022 (North American Industry Classification System)

- Industry classification for business activities
- Essential for government contracting and reporting

## Stack

- **Backend**: FastAPI with Python
- **Frontend**: TypeScript, Tailwind CSS, HTMX
- **Vector Database**: Qdrant for semantic search
- **Embedding Models**: Google Gemini
- **Deployment**: Docker containerized

## Demo

Visit the working instance at [classifast.com](https://classifast.com) to try.

## Installation

1. Clone the repository
2. Install Python dependencies: `pip install -r requirements.txt`
3. Install frontend dependencies: `bun install`
4. Set environment variables for API keys
5. Run frontend watchers and the app: `bun run dev`

Frontend JS/CSS files under `app/static/` are build artifacts. They are generated from `app/assets/` and are intentionally not committed to git.

## Deployment

For production, deploy from the repo root with the provided `Dockerfile`. The image builds the frontend assets in a Bun builder stage and copies only the compiled files into the final Python runtime image.

For Coolify, use:

- Source directory: repo root
- Build pack: Dockerfile
- Dockerfile path: `./Dockerfile`
- Include Source Commit in Build: disabled

`docker-compose.yaml` can still be used for local or manual container runs, but the recommended Coolify production path is the Dockerfile build directly.

## API

- `GET /` Homepage
- `GET /{classifier_type}` Classification page (unspsc, etim, naics)
- `POST /{classifier_type}` Submit classification request
- `GET /health` Health check endpoint

## SEO

- Structured data markup (JSON-LD)
- FAQ schema for common questions
- Optimized meta descriptions and titles
- Breadcrumb navigation
- Semantic HTML structure
- Performance optimized loading
