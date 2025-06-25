# Classifast

Classifast is a web application that provides easy classification of any text input according to international product and service standards like UNSPSC, NAICS, ISIC, ETIM. Built with FastAPI and modern web technologies, it offers fast, accurate semantic search capabilities for automated yet intelligent categorization.

## Features

- 🚀 **Fast Classification**: Semantic search using advanced embedding models
- 🎯 **High Accuracy**: Confidence scores for each classification result
- 📊 **Multiple Standards**: Support for UNSPSC, ETIM, and NAICS classification standards
- 🔄 **Bulk Processing**: Classify large datasets efficiently
- 🌐 **Modern Interface**: Clean, responsive design built with Tailwind CSS
- 🔍 **SEO Optimized**: Structured data, meta tags, and performance optimized

## Supported Classification Standards

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

## Tech Stack

- **Backend**: FastAPI with Python
- **Frontend**: HTML5, Tailwind CSS, HTMX
- **Vector Database**: Qdrant for semantic search
- **Embedding Models**: Google Gemini
- **Deployment**: Docker containerized

## Live Demo

Visit the working preview at [classifast.com](https://classifast.com) to try.

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables for API keys
4. Run with: `uvicorn app.main:app --reload`

## API Endpoints (WIP)

- `GET /` - Homepage
- `GET /{classifier_type}` - Classification page (etim, unspsc)
- `POST /{classifier_type}` - Submit classification request
- `GET /health` - Health check endpoint

## SEO Features

- Structured data markup (JSON-LD)
- FAQ schema for common questions
- Optimized meta descriptions and titles
- Breadcrumb navigation
- Semantic HTML structure
- Performance optimized loading

---

2025 Dmitry Matv
