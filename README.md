# Classifast - AI-Powered Classification Tool

Classifast is a web application that provides AI-powered classification for products and services using international standards like UNSPSC, ETIM, and NAICS. Built with FastAPI and modern web technologies, it offers fast, accurate semantic search capabilities for automated categorization.

## Features

- 🚀 **Fast Classification**: AI-powered semantic search using state-of-the-art embedding models
- 🎯 **High Accuracy**: Confidence scores for each classification result
- 📊 **Multiple Standards**: Support for UNSPSC, ETIM, and NAICS classification systems
- 🔄 **Bulk Processing**: Classify large datasets efficiently
- 🌐 **Modern UI**: Clean, responsive interface built with Tailwind CSS
- 🔍 **SEO Optimized**: Structured data, meta tags, and performance optimized

## Supported Classification Standards

### UNSPSC (United Nations Standard Products and Services Code)
- Global standard for product and service categorization
- Improves spend analytics and procurement processes
- Version: UNv260801 (August 14, 2023)

### ETIM (Technical Information Model)
- Open standard for technical product classification
- Specialized for electrical and technical products
- Version: 10.0 (2024-12-10)

### NAICS (North American Industry Classification System)
- Industry classification for business activities
- Essential for government contracting and reporting

## Technology Stack

- **Backend**: FastAPI with Python
- **Frontend**: HTML5, Tailwind CSS, HTMX
- **Vector Database**: Qdrant for semantic search
- **AI Models**: Google Gemini embedding models
- **Deployment**: Docker containerized

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables for API keys
4. Run with: `uvicorn app.main:app --reload`

## API Endpoints

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

## Contributing

Contributions welcome! Please read our contributing guidelines before submitting PRs.

## License

© 2025 Dmitry Matv

---

For more information, visit [classifast.com](https://classifast.com)
