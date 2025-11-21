from fastapi.templating import Jinja2Templates
from slowapi import Limiter
from slowapi.util import get_remote_address

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])

# Separate limiter for RapidAPI endpoints
rapid_limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["60/minute"],  # Stricter limits for external API
)

# Setup Jinja2 templates
templates = Jinja2Templates(directory="app/templates")
