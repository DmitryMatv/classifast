from dotenv import load_dotenv

# Module-level settings across the app read os.environ at import time, so the
# .env file must load before any app module.
load_dotenv()
