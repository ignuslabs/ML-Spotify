# debug_config.py
from app.core.config import settings
print("Loaded URI:", settings.MONGODB_URI)
print("Loaded DB Name:", settings.MONGODB_DB_NAME)