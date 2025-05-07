# data/db/mongo_client.py
"""
MongoDB client manager: cached connections and utilities for getting and closing the database.
"""
import logging
from typing import Optional
import ssl

import certifi
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, ConfigurationError

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level cache for client and database
_mongo_client: Optional[MongoClient] = None
_mongo_db: Optional[Database] = None

def get_mongo_client() -> MongoClient:
    global _mongo_client
    if _mongo_client is None:
        try:
            # Establish a TLS-secured connection using system CA certificates
            # and allow invalid certificates and hostnames for development
            _mongo_client = MongoClient(
                settings.MONGODB_URI,
                tls=True,
                tlsCAFile=certifi.where(),
                tlsAllowInvalidCertificates=True,
                tlsAllowInvalidHostnames=True,
                serverSelectionTimeoutMS=5000,
                socketTimeoutMS=20000,
            )
            # Force a connection to verify settings
            _mongo_client.admin.command("ping")
            logger.info(f"MongoDB connection established to {settings.MONGODB_URI}")
        except (ConnectionFailure, ConfigurationError) as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    return _mongo_client

def get_mongo_db() -> Database:
    global _mongo_db
    if _mongo_db is None:
        client = get_mongo_client()
        _mongo_db = client[settings.MONGODB_DB_NAME]
        logger.info(f"Using MongoDB database: {settings.MONGODB_DB_NAME}")
    return _mongo_db

def close_mongo_connection() -> None:
    global _mongo_client, _mongo_db
    if _mongo_client:
        _mongo_client.close()
        logger.info("Closed MongoDB connection")
        _mongo_client = None
        _mongo_db = None

# Example usage
if __name__ == "__main__":
    try:
        db = get_mongo_db()
        print(f"Connected to database '{settings.MONGODB_DB_NAME}'. Collections: {db.list_collection_names()}")
    except Exception as e:
        print(f"MongoDB connection test failed: {e}")
    finally:
        close_mongo_connection()
