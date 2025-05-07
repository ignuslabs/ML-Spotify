import os
from pymongo import MongoClient

uri = os.getenv("MONGODB_URI")
print("MONGODB_URI=", repr(uri))

client = MongoClient(uri, serverSelectionTimeoutMS=5000)
print("OK, connected:", client.list_database_names())