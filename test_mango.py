from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

client = MongoClient(
    os.getenv("MONGO_URI")
)

db = client["travel_rag"]

print("Connected!")

print(
    db.list_collection_names()
)