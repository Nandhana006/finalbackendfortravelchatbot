from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime
import os

load_dotenv()


class ChatHistoryService:

    def __init__(self):

        self.client = MongoClient(
            os.getenv("MONGO_URI")
        )

        self.db = self.client[
            os.getenv(
                "MONGO_DB",
                "travel_rag"
            )
        ]

        self.collection = (
            self.db["chat_history"]
        )

    def save_message(
        self,
        session_id,
        role,
        content
    ):

        self.collection.insert_one(
            {
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow()
            }
        )

    def get_history(
        self,
        session_id
    ):

        return list(
            self.collection.find(
                {
                    "session_id": session_id
                },
                {
                    "_id": 0
                }
            )
        )

    def get_sessions(self):

        pipeline = [
            {
                "$group": {
                    "_id": "$session_id",
                    "title": {
                        "$first": "$content"
                    },
                    "timestamp": {
                        "$max": "$timestamp"
                    }
                }
            },
            {
                "$sort": {
                    "timestamp": -1
                }
            }
        ]

        results = list(
            self.collection.aggregate(
                pipeline
            )
        )

        sessions = []

        for item in results:

            sessions.append(
                {
                    "session_id": item["_id"],
                    "title": item["title"][:50],
                    "timestamp": item["timestamp"]
                }
            )

        return sessions


chat_history_service = (
    ChatHistoryService()
)