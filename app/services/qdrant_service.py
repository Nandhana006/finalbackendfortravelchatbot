import os
import uuid

from dotenv import load_dotenv

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

load_dotenv()

COLLECTION_NAME = "travel_docs"

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=120
)


def create_collection():

    collections = client.get_collections().collections

    existing = [
        collection.name
        for collection in collections
    ]

    if COLLECTION_NAME not in existing:

        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=384,
                distance=Distance.COSINE
            )
        )


create_collection()


def upload_chunks(chunks, embeddings):

    batch_size = 500

    for i in range(
        0,
        len(chunks),
        batch_size
    ):

        points = []

        batch_chunks = chunks[
            i:i + batch_size
        ]

        batch_embeddings = embeddings[
            i:i + batch_size
        ]

        for chunk_data, embedding in zip(
            batch_chunks,
            batch_embeddings
        ):

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "text": chunk_data["text"],
                        "page": chunk_data["page"],
                        "source": chunk_data["source"]
                    }
                )
            )

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )

        print(
            f"Uploaded batch {i // batch_size + 1}"
        )


def search_similar(query_embedding, top_k=10, score_threshold=None):

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=top_k,
        score_threshold=score_threshold,
        with_payload=True,
        with_vectors=False
    )

    return [
        {
            "text": point.payload["text"],
            "page": point.payload.get("page"),
            "source": point.payload.get("source")
        }
        for point in results.points
        if point.payload and "text" in point.payload
    ]


def search_by_page(page_number):

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )

    return [
        {
            "text": point.payload["text"],
            "page": point.payload.get("page"),
            "source": point.payload.get("source")
        }
        for point in points
        if (
            point.payload
            and point.payload.get("page") == page_number
        )
    ]


def get_collection_info():

    return client.get_collection(
        collection_name=COLLECTION_NAME
    )

def file_exists(filename):

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=10000,
        with_payload=True,
        with_vectors=False
    )

    for point in points:

        if (
            point.payload
            and point.payload.get("source") == filename
        ):
            return True

    return False
