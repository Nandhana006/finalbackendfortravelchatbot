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


class QdrantService:

    COLLECTION_NAME = "travel_docs"

    def __init__(self):

        self.client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=120
        )

        self.create_collection()

    def create_collection(self):

        collections = (
            self.client.get_collections()
            .collections
        )

        existing = [
            collection.name
            for collection in collections
        ]

        if self.COLLECTION_NAME not in existing:

            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )

    def upload_chunks(
        self,
        chunks,
        embeddings
    ):

        batch_size = 500

        for i in range(
            0,
            len(chunks),
            batch_size
        ):

            points = []

            batch_chunks = (
                chunks[i:i + batch_size]
            )

            batch_embeddings = (
                embeddings[i:i + batch_size]
            )

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

            self.client.upsert(

                collection_name=
                self.COLLECTION_NAME,

                points=points,

                wait=True

            )

            print(
                f"Uploaded batch "
                f"{i // batch_size + 1}"
            )

    def search_similar(

        self,

        query_embedding,

        filename=None,

        top_k=7,

        score_threshold=0.3

    ):

        results = self.client.query_points(

            collection_name=
            self.COLLECTION_NAME,

            query=query_embedding,

            limit=100,

            score_threshold=score_threshold,

            with_payload=True,

            with_vectors=False

        )

        filtered = []

        for point in results.points:

            if not point.payload:
                continue

            if filename is not None:

                if (
                    point.payload.get("source")
                    != filename
                ):
                    continue

            filtered.append(

                {

                    "text":
                    point.payload["text"],

                    "page":
                    point.payload.get("page"),

                    "source":
                    point.payload.get("source"),

                    "score":
                    point.score

                }

            )

        return filtered[:top_k]

    def search_by_page(
        self,
        page_number
    ):

        points, _ = self.client.scroll(

            collection_name=
            self.COLLECTION_NAME,

            limit=10000,

            with_payload=True,

            with_vectors=False

        )

        return [

            {

                "text":
                point.payload["text"],

                "page":
                point.payload.get("page"),

                "source":
                point.payload.get("source")

            }

            for point in points

            if (

                point.payload

                and point.payload.get("page")
                == page_number

            )

        ]

    def get_collection_info(
        self
    ):

        return self.client.get_collection(

            collection_name=
            self.COLLECTION_NAME

        )

    def file_exists(
        self,
        filename
    ):

        filename = (
            os.path.basename(filename)
            .lower()
        )

        offset = None

        while True:

            points, offset = (
                self.client.scroll(

                    collection_name=
                    self.COLLECTION_NAME,

                    limit=500,

                    offset=offset,

                    with_payload=True,

                    with_vectors=False

                )
            )

            for point in points:

                if not point.payload:
                    continue

                source = (
                    point.payload.get(
                        "source",
                        ""
                    )
                )

                source = (
                    os.path.basename(source)
                    .lower()
                )

                if source == filename:
                    return True

            if offset is None:
                break

        return False


qdrant_service = QdrantService()