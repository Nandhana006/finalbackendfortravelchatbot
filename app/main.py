from fastapi import (
    FastAPI,
    UploadFile,
    File
)

from fastapi.responses import HTMLResponse

from fastapi.middleware.cors import CORSMiddleware

from fastapi.openapi.utils import get_openapi

from pydantic import BaseModel

import os
import time

from app.services.document_loader import (
    load_document
)

from app.services.rag_service import (
    process_document,
    chat_with_rag
)

from app.utils.log import logger


app = FastAPI()


UPLOAD_DIR = "app/uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


class ChatRequest(BaseModel):

    question: str


def custom_openapi():

    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        routes=app.routes,
    )

    upload_schema = (
        openapi_schema
        .get("components", {})
        .get("schemas", {})
        .get("Body_upload_files_upload_post")
    )

    if upload_schema is not None:

        files_prop = (
            upload_schema
            .get("properties", {})
            .get("files")
        )

        if (
            files_prop is not None
            and files_prop.get("items") is not None
        ):
            files_prop["items"]["format"] = "binary"
            files_prop["items"]["contentMediaType"] = (
                "application/pdf"
            )

    app.openapi_schema = openapi_schema

    return app.openapi_schema


app.openapi = custom_openapi


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():

    return {
        "message": "Travel RAG Backend Running"
    }


@app.get(
    "/upload",
    response_class=HTMLResponse
)
def upload_form():

    return """
    <html>
      <head>
        <title>Upload PDF Documents</title>
      </head>

      <body>

        <h1>Upload multiple PDF files</h1>

        <p>
          Select one or more PDF files
        </p>

        <form
          action="/upload"
          enctype="multipart/form-data"
          method="post"
        >

          <input
            name="files"
            type="file"
            multiple
            accept=".pdf"
          />

          <button type="submit">
            Upload
          </button>

        </form>

      </body>
    </html>
    """


@app.post("/upload")
async def upload_files(
    files: list[UploadFile] = File(
        ...,
        description="Upload one or more PDF files"
    )
):

    upload_start = time.time()

    if not files:

        return {
            "message": "No files uploaded"
        }

    all_pages = []

    saved_files = []

    errors = []

    logger.info(
        f"Uploaded {len(files)} files"
    )

    for file in files:

        if not file.filename.lower().endswith(
            ".pdf"
        ):

            errors.append(
                f"Skipped non-PDF file: {file.filename}"
            )

            continue

        logger.info(
            f"Processing file: {file.filename}"
        )

        file_path = os.path.join(
            UPLOAD_DIR,
            file.filename
        )

        contents = await file.read()

        if not contents:

            errors.append(
                f"Empty file: {file.filename}"
            )

            continue

        with open(
            file_path,
            "wb"
        ) as f:

            f.write(contents)

        saved_files.append(
            file.filename
        )

        try:

            pages = load_document(
                file_path
            )

            all_pages.extend(
                pages
            )

            logger.info(
                f"Loaded {len(pages)} pages from {file.filename}"
            )

        except Exception as exc:

            errors.append(
                f"Failed to parse {file.filename}: {exc}"
            )

            logger.error(
                f"Failed to parse {file.filename}: {exc}"
            )

            continue

    processed_chunks = 0

    if all_pages:

        processed_chunks = process_document(
            all_pages
        )

    total_time = (
        time.time() - upload_start
    )

    logger.info(
        f"Total upload completed in {total_time:.1f}s - {processed_chunks} chunks"
    )

    return {
        "message": "Upload completed",
        "files": saved_files,
        "processed_chunks": processed_chunks,
        "total_time_seconds": round(
            total_time,
            2
        ),
        "errors": errors
    }


@app.post("/chat")
def chat(
    request: ChatRequest
):

    result = chat_with_rag(
        request.question
    )

    return result