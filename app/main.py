from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from app.routes.chat import router as chat_router
from app.routes.upload import router as upload_router

app = FastAPI(
    title="Travel Semantic Search API",
    version="1.0.0"
)


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

    if upload_schema:

        files_prop = (
            upload_schema
            .get("properties", {})
            .get("files")
        )

        if files_prop and files_prop.get("items"):

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
        "message": "Travel Semantic Search Backend Running"
    }
app.include_router(upload_router)
app.include_router(chat_router)


