import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from document_loader import SUPPORTED_EXTENSIONS, extract_text
from rag import RAGPipeline


load_dotenv()

app = FastAPI(title="RAG Document Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    collection_name: str
    question: str


def get_rag_pipeline() -> RAGPipeline:
    try:
        return RAGPipeline()
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict[str, object]:
    extension = Path(file.filename or "").suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as temp_file:
        temp_file.write(contents)
        temp_path = temp_file.name

    try:
        text = extract_text(temp_path)
        rag_pipeline = get_rag_pipeline()
        return rag_pipeline.index_document(text, file.filename or "uploaded_document")
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    finally:
        os.remove(temp_path)


@app.post("/ask")
def ask_question(request: QuestionRequest) -> dict[str, object]:
    try:
        rag_pipeline = get_rag_pipeline()
        return rag_pipeline.answer_question(request.collection_name, request.question)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {error}") from error
