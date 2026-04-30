# RAG Document Assistant

A portfolio-ready Retrieval-Augmented Generation (RAG) app that lets users upload a PDF or TXT document, ask questions about it, and receive answers grounded in retrieved source snippets.

## Features

- Upload PDF or TXT documents from a Streamlit interface
- Extract text from PDFs with `pypdf`
- Split documents into overlapping chunks
- Generate OpenAI embeddings for each chunk
- Store and search document chunks with Chroma vector database
- Ask natural-language questions about the uploaded document
- Generate answers with OpenAI chat completion
- Display cited source snippets used to produce each answer
- Handle missing API keys, empty files, and unsupported file formats

## Tech Stack

- Python
- FastAPI
- Streamlit
- OpenAI API
- ChromaDB
- pypdf
- python-dotenv

## Project Structure

```text
rag-document-assistant/
  backend/
    main.py
    rag.py
    document_loader.py
    requirements.txt
    .env.example
  frontend/
    app.py
    requirements.txt
  README.md
```

## Setup

### 1. Clone or open the project

```bash
cd rag-document-assistant
```

### 2. Create and activate a virtual environment

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install backend dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Install frontend dependencies

```bash
pip install -r frontend/requirements.txt
```

### 5. Configure environment variables

Copy the example environment file:

```bash
copy backend\.env.example backend\.env
```

On macOS or Linux:

```bash
cp backend/.env.example backend/.env
```

Then edit `backend/.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

## Run Locally

Open one terminal for the backend:

```bash
cd rag-document-assistant/backend
uvicorn main:app --reload
```

Open a second terminal for the frontend:

```bash
cd rag-document-assistant/frontend
streamlit run app.py
```

Then open the Streamlit URL, usually:

```text
http://localhost:8501
```

The FastAPI docs are available at:

```text
http://127.0.0.1:8000/docs
```

## How It Works

1. A user uploads a PDF or TXT file in Streamlit.
2. The frontend sends the file to the FastAPI `/upload` endpoint.
3. The backend extracts text from the file.
4. The extracted text is split into overlapping chunks.
5. Each chunk is embedded with OpenAI embeddings.
6. Chunks and metadata are stored in a local Chroma collection.
7. The user asks a question through the frontend.
8. The backend retrieves the most relevant chunks from Chroma.
9. Retrieved context and the user question are sent to the OpenAI chat model.
10. The answer and source snippets are returned to the frontend.

## API Endpoints

### `GET /health`

Returns a simple health check.

### `POST /upload`

Accepts a PDF or TXT file, extracts text, chunks it, embeds it, and stores it in Chroma.

### `POST /ask`

Accepts a Chroma collection name and a question, then returns an answer with source snippets.

Example request:

```json
{
  "collection_name": "doc_abc123",
  "question": "What are the main findings in this document?"
}
```

## Screenshots

Add screenshots here after running the app locally:

- Document upload screen
- Successful indexing message
- Question and answer view
- Source snippet expanders

```text
screenshots/
  upload.png
  answer-with-sources.png
```

## Error Handling

The app includes clear error messages for:

- Missing `OPENAI_API_KEY`
- Empty uploaded files
- Unsupported file formats
- Documents with no readable text
- Backend connection issues from the frontend
- Empty questions

## Resume Bullets

- Built a full-stack RAG document assistant using FastAPI, Streamlit, OpenAI embeddings, and ChromaDB to answer user questions from uploaded PDF and TXT files.
- Implemented document parsing, text chunking, vector indexing, semantic retrieval, and grounded LLM response generation with cited source snippets.
- Designed beginner-friendly API endpoints for document upload and question answering, including validation for missing API keys, empty files, and unsupported formats.
- Created an interactive Streamlit frontend that lets users upload documents, trigger indexing, ask natural-language questions, and inspect retrieved evidence.
- Structured a production-style AI project with environment configuration, dependency files, modular backend code, and clear local setup documentation.

## Future Improvements

- Add user accounts and per-user document history
- Support multiple active documents at once
- Add OCR for scanned PDFs
- Add Docker and Docker Compose
- Add automated tests for parsing, chunking, and API behavior
- Add downloadable chat transcripts
