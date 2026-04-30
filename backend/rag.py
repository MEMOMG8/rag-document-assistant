import os
import uuid
from typing import Any

import chromadb
from openai import OpenAI


CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 4


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "replace_with_your_new_openai_key":
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to backend/.env.")
    return api_key


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if not text.strip():
        raise ValueError("Cannot chunk an empty document.")

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


class RAGPipeline:
    def __init__(self, persist_directory: str = "chroma_db") -> None:
        api_key = get_openai_api_key()
        self.client = OpenAI(api_key=api_key)
        self.chroma_client = chromadb.PersistentClient(path=persist_directory)

    def index_document(self, text: str, document_name: str) -> dict[str, Any]:
        chunks = chunk_text(text)
        collection_name = f"doc_{uuid.uuid4().hex[:12]}"
        collection = self.chroma_client.create_collection(name=collection_name)
        embeddings = self._embed_texts(chunks)

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {
                "document_name": document_name,
                "chunk_number": index + 1,
            }
            for index in range(len(chunks))
        ]

        collection.add(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)

        return {
            "collection_name": collection_name,
            "document_name": document_name,
            "chunk_count": len(chunks),
        }

    def answer_question(self, collection_name: str, question: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        collection = self.chroma_client.get_collection(name=collection_name)
        question_embedding = self._embed_texts([question])[0]

        results = collection.query(query_embeddings=[question_embedding], n_results=top_k)
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            raise ValueError("No relevant context was found for this document.")

        sources = [
            {
                "chunk_number": metadata.get("chunk_number", index + 1),
                "document_name": metadata.get("document_name", "Uploaded document"),
                "snippet": document[:500],
            }
            for index, (document, metadata) in enumerate(zip(documents, metadatas))
        ]

        context = "\n\n".join(
            f"Source {index + 1}:\n{document}" for index, document in enumerate(documents)
        )

        response = self.client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful RAG document assistant. Answer only from the provided context. "
                        "If the answer is not in the context, say you do not know. Cite sources using "
                        "Source 1, Source 2, and so on."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0.2,
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": sources,
        }

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]
