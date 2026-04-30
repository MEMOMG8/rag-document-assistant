from pathlib import Path

from pypdf import PdfReader


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def extract_text(file_path: str) -> str:
    """Extract text from a PDF or TXT file."""
    path = Path(file_path)
    extension = path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Unsupported file format. Please upload a PDF or TXT file.")

    if extension == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        text = _extract_pdf_text(path)

    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("The uploaded file does not contain readable text.")

    return cleaned_text


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            pages.append(f"[Page {page_number}]\n{page_text}")

    return "\n\n".join(pages)
