import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


st.set_page_config(page_title="RAG Document Assistant", page_icon=":material/description:", layout="wide")

st.title("RAG Document Assistant")
st.caption("Upload a PDF or TXT file, ask questions, and get answers with source snippets.")


if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "document_name" not in st.session_state:
    st.session_state.document_name = None


with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

    if uploaded_file and st.button("Index document", type="primary"):
        with st.spinner("Extracting text, chunking, and embedding..."):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type or "application/octet-stream",
                    )
                }
                response = requests.post(f"{API_URL}/upload", files=files, timeout=120)
                response.raise_for_status()
                data = response.json()

                st.session_state.collection_name = data["collection_name"]
                st.session_state.document_name = data["document_name"]

                st.success(f"Indexed {data['chunk_count']} chunks from {data['document_name']}.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI backend. Start it on port 8000.")
            except requests.exceptions.HTTPError:
                st.error(response.json().get("detail", "Upload failed."))
            except Exception as error:
                st.error(f"Upload failed: {error}")

    if st.session_state.document_name:
        st.info(f"Active document: {st.session_state.document_name}")


question = st.text_input("Ask a question about the uploaded document")

if st.button("Ask", disabled=not st.session_state.collection_name or not question.strip()):
    with st.spinner("Searching the document and generating an answer..."):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={
                    "collection_name": st.session_state.collection_name,
                    "question": question,
                },
                timeout=120,
            )
            response.raise_for_status()
            data = response.json()

            st.subheader("Answer")
            st.write(data["answer"])

            st.subheader("Sources")
            for source in data["sources"]:
                with st.expander(
                    f"{source['document_name']} - chunk {source['chunk_number']}",
                    expanded=False,
                ):
                    st.write(source["snippet"])
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the FastAPI backend. Start it on port 8000.")
        except requests.exceptions.HTTPError:
            st.error(response.json().get("detail", "Question failed."))
        except Exception as error:
            st.error(f"Question failed: {error}")


if not st.session_state.collection_name:
    st.info("Upload and index a document to begin.")
