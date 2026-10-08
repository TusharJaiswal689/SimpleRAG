# College Notes RAG — Capstone Project

This is a small capstone project for turning the theory of Retrieval-Augmented Generation (RAG) into a working application. It lets you upload college notes and ask questions about them. The goal is to explore how document ingestion, embeddings, vector search, and a language model work together—not to provide a production-ready service.
**NOTE:** System prompt can be changed to suite other domains if you wish to upload some other documents.

## What this project demonstrates

A language model can produce a more grounded answer when relevant material is retrieved and included in its prompt. This project makes that process concrete:

1. **Ingest:** Upload a PDF or DOCX through the web interface. The API saves it under `data/documents/` and extracts its text.
2. **Chunk:** Split the extracted text into smaller, overlapping pieces. Chunk size and overlap are configurable.
3. **Embed and index:** Use Ollama's embedding model to turn each piece into a vector, then store the vectors and text metadata in a Pinecone index.
4. **Prepare the query:** When chat history is available, use the local Ollama generation model to rewrite the latest question as a standalone search query.
5. **Retrieve:** Embed that search query and ask Pinecone for the most similar chunks.
6. **Generate:** Put the original question, conversation history, and retrieved chunks into a RAG prompt. Ollama generates and streams the response back to the browser.

In short:

```text
PDF/DOCX -> text -> chunks -> embeddings -> Pinecone
                                             ^
question -> optional rewrite -> embedding -> similarity search
                                             |
                    retrieved chunks + question + history
                                             |
                                  Ollama answer stream
```

## Technology used

- **FastAPI** for the API and serving the web interface
- **LlamaIndex file readers and sentence splitter** for document loading and chunking
- **Ollama** for local text embeddings, query rewriting, and answer generation
- **Pinecone** for hosted vector storage and similarity search
- **HTML, CSS, and JavaScript** for a minimal browser-based chat interface

## Prerequisites

- Python and pip
- [Ollama](https://ollama.com/) installed and running locally
- The configured Ollama models available locally (defaults: `qwen3:8b` and `nomic-embed-text`)
- A Pinecone account and an index whose vector dimension matches the configured embedding model

Pull the default models with:

```powershell
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

If Ollama is not already running as a service, start it in a separate terminal:

```powershell
ollama serve
```

## Setup

1. Clone the repository and open a terminal in the project directory.
2. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Create a local `.env` file in the project root with your Pinecone settings. Keep credentials private; do not commit this file.

   ```dotenv
   PINECONE_API_KEY=your-pinecone-api-key
   PINECONE_INDEX_NAME=college-notes
   ```

   The settings also support these optional values, which have defaults in the application:

   | Variable | Default | Purpose |
   | --- | --- | --- |
   | `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
   | `OLLAMA_GENERATION_MODEL` | `qwen3:8b` | Model used for query rewriting and answer generation |
   | `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Model used to embed document chunks and queries |
   | `PINECONE_INDEX_NAME` | `college-notes` | Pinecone index name |
   | `TOP_K` | `5` | Number of matching chunks retrieved |
   | `CHUNK_SIZE` | `500` | Target chunk size |
   | `CHUNK_OVERLAP` | `50` | Overlap between neighboring chunks |

5. Make sure the Pinecone index exists and is configured for the embedding dimensions of the selected Ollama embedding model.

## Run the app

From the project root, start the development server:

```powershell
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) for the chat interface. FastAPI's interactive API documentation is available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Try it

1. Upload a PDF or DOCX containing notes using the attachment control.
2. Wait for ingestion to finish; the upload response reports how many chunks were indexed.
3. Ask a question about the uploaded material.
4. Continue the conversation to see how history can help rewrite follow-up questions for retrieval.

The main API routes are:

| Method | Route | Purpose |
| --- | --- | --- |
| `POST` | `/documents/upload` | Upload and index a PDF or DOCX |
| `POST` | `/chat/` | Submit a question and receive a streamed answer |

## Project layout

```text
app/
  api/           FastAPI routes and request/response schemas
  core/          Settings and dependency wiring
  embedder/      Ollama embedding client
  ingestion/     Document loading, chunking, and Pinecone indexing
  llm/           Ollama text-generation client
  prompts/       RAG prompt construction
  retrieval/     Query rewriting and Pinecone retrieval
  services/      Ingestion and answer-generation workflows
frontend/        Browser chat interface
data/documents/  Uploaded documents (created/used by the app)
tests/           Tests for ingestion, retrieval, and RAG behavior
```

## Learning notes and limitations

- This is an educational prototype, not a production application. It has not been designed or assessed for production security, reliability, privacy, scalability, or answer quality.
- Pinecone is a hosted service, so document text and vector metadata are sent there. Ollama inference is configured locally by default.
- The project does not authenticate users or isolate uploaded documents by user. Do not expose it publicly or upload sensitive material.
- The upload route accepts files but does not implement production-grade file validation, size limits, or robust error handling.
- Retrieval quality depends on the source documents, chunking settings, embedding model, Pinecone index configuration, and `TOP_K`.
- Generated answers can be incomplete or incorrect. Check important information against the original notes.
- The included tests exercise parts of the workflows; they do not establish production readiness or end-to-end answer accuracy.

## Exploring RAG further

Useful experiments for the capstone include changing chunk size and overlap, comparing embedding or generation models, varying `TOP_K`, inspecting retrieved chunks, and testing how query rewriting affects follow-up questions. Change one setting at a time and compare the retrieved evidence and generated response.