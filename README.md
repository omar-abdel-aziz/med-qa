# Medical QA API

This repository provides a **FastAPI**-based local service for querying medical documents (PDFs or images) using Google’s Gemini LLM via the LangChain GenAI integration. It ingests a user’s medical document, builds a FAISS vector store, and answers questions with concise bullet-point summaries.

---

## 📂 Project Structure

```
medical-qa-api/
├── app/
│   ├── main.py          # FastAPI endpoints and core logic
│   ├── ocr.py           # PDF/image to text extraction
│   └── pipeline.py      # Text splitting, embedding, and FAISS index management
├── data/                # Runtime session data (auto-created)
├── .env                 # Environment variables (e.g. GOOGLE_API_KEY)
├── requirements.txt     # Python dependencies
└── README.md            # This documentation
```

---

## 📦 Features

- **OCR & Text Extraction**

  - PDF → images via `pdf2image` + text via `pytesseract`
  - Direct image OCR via `pytesseract`

- **Vector Store**

  - Chunks text with `CharacterTextSplitter` (1000-token chunks, 200 overlap)
  - Embeds with `HuggingFaceEmbeddings` (all‑MiniLM‑L6‑v2)
  - Indexes embeddings in FAISS, persisted per session

- **Retrieval‑Augmented QA**

  - Uses Google Gemini (`ChatGoogleGenerativeAI`) for LLM
  - Custom prompt produces concise bullet‑point summaries

- **Session Isolation**

  - Each user session stored under `./data/{session_id}`
  - Cleanup endpoint removes session data

---

## 🛠 Requirements

1. **Python 3.10+**
2. **Virtual Environment** (recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install** dependencies

   ```bash
   pip install -r requirements.txt
   ```

4. **Tesseract OCR** (for image and PDF OCR)

   ```bash
   # macOS
   brew install tesseract
   # Ubuntu
   sudo apt-get install tesseract-ocr
   ```

5. **Poppler** (for PDF → image conversion)

   ```bash
   # macOS
   brew install poppler
   # Ubuntu
   sudo apt-get install poppler-utils
   ```

6. **Google Gemini API Key**

   - Generate at [Google AI Studio → API Keys](https://ai.google.dev/gemini-api/docs/api-key)
   - Store in `.env`:

     ```ini
     GOOGLE_API_KEY=AIzaSy…<your_gemini_api_key>…
     ```

---

## ⚙️ Setup & Run

1. **Clone** repository

   ```bash
   git clone [https://github.com/omar-abdel-aziz/med-qa.git]
   cd medical-qa-api
   ```

2. **Activate** virtual environment

   ```bash
   source venv/bin/activate
   ```

3. **Install** dependencies

   ```bash
   pip install -r requirements.txt
   ```

4. **Create** `.env` with your `GOOGLE_API_KEY`
5. **Run** server

   ```bash
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```

---

## 🖥️ API Endpoints

All endpoints are prefixed by `/` on `http://127.0.0.1:8000`:

| Endpoint             | Method | Description                                                  |
| -------------------- | ------ | ------------------------------------------------------------ |
| `/upload`            | POST   | Upload a PDF/image. Returns `session_id`.                    |
| `/process/{session}` | POST   | Process uploaded file (OCR, chunk, embed, index).            |
| `/status/{session}`  | GET    | Check if processing complete. Returns `{ processed: bool }`. |
| `/query/{session}`   | POST   | Ask a question. Returns bullet‑point summary in `answer`.    |
| `/cleanup/{session}` | DELETE | Delete session data.                                         |

### 1. Upload

```http
POST /upload
Content-Type: multipart/form-data

Body:
  file: <PDF or image>

Response:
{
  "session_id": "<sid>"
}
```

### 2. Process

```http
POST /process/{sid}

Response:
{
  "status": "done"
}
```

### 3. Status

```http
GET /status/{sid}

Response:
{
  "processed": true
}
```

### 4. Query

```http
POST /query/{sid}
Content-Type: application/json

Body:
{
  "question": "Your medical question"
}

Response:
{
  "answer": [
    "- Bullet point 1",
    "- Bullet point 2",
    "…"
  ]
}
```

### 5. Cleanup

```http
DELETE /cleanup/{sid}

Response:
{
  "deleted": true
}
```

---

## 🔧 Internal Modules

- **`app/ocr.py`**: Extracts text via OCR
- **`app/pipeline.py`**: Splits, embeds, and persists FAISS index
- **`app/main.py`**: Defines FastAPI routes and RAG workflow

---

## 🌐 Frontend Example

Use Axios in React to interact:

```js
import axios from "axios";
const api = axios.create({ baseURL: "http://127.0.0.1:8000" });

// Upload
const { data } = await api.post("/upload", formData);
const sid = data.session_id;

// Process
await api.post(`/process/${sid}`);

// Query
const res = await api.post(`/query/${sid}`, { question: "..." });
console.log(res.data.answer);
```

---

## 📝 Notes

- **Local demo**—no external storage or paid services.
- All session data under `./data/{session_id}` is removed by `/cleanup`.
- For production, tighten CORS in `app/main.py`.

---
