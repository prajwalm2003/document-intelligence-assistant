# 🧠 Document Intelligence Assistant

AI-powered multi-format document Q&A system using RAG (Retrieval Augmented Generation)

## 🚀 Features
- 📄 Multi-format support — PDF, DOCX, TXT, CSV, XLSX, PPTX
- 💬 Conversational chat with memory
- 🧠 Smart Answer Routing — PDF vs General Knowledge
- 📍 Source citations with exact page numbers
- ❓ Auto MCQ Generator
- 📊 Table Extractor
- 📝 Document Summarizer

## 📁 Project Structure

```
pdf-qa-app/
├── app.py                  # Entry point — wires sidebar + tabs together
├── config.py               # Env vars, model names, chunk sizes
├── requirements.txt
├── loaders/                # One module per file format
│   ├── base.py             # Shared Document metadata helper
│   ├── pdf_loader.py
│   ├── docx_loader.py
│   ├── txt_loader.py
│   ├── tabular_loader.py   # CSV / Excel
│   ├── pptx_loader.py
│   └── factory.py          # load_any_file() dispatcher (Factory pattern)
├── core/                   # Business logic, no Streamlit imports here
│   ├── llm.py               # Gemini model factory
│   ├── embeddings.py        # Local HuggingFace embeddings
│   ├── indexer.py           # load -> chunk -> embed -> persist
│   ├── qa_engine.py         # Chat answering (doc vs general knowledge)
│   ├── summarizer.py
│   ├── qa_generator.py      # MCQ generation
│   └── table_extractor.py
├── ui/                      # Streamlit rendering, split by screen
│   ├── sidebar.py
│   ├── landing.py
│   ├── tab_chat.py
│   ├── tab_summary.py
│   ├── tab_qa.py
│   └── tab_tables.py
└── tests/
    ├── test_loaders.py
    └── test_indexer.py
```

Run the test suite with:
```
pytest
```

## 🛠️ Tech Stack
- **LangChain** — RAG orchestration
- **Google Gemini 2.5 Flash** — LLM
- **ChromaDB** — Vector database
- **Streamlit** — Frontend UI
- **Python 3.11**

## ⚙️ Setup

1. Clone the repo
   git clone https://github.com/your-username/document-intelligence-assistant.git

2. Create virtual environment
   python -m venv venv
   venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Add your API key — create .env file
   GEMINI_API_KEY=your-key-here

5. Run the app
   streamlit run app.py

## 📸 Demo
[Add screenshot here after running]
