# rag_pipeline.py
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from doc_loader import load_any_file
from dotenv import load_dotenv
import os
import pandas as pd

load_dotenv()

def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY"),
        temperature=0
    )

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

def load_and_index_files(file_paths):
    """
    UPGRADED: Now handles ANY file type, not just PDFs
    WHY: Real enterprise apps handle multiple document formats
    """
    all_chunks = []

    for file_path in file_paths:
        try:
            documents = load_any_file(file_path)
            file_type = documents[0].metadata.get("file_type", "Unknown")

            # Only split text-heavy files
            # WHY: CSV/Excel rows are already small — no need to split further
            if file_type in ["PDF", "DOCX", "TXT", "PPTX"]:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks = splitter.split_documents(documents)
            else:
                chunks = documents  # CSV/Excel already chunked in loader

            all_chunks.extend(chunks)
            print(f"✅ {len(chunks)} chunks from {os.path.basename(file_path)}")

        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
            continue

    print(f"\n🔢 Total chunks: {len(all_chunks)}")
    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=get_embeddings(),
        persist_directory="./chroma_db"
    )
    print("✅ All files indexed!")
    return vectorstore


def get_answer(vectorstore, question, chat_history=[]):
    """Smart Answer Mode — PDF or General Knowledge"""
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    history_text = ""
    if chat_history:
        history_text = "\n\nPrevious conversation:\n"
        for msg in chat_history[-4:]:
            history_text += f"Human: {msg['question']}\nAssistant: {msg['answer']}\n"

    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    # Check relevance
    relevance_prompt = f"""Can this question be answered from the document context below?
Context: {context}
Question: {question}
Reply with ONLY: YES or NO"""

    relevance_response = llm.invoke(relevance_prompt)
    is_relevant = "YES" in relevance_response.content.upper()

    if is_relevant:
        prompt = f"""Answer based on the document context below.
{history_text}
Context: {context}
Question: {question}
Give a detailed answer. End with: "📄 Source: Answered from uploaded document"
"""
        response = llm.invoke(prompt)
        return response.content, relevant_docs, "document"
    else:
        prompt = f"""Answer using your general knowledge.
{history_text}
Question: {question}
End with: "🌐 Source: General knowledge"
"""
        response = llm.invoke(prompt)
        return response.content, [], "general"


# ==========================================
# UPGRADE: AUTO Q&A GENERATOR
# WHY: Automatically generates MCQ questions from document
# Shows NLP + prompt engineering skills
# Teachers use this to test students — very practical feature
# ==========================================
def generate_qa(vectorstore, num_questions=5):
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke("main concepts and important topics")
    context = "\n\n".join([doc.page_content for doc in docs])[:4000]

    prompt = f"""You are an expert teacher. Generate exactly {num_questions} multiple choice questions from the document below.

For each question follow this EXACT format:
Q1. [Question here]
A) [Option 1]
B) [Option 2]  
C) [Option 3]
D) [Option 4]
Answer: [Correct letter]
Explanation: [Why this is correct, 1 sentence]

---

Document:
{context}

Generate {num_questions} questions now:"""

    response = llm.invoke(prompt)
    return response.content


# ==========================================
# UPGRADE: TABLE/DATA EXTRACTOR
# WHY: Extracts structured data from documents
# Shows data engineering skills — very valued in industry
# ==========================================
def extract_tables(file_paths):
    llm = get_llm()
    results = []

    for file_path in file_paths:
        ext = os.path.splitext(file_path)[1].lower()
        filename = os.path.basename(file_path)

        # For CSV/Excel — already tabular, just clean it up
        if ext in [".csv", ".xlsx", ".xls"]:
            if ext == ".csv":
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            results.append({
                "file": filename,
                "type": "Structured Data",
                "content": df,
                "summary": f"{len(df)} rows × {len(df.columns)} columns"
            })

        # For PDFs/DOCX — use Gemini to extract table-like data
        elif ext in [".pdf", ".docx", ".txt", ".pptx"]:
            docs = load_any_file(file_path)
            text = "\n".join([d.page_content for d in docs])[:4000]

            prompt = f"""Extract ALL tables, lists, and structured data from this document.
Format each table clearly with headers and rows.
If no tables found, extract key-value pairs or bullet point lists.
Document:
{text}"""

            response = llm.invoke(prompt)
            results.append({
                "file": filename,
                "type": "Extracted Data",
                "content": response.content,
                "summary": "AI-extracted structured data"
            })

    return results


def generate_summary(file_paths):
    """Generate structured summary from any file type"""
    llm = get_llm()
    all_text = ""

    for file_path in file_paths:
        docs = load_any_file(file_path)
        for doc in docs[:3]:
            all_text += doc.page_content + "\n"

    all_text = all_text[:5000]

    prompt = f"""Analyze this document and provide:
1. 📌 Main Topic (1 sentence)
2. 🔑 Key Points (5 bullet points)
3. 💡 Important Terms or Concepts
4. 📊 Conclusion or Findings (2-3 sentences)

Document:
{all_text}"""

    response = llm.invoke(prompt)
    return response.content