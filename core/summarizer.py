# core/summarizer.py
from core.llm import get_llm
from loaders import load_any_file


def generate_summary(file_paths):
    """Generate a structured summary across all uploaded files."""
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
