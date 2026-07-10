# core/qa_generator.py
# Auto-generates MCQ questions from the indexed documents.
# Useful for teachers/students testing themselves on the material.

from core.llm import get_llm


def generate_qa(vectorstore, num_questions=5):
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    docs = retriever.invoke("main concepts and important topics")
    context = "\n\n".join(doc.page_content for doc in docs)[:4000]

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
