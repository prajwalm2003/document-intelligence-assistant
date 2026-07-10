# core/qa_engine.py
# "Smart answer" mode: decides whether a question can be answered from
# the indexed documents, and falls back to general knowledge if not.

from core.llm import get_llm


def _build_history_text(chat_history):
    if not chat_history:
        return ""
    history_text = "\n\nPrevious conversation:\n"
    for msg in chat_history[-4:]:
        history_text += f"Human: {msg['question']}\nAssistant: {msg['answer']}\n"
    return history_text


def _is_question_relevant(llm, context, question):
    relevance_prompt = f"""Can this question be answered from the document context below?
Context: {context}
Question: {question}
Reply with ONLY: YES or NO"""
    relevance_response = llm.invoke(relevance_prompt)
    return "YES" in relevance_response.content.upper()


def get_answer(vectorstore, question, chat_history=None):
    chat_history = chat_history or []
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    history_text = _build_history_text(chat_history)
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    if _is_question_relevant(llm, context, question):
        prompt = f"""Answer based on the document context below.
{history_text}
Context: {context}
Question: {question}
Give a detailed answer. End with: "📄 Source: Answered from uploaded document"
"""
        response = llm.invoke(prompt)
        return response.content, relevant_docs, "document"

    prompt = f"""Answer using your general knowledge.
{history_text}
Question: {question}
End with: "🌐 Source: General knowledge"
"""
    response = llm.invoke(prompt)
    return response.content, [], "general"
