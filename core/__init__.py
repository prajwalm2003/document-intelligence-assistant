# core package
# Business logic for the RAG pipeline, split by responsibility:
#   embeddings.py / llm.py -> model construction
#   indexer.py             -> loading + chunking + vectorstore build
#   qa_engine.py           -> chat / question answering
#   summarizer.py          -> document summary
#   qa_generator.py        -> MCQ generation
#   table_extractor.py     -> structured data extraction
