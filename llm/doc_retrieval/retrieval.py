from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# this is required when exporting the FAISS store
# from langchain_community.document_loaders import DirectoryLoader

from app.cache import cache

def get_docs_faiss(prompt: str):
    db = cache.get('db')
    if db is None:
        embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        db = FAISS.load_local("faiss_data/ctftime", embed_model, allow_dangerous_deserialization=True)
        cache.set('db', db)

    # k in config
    docs_filter = db.similarity_search_with_score(prompt, k=3)

    return docs_filter