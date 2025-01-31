import dotenv
import langchain
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.document_loaders import DirectoryLoader

from utils.langchain_adapter import ChatOpenRouter

# issue https://github.com/langchain-ai/langchain/issues/4164?ref=gettingstarted.ai
langchain.verbose = False
# https://stackoverflow.com/questions/78552532/what-does-the-error-module-langchain-has-no-attribute-verbose-refer-to
langchain.debug = False
langchain.llm_cache = False

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, only say that "it is an irrelevant question, please talk about CTFtime."
If you know the answer, keep the answer as detailed as possible and Do not need to explain the answer nor make up an answer.
Always say "thanks for asking!" at the end of the answer.

Question: {question}

Context: {context}"""

def format_docs(docs):
    # TextLoader auto converts single linebreak to double linebreaks, remove it to avoid wrong context intepretation
    return "\n\n".join(doc[0].page_content.replace('\n\n', '\n') for doc in docs)

if __name__ == '__main__':

    # chat_model = ChatOpenRouter(model_name="meta-llama/llama-3.2-1b-instruct:free")
    chat_model = ChatOpenRouter(model_name="meta-llama/llama-3.1-70b-instruct:free")
    # chat_model = ChatOpenRouter(model_name="mistralai/mistral-small-24b-instruct-2501")

    # load data and get embeddings
    # loader = DirectoryLoader('./raw_data/ctftime', use_multithreading=False)
    # docs = loader.load()
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    # db = FAISS.from_documents(docs, embed_model)

    # pre-save the vectors of docs with the embeddings
    # db.save_local('faiss_data/ctftime')
    db = FAISS.load_local("faiss_data/ctftime", embed_model, allow_dangerous_deserialization=True)

    while True:
        question = input('What can I help you?\n> ')
        docs_filter = db.similarity_search_with_score(question, k=3)
        try:
            resp = chat_model.invoke([HumanMessage(template.format(**{'question': question, 'context': format_docs(docs_filter)}))])
            print(resp.content)
            print('\n')
        except Exception as e:
            print('[Internal Error] Please try again\n\n')
