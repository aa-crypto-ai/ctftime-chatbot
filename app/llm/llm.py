import langchain
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.document_loaders import DirectoryLoader

# for RAG evaluation purpose
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference

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

# should not do this loading once server is up, should use some caching thing
embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.load_local("faiss_data/ctftime", embed_model, allow_dangerous_deserialization=True)

def format_docs(docs):
    # TextLoader auto converts single linebreak to double linebreaks, remove it to avoid wrong context intepretation
    return "\n\n".join(doc[0].page_content.replace('\n\n', '\n') for doc in docs)

def get_response(prompt: str, family_name: str, model_name: str):
    global db, template

    # potentially use caching
    chat_model = ChatOpenRouter(model_name=f"{family_name:s}/{model_name:s}")

    evaluator_llm = LangchainLLMWrapper(chat_model)
    context_precision = LLMContextPrecisionWithoutReference(llm=evaluator_llm)

    # the whole docs retrieval algo should be at another place
    # k in config
    docs_filter = db.similarity_search_with_score(prompt, k=3)
    docs_content = [doc.page_content for doc, sim_score in docs_filter]

    # max 10 tries, OpenRouter API could be unstable
    # this is getting the inference and evaluation codes together, need to separate it when understanding more about the structure
    for _ in range(20):
        try:
            response = chat_model.invoke([HumanMessage(template.format(**{'question': prompt, 'context': format_docs(docs_filter)}))])
            sample = SingleTurnSample(
                user_input=prompt,
                response=response.content,
                retrieved_contexts=docs_content,
            )

            # note it could return 0.9999999999 == 1 / (1+1e-10) when theoretically it should be 1
            # because of this line adding 1e-10 to the denominator for preventing 0 denom when there are no valid verdicts
            # https://github.com/xi-zhou/ragas/blob/ae48b4e837dd57964356a4c977724c15f93936c4/src/ragas/metrics/_context_precision.py#L132
            context_precision = context_precision.single_turn_score(sample)   # there is an async version
            return {
                'docs': docs_content,
                'response': response.content,
                'context_precision': context_precision,
            }
        except Exception as e:
            # do logging
            continue

    return None
