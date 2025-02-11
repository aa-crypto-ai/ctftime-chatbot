import langchain
from langchain_core.messages import HumanMessage

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
    return "\n\n".join(doc.page_content.replace('\n\n', '\n') for doc, score in docs)

def get_rag_prompt(prompt: str, docs: list):
    """ Note: for a complete chatbot, it will need to handle a continuous conversation instead of only one-time Q&A
    """
    global template
    return [HumanMessage(template.format(**{'question': prompt, 'context': format_docs(docs)}))]