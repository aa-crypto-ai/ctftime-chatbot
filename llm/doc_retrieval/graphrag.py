from langchain_community.graphs.graph_document import Node, Relationship
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph

#from utils.neo4j import OpenRouterAILLM
from utils.langchain_adapter import ChatOpenRouter
from utils import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

if __name__ == '__main__':
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)
    chat_model = ChatOpenRouter(model_name="meta-llama/llama-3.1-70b-instruct:free")
    chain = GraphCypherQAChain.from_llm(
        chat_model, graph=graph, top_k=100, verbose=True, allow_dangerous_requests=True
    )

    print('How many atheletes did Japan have?')
    print(chain.invoke("How many atheletes did Japan have?")['result'])
