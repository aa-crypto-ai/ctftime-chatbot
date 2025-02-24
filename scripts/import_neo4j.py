import neo4j
from neo4j_graphrag.llm import OpenAILLM
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import Node, Relationship, GraphDocument
from langchain_neo4j import Neo4jGraph
from yfiles_jupyter_graphs import GraphWidget
import pandas as pd

from utils import NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD

def to_graph_doc(row):
    global medal_nodes
    event_type = row['event_type']
    event = row['discipline']
    category = row['event']
    medal_idx = row['medal_code']
    country = row['country']
    gender = row['gender']
    medal = {1: 'Gold', 2: 'Silver', 3: 'Bronze'}[medal_idx].upper()
    indiv_exceptional_case = (event_type == 'COUP' and category != 'Duet') or event in ['Beach Volleyball', 'Sailing'] or category == 'Marathon Race Walk Relay Mixed'
    if event_type in ['ATH', 'HATH', 'HCOUP'] or indiv_exceptional_case:
        node_person = Node(id=row['name'], type='Person')
        edge_event = Relationship(source=node_person, target=category_nodes[category], type='RECEIVED_MEDAL')
        edge_country = Relationship(source=node_person, target=country_nodes[country], type='REPRESENTS_COUNTRY')
        name = row['name']

        nodes = [node_person]
        edges = [edge_event, edge_country]
        if gender not in ['O', 'X']:
            edge_gender = Relationship(source=node_person, target=gender_nodes[gender], type='IS_GENDER')
            nodes.append(gender_nodes[gender])

            edges.append(edge_gender)
    else:
        country = row['name']
        # just a quick way to check if there is a single digit number to the national team, won't be 2-digits
        if country.split(' ')[-1] in '0123456789':
            country = ' '.join(country.split(' ')[:-1])
        node_country = country_nodes[country]
        edge_event = Relationship(source=node_country, target=category_nodes[category], type='RECEIVED_MEDAL')
        nodes = []
        edges = [edge_event]
        name = country

    return GraphDocument(nodes=nodes, relationships=edges, source=Document(page_content=f'{name} got {medal} medal'))

if __name__ == '__main__':

    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

    df = pd.read_csv('data/raw_data/olympics_2024/medals.csv')

    # 1, 2, 3 are the medal indices from csv
    # didn't know how to utilize this yet on this graph structure
    # medal_nodes = {
    #     1: Node(id='Gold', type='Medal'),
    #     2: Node(id='Silver', type='Medal'),
    #     3: Node(id='Bronze', type='Medal'),
    # }
    gender_nodes = {
        'M': Node(id='Male', type='Gender'),
        'W': Node(id='Female', type='Gender'),
    }
    country_nodes = {country: Node(id=country, type='Country') for country in df['country'].unique()}
    event_nodes = {event: Node(id=event, type='Event') for event in df['discipline'].unique()}
    category_nodes = {category: Node(id=category, type='Category') for category in df['event'].unique()}

    category_edges = []
    for event, category in df.groupby(['discipline', 'event'])[['discipline', 'event']].groups:
        category_edge = Relationship(source=category_nodes[category], target=event_nodes[event], type='BELONGS_TO')
        category_edges.append(category_edge)

    graph_docs = []
    for idx, row in df.iterrows():
        graph_doc = to_graph_doc(row)
        graph_docs.append(graph_doc)

    print('Adding documents to graph DB...')
    graph.add_graph_documents(graph_docs)
    print('added %d graph documents' % len(graph_docs))

    # visualize graph
    # !!! only works for jupyter environment
    # https://github.com/yWorks/yfiles-jupyter-graphs
    default_cypher = "MATCH (s)-[r]->(t) RETURN s,r,t LIMIT 100"
    def showGraph(cypher: str = default_cypher):
        neo4j_driver = neo4j.GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
        )
        
        session = neo4j_driver.session()
        widget = GraphWidget(graph = session.run(cypher).graph())
        widget.node_label_mapping = 'id'
        return widget

    # showGraph()