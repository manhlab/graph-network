import networkx as nx
import pandas as pd
from itertools import chain

### This function try to read and write graph to csv format, to better read and delivery graph data

def load_graph(node_src, edge_src, node_attr=None, edge_attr=None):
  edge_info = pd.read_csv(edge_src)
  node_info = pd.read_csv(node_src)
  G = nx.from_pandas_edgelist(edge_info, edge_attr=edge_attr, create_using = nx.Graph())
  
  for node in node_info.iterrows():
    adict = dict(node[1])
    if node_attr:
        fea = {k:adict[k] for k in node_attr if k in adict}
        G.add_node(node[1]['node'], **fea, size=1)
    else:
        G.add_node(node[1]['node'], size=1)
  return G


def save_graph(G, node_src, edge_src):
 
  edge_dict  = {}
  node_dict = {}
  edge_dict['source'] = []
  edge_dict['target'] = []
  node_dict['node'] = []
  edge_fea = set(chain.from_iterable(d.keys() for *_, d in G.edges(data=True)))
  node_fea = set(chain.from_iterable(d.keys() for *_, d in G.nodes(data=True)))

  for fea in edge_fea:
    edge_dict[fea] = []

  for fea in node_fea:
    node_dict[fea] = []

  for edge in list(G.edges(data=True)):
    edge_dict['source'].append(edge[0])
    edge_dict['target'].append(edge[1])
    for k,v in edge[2].items():
      edge_dict[k].append(v)
  
  for node in list(G.nodes(data=True)):
    node_dict['node'].append(node[0])
    for k,v in node[1].items():
      node_dict[k] = v
  edge_csv = pd.DataFrame.from_dict(edge_dict)
  node_csv = pd.DataFrame.from_dict(node_dict)
  edge_csv.to_csv(edge_src, index=False)
  node_csv.to_csv(node_src, index=False)