import pandas as pd, numpy as np
from graphviz import Digraph

from infogain import ID3
from gini import gini_tree

# Convert tree dict to dot (graphviz) format
def tree_to_dot(tree, parent_name='', graph=None, node_id=0):
    if graph is None:
        graph = Digraph(
            graph_attr={
                'splines': 'false',
                'nodesep': '1.0',
                'ranksep': '1.0',
            },
            node_attr={'fontname': 'Helvetica'},
            edge_attr={'fontname': 'Helvetica'}
        )
        graph.node( name=str(node_id), label=str(list(tree.keys())[0]))
        root_id = node_id
        node_id += 1
    else:
        root_id = parent_name

    for value, subtree in tree[list(tree.keys())[0]].items():
        if isinstance(subtree, dict):
            graph.node(name=str(node_id), label=str(list(subtree.keys())[0]))
            if "{" in str(value):
                value = value.split("{")[1].split("}")[0]
            graph.edge(str(root_id), str(node_id), label=str(value))
            graph, node_id = tree_to_dot(subtree, parent_name=node_id, graph=graph, node_id=node_id+1)
        else:
            color = 'palegreen' if str(subtree) == 'Yes' else 'lightcoral'
            graph.node(name=str(node_id), label=str(subtree), shape='box',
                       fillcolor=color, style='filled')
            if "{" in str(value):
                value = value.split("{")[1].split("}")[0]
            graph.edge(str(root_id), str(node_id), xlabel=str(value))
            node_id += 1

    return graph, node_id

# Function to extract classification rules
def extract_rules(tree, parent_name='', current_rule=''):
    rules = []
    root = list(tree.keys())[0]
    
    for value, subtree in tree[root].items():
        if isinstance(subtree, dict):
            rules += extract_rules(subtree, root, f"{current_rule} AND {root}={value}" if current_rule else f"{root}={value}")
        else:
            rule = f"{current_rule} AND {root}='{value}'" if current_rule else f"{root}='{value}'"
            rule = f"IF {rule} THEN Buys_Computer='{subtree}'"
            rules.append(rule)
    
    return rules


df = pd.read_csv('buys_computer.csv')

def boxprint(text):
    print("┌" + "─" * (len(text) + 2) + "┐")
    print("│ " + text + " │")
    print("└" + "─" * (len(text) + 2) + "┘")

def run_id3():
    # Run ID3 algorithm using Information Gain
    boxprint("Decision tree using Information Gain (ID3)")
    entropy_tree = ID3(df, df, df.columns[:-1])
    print(entropy_tree)

    graph, _ = tree_to_dot(entropy_tree)
    graph.render('id3_tree', format='svg', cleanup=True)
    boxprint("ID3 tree saved as 'id3_tree.svg'.")

    rules = extract_rules(entropy_tree)
    boxprint("Classification rules using Information Gain:")
    for rule in rules:
        print("  " + rule)


def run_gini():
    # Run Gini-based decision tree algorithm
    boxprint("Decision tree using Gini Index (binary splits)")
    gt = gini_tree(df, df, df.columns[:-1])
    print(gt)

    graph, _ = tree_to_dot(gt)
    graph.render('gini_tree', format='svg', cleanup=True)
    boxprint("Gini tree saved as 'gini_tree.svg'.")

    rules = extract_rules(gt)
    boxprint("Classification rules using Gini Index:")
    for rule in rules:
        print("  " + rule)

if __name__ == "__main__":
    run_id3()
    run_gini()