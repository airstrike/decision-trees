import pandas as pd
import numpy as np
from graphviz import Digraph
from infogain import ID3
from gini import gini_tree
import argparse

def get_target_classes(df, target_var):
    """Extract all unique values of the target variable from the dataframe"""
    return sorted(df[target_var].unique())

def tree_to_dot(tree, target_var, df, parent_name='', graph=None, node_id=0):
    # From Plotly's "Set3" color scale which is nice and pastel
    colors = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', 
              '#80b1d3', '#fdb462', '#b3de69', '#fccde5', 
              '#d9d9d9', '#bc80bd', '#ccebc5', '#ffed6f']
                
    # Get unique target classes and create color mapping
    target_classes = get_target_classes(df, target_var)
    
    # Create color mapping based on number of target classes
    if len(target_classes) == 2:
        color_map = {
            str(target_classes[0]): 'lightcoral',
            str(target_classes[1]): 'palegreen'
        }
    else:
        color_map = {
            str(class_val): color 
            for class_val, color in zip(target_classes, colors)
        }
    
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
        graph.node(name=str(node_id), label=str(list(tree.keys())[0]))
        root_id = node_id
        node_id += 1
    else:
        root_id = parent_name
        
    for value, subtree in tree[list(tree.keys())[0]].items():
        if isinstance(subtree, dict):
            # Internal decision node - no color
            graph.node(name=str(node_id), label=str(list(subtree.keys())[0]))
            if "{" in str(value):
                value = value.split("{")[1].split("}")[0]
            graph.edge(str(root_id), str(node_id), label=str(value))
            graph, node_id = tree_to_dot(subtree, target_var, df, parent_name=node_id, graph=graph, node_id=node_id+1)
        else:
            # Leaf node - color based on target variable value
            color = color_map.get(str(subtree))
            graph.node(name=str(node_id), 
                      label=str(subtree), 
                      shape='box',
                      fillcolor=color, 
                      style='filled')
            if "{" in str(value):
                value = value.split("{")[1].split("}")[0]
            graph.edge(str(root_id), str(node_id), xlabel=str(value))
            node_id += 1
            
    return graph, node_id

# Function to extract classification rules
def extract_rules(tree, target_var, parent_name='', current_rule=''):
    rules = []
    root = list(tree.keys())[0]
    
    for value, subtree in tree[root].items():
        if isinstance(subtree, dict):
            rules += extract_rules(subtree, target_var, root, f"{current_rule} AND {root}={value}" if current_rule else f"{root}={value}")
        else:
            rule = f"{current_rule} AND {root}='{value}'" if current_rule else f"{root}='{value}'"
            rule = f"IF {rule} THEN {target_var}='{subtree}'"
            rules.append(rule)
    
    return rules

def boxprint(text):
    print("┌" + "─" * (len(text) + 2) + "┐")
    print("│ " + text + " │")
    print("└" + "─" * (len(text) + 2) + "┘")

def run_id3(df, target_var):
    # Run ID3 algorithm using Information Gain
    boxprint(f"Decision tree using Information Gain (ID3) for {target_var}")
    feature_columns = [col for col in df.columns if col != target_var]
    entropy_tree = ID3(df, df, feature_columns, target_var)
    print(entropy_tree)
    graph, _ = tree_to_dot(entropy_tree, target_var, df)
    graph.render('id3_tree', format='svg', cleanup=True)
    boxprint("ID3 tree saved as 'id3_tree.svg'.")
    rules = extract_rules(entropy_tree, target_var)
    boxprint(f"Classification rules using Information Gain for {target_var}:")
    for rule in rules:
        print("  " + rule)

def run_gini(df, target_var):
    # Run Gini-based decision tree algorithm
    boxprint(f"Decision tree using Gini Index (binary splits) for {target_var}")
    feature_columns = [col for col in df.columns if col != target_var]
    gt = gini_tree(df, df, feature_columns, target_var)
    print(gt)
    graph, _ = tree_to_dot(gt, target_var, df)
    graph.render('gini_tree', format='svg', cleanup=True)
    boxprint("Gini tree saved as 'gini_tree.svg'.")
    rules = extract_rules(gt, target_var)
    boxprint(f"Classification rules using Gini Index for {target_var}:")
    for rule in rules:
        print("  " + rule)

def main():
    parser = argparse.ArgumentParser(description='Generate decision trees with a specified target variable')
    parser.add_argument('target_variable', type=str, help='The target variable to predict')
    parser.add_argument('input', type=str, help='Input CSV file path')
    args = parser.parse_args()
    
    # Read the data
    try:
        df = pd.read_csv(args.input)
    except FileNotFoundError:
        print(f"Error: Could not find input file '{args.input}'")
        return
        
    # Verify target variable exists in dataset
    if args.target_variable not in df.columns:
        print(f"Error: Target variable '{args.target_variable}' not found in dataset")
        print(f"Available columns: {', '.join(df.columns)}")
        return
    
    # Run both algorithms
    run_id3(df, args.target_variable)
    run_gini(df, args.target_variable)

if __name__ == "__main__":
    main()