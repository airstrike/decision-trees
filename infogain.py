import pandas as pd, numpy as np
from graphviz import Digraph

# Entropy function
def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    total_elements = np.sum(counts)
    entropy_components = [(-counts[i]/total_elements) * np.log2(counts[i]/total_elements) for i in range(len(elements))]
    total_entropy = np.sum(entropy_components)
    return total_entropy, elements, counts, entropy_components

# Helper function to print entropy calculation details
def print_entropy_details(elements, counts, entropy_components, depth=0):
    tree_block = "│" if depth > 0 else " "
    indent = f"{tree_block}  " * depth
    total_elements = np.sum(counts)
    details = []
    for i in range(len(elements)):
        proportion = counts[i] / total_elements
        entropy_component = entropy_components[i]
        details.append(f"-{counts[i]}/{total_elements} log2({counts[i]}/{total_elements})")
    joined_details = " + ".join(details)
    entropy_values = " + ".join([f"{component:.3f}" for component in entropy_components])
    total_entropy = np.sum(entropy_components)
    print(f"{indent}│  └─── {joined_details} = {entropy_values} = {total_entropy:.3f}")

# Information gain function
def info_gain(data, split_attribute, target_attribute="Buys_Computer"):
    total_entropy, _, _, _ = entropy(data[target_attribute])
    vals, counts = np.unique(data[split_attribute], return_counts=True)
    
    weighted_entropy = np.sum([(counts[i]/np.sum(counts)) * entropy(data[data[split_attribute] == vals[i]][target_attribute])[0] for i in range(len(vals))])
    information_gain = total_entropy - weighted_entropy
    return information_gain, total_entropy, weighted_entropy

# ID3 algorithm
def ID3(data, original_data, features, target_attribute="Buys_Computer", parent_node_class=None, depth=0):
    t = ID3_(data, original_data, features, target_attribute, parent_node_class, depth)
    print("│\n└──── Finished ID3 algorithm")
    return t

def ID3_(data, original_data, features, target_attribute="Buys_Computer", parent_node_class=None, depth=0):
    # Print the current depth of the tree
    tree_block = "│" if depth > 0 else " "
    indent = f"{tree_block}  " * depth
    
    # If all target values are the same, return the single class
    if len(np.unique(data[target_attribute])) <= 1:
        print(f"{indent}└─── Done! Only one '{target_attribute}' remains: '{np.unique(data[target_attribute])[0]}'")
        return np.unique(data[target_attribute])[0]
    
    # If the dataset is empty, return the mode target feature value from the original dataset
    elif len(data) == 0:
        mode_value = np.unique(original_data[target_attribute])[np.argmax(np.unique(original_data[target_attribute], return_counts=True)[1])]
        print(f"{indent}└─── Dataset is empty, returning mode value: {mode_value}")
        return mode_value
    
    # If the feature space is empty, return the parent node class
    elif len(features) == 0:
        print(f"{indent}└─── Feature space is empty, returning parent node class: {parent_node_class}")
        return parent_node_class
    
    # If none of the above conditions are met, grow the tree
    else:
        parent_node_class = np.unique(data[target_attribute])[np.argmax(np.unique(data[target_attribute], return_counts=True)[1])]
        parent_entropy, elements, counts, entropy_components = entropy(data[target_attribute])
        block = "├" if depth > 0 else "┌"
        print(f"{indent}{block}──┬─ Parent class entropy: {parent_entropy:.3f}")
        print_entropy_details(elements, counts, entropy_components, depth)
        
        item_values = [info_gain(data, feature, target_attribute) for feature in features]
        info_gains = [item[0] for item in item_values]
        info_gain_dict = dict(zip(features, info_gains))
        best_feature_index = np.argmax(info_gains)
        best_feature = features[best_feature_index]
        
        count_features = len(info_gain_dict.keys())
        if count_features == 1:
            feature, gain = list(info_gain_dict.items())[0]
            tree = {feature: {}}
        else:
            print(f"{indent}├──┬─ Calculating Information Gain for:")
            for e, (feature, gain) in enumerate(info_gain_dict.items()):
                if e == count_features - 1:
                    block = "└────"
                else:
                    block = "├────"
                if feature == best_feature:
                    print(f"{indent}│  {block} {feature:<15}: {parent_entropy:.3f}-{parent_entropy-gain:.3f}= {gain:.3f} <- best feature")
                else:
                    print(f"{indent}│  {block} {feature:<15}: {parent_entropy:.3f}-{parent_entropy-gain:.3f}= {gain:.3f}")
            
            tree = {best_feature: {}}
        
        features = [i for i in features if i != best_feature]
        
        count_trees = len(np.unique(data[best_feature]))
        for i, value in enumerate(np.unique(data[best_feature])):
            if i == count_trees - 1:
                block = "└"
                done_block = " "
            else:
                block = "├"
                done_block = "│"
            sub_data = data.where(data[best_feature] == value).dropna()
            if len(np.unique(sub_data[target_attribute])) > 1:
                print(f"{indent}│")
                print(f"{indent}├──┬─ Creating subtree for {best_feature}='{value}'")
                subtree = ID3_(sub_data, original_data, features, target_attribute, parent_node_class, depth + 1)
                tree[best_feature][value] = subtree
            else:
                print(f"{indent}{block}──┬─ Creating subtree for {best_feature}='{value}'")
                print(f"{indent}{done_block}  └──── Done! Only one possible value remains for '{target_attribute}': '{np.unique(sub_data[target_attribute])[0]}'")
                subtree = np.unique(sub_data[target_attribute])[0]
                tree[best_feature][value] = subtree
        
        return tree
