import pandas as pd, numpy as np
from itertools import combinations

def generate_indent_tree_blocks(depth, right):
    return "".join(["│  " if not int(right[i]) else "   " for i in range(depth)])

# Gini Index function
def gini_index(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    total_elements = np.sum(counts)
    gini_components = [(counts[i] / total_elements) ** 2 for i in range(len(elements))]
    gini = 1 - np.sum(gini_components)
    return gini, elements, counts, gini_components

# Helper function to print Gini calculation details
def print_gini_details(elements, counts, gini_components):
    total_elements = np.sum(counts)
    details = []
    for i in range(len(elements)):
        details.append(f"({counts[i]}/{total_elements})^2")
    joined_details = " + ".join(details)
    gini_values = " + ".join([f"{1 - gini_component:0.4f}" for gini_component in gini_components])
    total_gini = 1 - np.sum(gini_components)
    return f"{joined_details} = {gini_values} = {total_gini:0.4f}"

# Gini binary split function
def gini_split(data, feature, target_attribute="Buys_Computer", depth=0, right=""):
    total_gini, _, _, _ = gini_index(data[target_attribute])
    vals = np.unique(data[feature])
    
    min_gini = float('inf')
    best_split = None
    split_details = []

    indent = generate_indent_tree_blocks(depth, right)
    if data[feature].dtype.kind in 'iufc':  # For continuous attributes
        print("--> WARNING: CONTINUOUS SPLITS HAVE NOT BEEN TESTED <---")
        vals = sorted(vals)
        for i in range(1, len(vals)):
            split_point = (vals[i-1] + vals[i]) / 2
            D1 = data[data[feature] <= split_point]
            D2 = data[data[feature] > split_point]
            gini_D1 = gini_index(D1[target_attribute])[0]
            gini_D2 = gini_index(D2[target_attribute])[0]
            weighted_gini = (len(D1) / len(data)) * gini_D1 + (len(D2) / len(data)) * gini_D2
            split_details.append((f"{feature} <= {split_point}", weighted_gini))
            print(f"{indent}     ├─ Considering: {feature} <= {split_point}:\t {weighted_gini:0.4f}")
            if weighted_gini < min_gini:
                min_gini = weighted_gini
                best_split = (f"{feature} <= {split_point}", D1, D2)
    else:  # For categorical attributes
        seen_splits = set()
        for i in range(1, len(vals)):
            subsets = list(combinations(vals, i))
            for subset in subsets:
                subset1 = frozenset(subset)
                subset2 = frozenset(vals) - subset1
                if (subset1, subset2) not in seen_splits and (subset2, subset1) not in seen_splits:
                    seen_splits.add((subset1, subset2))
                    D1 = data[data[feature].isin(subset1)]
                    D2 = data[data[feature].isin(subset2)]
                    gini_D1 = gini_index(D1[target_attribute])[0]
                    gini_D2 = gini_index(D2[target_attribute])[0]
                    weighted_gini = (len(D1) / len(data)) * gini_D1 + (len(D2) / len(data)) * gini_D2
                    split_details.append((f"{feature} in {set(subset1)} vs. {set(subset2)}", weighted_gini))
                    lhs_string = f"{indent}│  ├──── {feature} in {set(subset1)} vs. {set(subset2)}"
                    rhs_string = f": {total_gini:0.4f} - {total_gini-weighted_gini:0.4f} = {weighted_gini:0.4f}"
                    print(f"{lhs_string:<55}{rhs_string}")
                    if weighted_gini < min_gini:
                        min_gini = weighted_gini
                        best_split = (f"{feature} in {set(subset1)}", f"{feature} in {set(subset2)}", D1, D2)

    reduction_in_impurity = total_gini - min_gini
    
    return reduction_in_impurity, min_gini, best_split, split_details

# Helper function to print split details
def print_split_details(split_details, feature, depth=0, right=""):
    indent = generate_indent_tree_blocks(depth, right)
    count_splits = len(split_details)
    for i, (split, gini) in enumerate(split_details):
        if i == count_splits - 1:
            block = "└───"
        else:
            block = "├───"
        split_str = split.replace("frozenset", "")
        if i == 0 and count_splits > 1:
            print(f"{indent}│  {block} {split_str}: {gini:0.4f} <- best split")
        else:
            print(f"{indent}│  {block} {split_str}: {gini:0.4f}")

def find_best_split(data, features, target_attribute="Buys_Computer", depth=0, right=""):
    indent = generate_indent_tree_blocks(depth, right)
    total_gini, elements, counts, gini_components = gini_index(data[target_attribute])
    
    best_feature = None
    best_split = None
    max_reduction = 0
    best_split_details = []

    print(f"{indent}├──┬─ Calculating reduction in impurity for features with binary splits:")
    for i, feature in enumerate(features):
        reduction, min_gini, split, split_details = gini_split(data, feature, target_attribute, depth, right)
        if reduction > max_reduction:
            max_reduction = reduction
            best_feature = feature
            best_split = split
            best_split_details = split_details
        # print(f"{indent}│  ├──── {feature:<15}: {total_gini:0.4f} - {min_gini:0.4f} = {reduction:0.4f}")

    if best_feature:
        # find the best (split, gini) in `best_split_details` with a oneliner
        very_best_split = sorted(best_split_details, key=lambda x: x[1])[0]
        print(f"{indent}│  │ ")
        print(f"{indent}│  └──── Best feature to split on: {very_best_split[0]}: {very_best_split[1]:0.4f}")
    
    return best_feature, best_split

def gini_tree(data, original_data, features, target_attribute="Buys_Computer", parent_node_class=None):
    gt = gini_tree_(data, original_data, features, target_attribute, parent_node_class)
    print("\nFinished Gini tree algorithm")
    return gt

# Define Gini-based decision tree algorithm
def gini_tree_(data, original_data, features, target_attribute="Buys_Computer", parent_node_class=None, depth=0, right=""):
    indent = generate_indent_tree_blocks(depth, right)
    
    # If all target values are the same, return the single class
    if len(np.unique(data[target_attribute])) <= 1:
        print(f"{indent}└─── Done! Only one value for '{target_attribute}' remains: '{np.unique(data[target_attribute])[0]}'")
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
        parent_gini, elements, counts, gini_components = gini_index(data[target_attribute])
        block = "├" if depth > 0 else "┌"
        root_or_parent = "Root" if depth == 0 else "Parent"
        calc = print_gini_details(elements, counts, gini_components)
        print(f"{indent}{block}──── {root_or_parent} class Gini: {calc}")
        print(f"{indent}│")
        
        best_feature, best_split = find_best_split(data, features, target_attribute, depth, right)
        
        if not best_feature:
            print(f"{indent}└─── No valid feature found, returning parent node class: {parent_node_class}")
            return parent_node_class
        
        tree = {best_feature: {}}
        
        best_split_str, best_split_complement_str, D1, D2 = best_split
        best_split_str = best_split_str.replace("frozenset", "")
        best_split_complement_str = best_split_complement_str.replace("frozenset", "")
        
        print(f"{indent}│")
        if len(features) > 0 and len(np.unique(D1[target_attribute])) > 1:
            print(f"{indent}├──┬─ Left branch: {best_split_str}")
            subtree_left = gini_tree_(D1, original_data, features, target_attribute, parent_node_class, depth + 1, right + "0")
        else:
            print(f"{indent}├──── Left branch: {best_split_str} only has one value for '{target_attribute}': '{np.unique(D1[target_attribute])[0]}'")
            subtree_left = np.unique(D1[target_attribute])[0]
        tree[best_feature][best_split_str] = subtree_left
        
        print(f"{indent}│")
        if len(features) > 0 and len(np.unique(D2[target_attribute])) > 1:
            print(f"{indent}└──┬─ Right branch: {best_split_complement_str}")
            subtree_right = gini_tree_(D2, original_data, features, target_attribute, parent_node_class, depth + 1, right + "1")
        else:
            print(f"{indent}└──── Right branch: {best_split_complement_str} only has one value for '{target_attribute}': '{np.unique(D2[target_attribute])[0]}'")
            subtree_right = np.unique(D2[target_attribute])[0]
        tree[best_feature][best_split_complement_str] = subtree_right
        
        return tree
