
# Decision Tree Algorithms

This repository contains implementations of decision tree algorithms using two
different criteria for splitting: Information Gain (ID3) and Gini Index. The
provided scripts allow you to visualize the decision trees and extract
classification rules from them.

## Usage

1. Clone the repository or download the script:

```bash
# Using Git
git clone https://github.com/foo/trees.git
cd trees

# Using curl
curl -LO https://github.com/foo/trees/archive/refs/heads/main.zip
unzip main.zip
cd trees-main
```

2. Ensure you have the required packages installed (see Requirements section).

3. Run the script:

```bash
python trees.py
```

This will execute the script, generate the decision trees, save them as SVG
files, and print the classification rules.

## Files

- `trees.py`: The main script to run the decision tree algorithms and visualize
  the results.
- `gini.py`: Contains the implementation of the Gini Index-based decision tree
  algorithm.
- `infogain.py`: Contains the implementation of the ID3 decision tree algorithm
  using Information Gain.
- `buys_computer.csv`: The dataset used for building the decision trees.

## Requirements

The following Python packages are required to run the scripts:

- pandas
- numpy
- graphviz

You can install the required packages using the following command:

```bash
pip install pandas numpy graphviz
```

Additionally, you need to have Graphviz installed on your system. You can
download it from [Graphviz's official website](https://graphviz.org/download/)
and follow the installation instructions for your operating system.

## Algorithms

### ID3 Algorithm (Information Gain)

The ID3 algorithm builds a decision tree by recursively selecting the attribute
that provides the highest information gain. Information Gain is calculated based
on the entropy of the target attribute. The steps are as follows:

1. Calculate the entropy of the target attribute.
2. For each attribute, calculate the entropy and the information gain.
3. Select the attribute with the highest information gain.
4. Split the dataset based on the selected attribute and repeat the process for
   each subset until all instances belong to the same class or there are no more
   attributes to split on.

The resulting tree and classification rules are saved as `id3_tree.svg` and
printed to the console, respectively.

### Gini Index Algorithm

The Gini Index algorithm builds a decision tree by recursively selecting the
attribute that results in the lowest Gini impurity for binary splits. The steps
are as follows:

1. Calculate the Gini impurity of the target attribute.
2. For each attribute, calculate the Gini impurity for all possible binary splits.
3. Select the attribute and split that results in the lowest Gini impurity.
4. Split the dataset based on the selected attribute and repeat the process for
   each subset until all instances belong to the same class or there are no more
   attributes to split on.

The resulting tree and classification rules are saved as `gini_tree.svg` and
printed to the console, respectively.

## Output

- `id3_tree.svg`: The decision tree generated using the ID3 algorithm
  (Information Gain).
- `gini_tree.svg`: The decision tree generated using the Gini Index algorithm.
- Classification rules are printed to the console for both algorithms.

## Example Dataset

The example dataset `buys_computer.csv` is used to build the decision trees. The
dataset contains the following columns:

- Age
- Income
- Student
- Credit_Rating
- Buys_Computer

Each row represents an instance with attributes and the target class
`Buys_Computer`, indicating whether a computer was purchased (Yes or No).

Feel free to modify the dataset or use your own to see how the decision trees
and classification rules change.

---

Enjoy exploring decision tree algorithms! If you have any questions or
suggestions, feel free to open an issue or submit a pull request.
