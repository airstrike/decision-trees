# decision-trees 🌳

A flexible implementation of ID3 and Gini decision tree algorithms with visualization capabilities.

## Overview

`decision-trees` provides an intuitive interface for building and visualizing decision trees using both information gain (ID3) and Gini impurity splitting criteria. It generates comprehensive decision rules and creates visual representations of the resulting trees.

Key features:
- Support for both ID3 and Gini-based decision trees
- Automatic rule extraction from trained trees
- SVG visualization generation
- Dynamic target variable selection
- Clean command-line interface
- Flexible input data handling

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/airstrike/decision-trees.git
cd decision-trees
pip install -r requirements.txt
```

## Quick Start

The script accepts two required arguments:
- Target variable to predict
- Input CSV file path

```bash
python trees.py Buys_Computer data.csv
```

This will:
1. Generate both ID3 and Gini-based decision trees
2. Create SVG visualizations of the trees
3. Extract and display classification rules
4. Save visualizations as 'id3_tree.svg' and 'gini_tree.svg'

## Example Dataset

The repository includes `buys_computer.csv` as a sample dataset. This dataset contains customer attributes and their computer purchasing decisions, making it perfect for demonstrating binary classification trees.

Example usage with the provided dataset:

```bash
python trees.py Buys_Computer buys_computer.csv
```

## Project Structure

```
decision-trees/
├── trees.py        # Main implementation
├── infogain.py     # ID3 algorithm implementation
├── gini.py         # Gini index implementation
└── buys_computer.csv  # Sample dataset
```

## Development

Running the code:
```bash
python trees.py <target_variable> <input_csv>
```

## Dependencies

Built with:
- pandas - Data manipulation
- numpy - Numerical operations
- graphviz - Tree visualization
- argparse - Command-line interface

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.