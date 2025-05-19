import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def calculate_probability_table_sizes(vocab_size, sequence_length):
    """
    Calculate the size of probability tables for autoregressive language models
    with different vocabulary sizes and sequence lengths.

    Args:
        vocab_size: Number of unique tokens in vocabulary
        sequence_length: Maximum length of sequences

    Returns:
        Dictionary containing table sizes for each position n
    """
    table_sizes = {}
    total_entries = 0

    # Calculate table size for each position n in the sequence
    for n in range(1, sequence_length + 1):
        # For position n, we need V^n entries (V^(n-1) contexts * V possible next tokens)
        entries = vocab_size ** n
        table_sizes[f'p(x_{n}|x_1,...,x_{{{n - 1}}}))'] = entries
        total_entries += entries

    table_sizes['Total'] = total_entries
    table_sizes['Formula'] = vocab_size * (1 - vocab_size ** sequence_length) / (1 - vocab_size)

    return table_sizes


def visualize_growth(vocab_sizes, max_sequence_length):
    """
    Visualize the exponential growth of probability table sizes

    Args:
        vocab_sizes: List of vocabulary sizes to compare
        max_sequence_length: Maximum sequence length to consider
    """
    plt.figure(figsize=(12, 8))

    for v in vocab_sizes:
        sequence_lengths = range(1, max_sequence_length + 1)
        table_sizes = [calculate_probability_table_sizes(v, n)['Total'] for n in sequence_lengths]

        plt.semilogy(sequence_lengths, table_sizes, marker='o', label=f'V = {v}')

    plt.grid(True, which="both", ls="--")
    plt.xlabel('Sequence Length (N)')
    plt.ylabel('Number of Table Entries (log scale)')
    plt.title('Exponential Growth of Probability Tables in Autoregressive Language Models')
    plt.legend()

    return plt


# Example usage with a simple demonstration
def main():
    print("DEMONSTRATING EXPONENTIAL GROWTH IN AUTOREGRESSIVE LANGUAGE MODELS")
    print("=" * 70)

    # Parameters for demonstration
    vocab_size = 5  # Small vocabulary for demonstration (e.g., A, B, C, D, E)
    sequence_length = 8  # Maximum sequence length to consider

    # Calculate table sizes
    table_sizes = calculate_probability_table_sizes(vocab_size, sequence_length)

    # Display results in a table
    headers = ["Conditional Probability", "Number of Entries"]
    rows = [(k, f"{v:,}") for k, v in table_sizes.items() if k != 'Formula']

    print(f"For vocabulary size V = {vocab_size}:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    print()
    print(f"Note: Formula verification: {table_sizes['Total']:,} = {table_sizes['Formula']:,.0f}")
    print("The formula is: V * (1 - V^N) / (1 - V)")

    # Now compare different vocabulary sizes
    vocab_sizes = [2, 5, 10, 26]  # Binary, small, medium, alphabet
    plt = visualize_growth(vocab_sizes, sequence_length)

    print("\nGenerating plot to visualize the exponential growth...")
    plt.savefig('autoregressive_growth.png')
    print("Plot saved as 'autoregressive_growth.png'")

    # Show a practical example with a real language model
    print("\nPRACTICAL EXAMPLE")
    print("For GPT models with vocabulary size of ~50,000 tokens:")
    real_vocab = 50000

    practical_examples = [
        ("For a 2-gram model (n=2)", real_vocab ** 2),
        ("For a 3-gram model (n=3)", real_vocab ** 3),
        ("For a context of 10 tokens", real_vocab ** 10),
        ("For a context of 100 tokens", float('inf'))  # This would be a huge number
    ]

    for desc, entries in practical_examples:
        if entries == float('inf'):
            print(f"{desc}: Too large to compute (more than atoms in the universe)")
        else:
            print(f"{desc}: {entries:,.0f} entries needed")

    print("\nThis is why neural networks are used to parameterize these distributions")
    print("instead of explicit probability tables.")


if __name__ == "__main__":
    main()


"""

DEMONSTRATING EXPONENTIAL GROWTH IN AUTOREGRESSIVE LANGUAGE MODELS
======================================================================
For vocabulary size V = 5:
+---------------------------+---------------------+
| Conditional Probability   | Number of Entries   |
+===========================+=====================+
| p(x_1|x_1,...,x_{0}))     | 5                   |
+---------------------------+---------------------+
| p(x_2|x_1,...,x_{1}))     | 25                  |
+---------------------------+---------------------+
| p(x_3|x_1,...,x_{2}))     | 125                 |
+---------------------------+---------------------+
| p(x_4|x_1,...,x_{3}))     | 625                 |
+---------------------------+---------------------+
| p(x_5|x_1,...,x_{4}))     | 3,125               |
+---------------------------+---------------------+
| p(x_6|x_1,...,x_{5}))     | 15,625              |
+---------------------------+---------------------+
| p(x_7|x_1,...,x_{6}))     | 78,125              |
+---------------------------+---------------------+
| p(x_8|x_1,...,x_{7}))     | 390,625             |
+---------------------------+---------------------+
| Total                     | 488,280             |
+---------------------------+---------------------+

Note: Formula verification: 488,280 = 488,280
The formula is: V * (1 - V^N) / (1 - V)

Generating plot to visualize the exponential growth...
Plot saved as 'autoregressive_growth.png'

PRACTICAL EXAMPLE
For GPT models with vocabulary size of ~50,000 tokens:
For a 2-gram model (n=2): 2,500,000,000 entries needed
For a 3-gram model (n=3): 125,000,000,000,000 entries needed
For a context of 10 tokens: 97,656,250,000,000,007,450,947,110,441,295,901,655,239,229,440 entries needed
For a context of 100 tokens: Too large to compute (more than atoms in the universe)

This is why neural networks are used to parameterize these distributions
instead of explicit probability tables.

Process finished with exit code 0
"""