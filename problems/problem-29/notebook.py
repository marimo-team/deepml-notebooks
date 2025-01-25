# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.2.1",
#     "matplotlib==3.10.0",
# ]
# ///

import marimo

__generated_with = "0.10.16"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    # Title and Introduction with LaTeX
    mo.md(
        r"""
        # Random Dataset Shuffling in Machine Learning

        ## Understanding Data Randomization

        In Machine Learning, **dataset shuffling** is a critical preprocessing technique that ensures unbiased model training. This interactive notebook explores the working of randomly shuffling datasets while maintaining the crucial correspondence between features and labels.
        """
    ).center()
    return


@app.cell(hide_code=True)
def _(mo):
    # Mathematical Explanation with Accordion
    math_explanation = mo.md(r"""
    ### The Shuffling Process

    Key mathematical principles of dataset shuffling:

    1. **Index Randomization**: 

           - Generate an array of indices: $\text{indices} = [0, 1, 2, \ldots, n-1]$
           - Apply a random permutation: $\text{shuffled\_indices} = \text{shuffle}(\text{indices})$

    2. **Data Reordering**:
       For features $X$ and labels $y$:
       $X_{\text{shuffled}} = X[\text{shuffled\_indices}]$
       $y_{\text{shuffled}} = y[\text{shuffled\_indices}]$

    This ensures:

    - Random distribution of data

    - Preservation of feature-label pairs

    - Potential reduction of learning biases
    """)

    # Use an Accordion to make the mathematical explanation interactive
    mo.accordion({"🔢 Mathematical Foundations": math_explanation})
    return (math_explanation,)


@app.cell
def _(seed_slider):
    seed_slider
    return


@app.cell(hide_code=True)
def _(mo):
    # Interactive Seed Selection
    seed_slider = mo.ui.slider(
        start=0, 
        stop=1000,  # Increased range for more variability 
        step=1, 
        value=42,
        label="Random Seed"
    )
    return (seed_slider,)


@app.cell
def _(mo, np, seed_slider):
    # demonstrate shuffling
    def shuffle_dataset(X, y, seed=None):
        """
        Shuffle two numpy arrays while maintaining their correspondence.

        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Label vector
            seed (int, optional): Random seed for reproducibility

        Returns:
            tuple: Shuffled (X, y)
        """
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)

        # shuffling indices
        indices = np.random.permutation(len(X))

        # Apply shuffling
        return X[indices], y[indices]

    # Create a dataset
    def generate_dataset(seed=None):
        if seed is not None:
            np.random.seed(seed)

        # Create a more complex dataset
        n_samples = 50
        X = np.random.randn(n_samples, 2)

        # Create labels with some structure
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        return X, y

    # initial dataset
    X, y = generate_dataset()
    X_shuffled, y_shuffled = shuffle_dataset(X, y, seed=seed_slider.value)

    # Display original and shuffled datasets
    original_table = mo.ui.table(
        data={
            "Original X": list(map(list, X)),
            "Original y": list(y)
        },
        label="Original Dataset"
    )

    shuffled_table = mo.ui.table(
        data={
            "Shuffled X": list(map(list, X_shuffled)),
            "Shuffled y": list(y_shuffled)
        },
        label="Shuffled Dataset"
    )
    return (
        X,
        X_shuffled,
        generate_dataset,
        original_table,
        shuffle_dataset,
        shuffled_table,
        y,
        y_shuffled,
    )


@app.cell
def _(generate_dataset, mo, plt, seed_slider):
    # Visualization of shuffling
    def create_scatter_plot(X, y, title):
        plt.figure(figsize=(8, 6))
        plt.clf()  # Clear previous plot
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(scatter, label="Label")
        plt.grid(True, linestyle='--', alpha=0.7)

        # Save the plot to a BytesIO object
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300)
        buf.seek(0)
        plt.close()

        return buf

    # generate plots
    def generate_plots(seed):
        # Regenerate dataset with new seed
        X_orig, y_orig = generate_dataset()
        X_shuf, y_shuf = generate_dataset(seed=seed)

        return mo.ui.tabs({
            "Original Dataset": mo.image(create_scatter_plot(X_orig, y_orig, "Original Dataset")),
            f"Shuffled Dataset (Seed: {seed})": mo.image(create_scatter_plot(X_shuf, y_shuf, f"Shuffled Dataset (Seed: {seed})"))
        })

    # Initial plots
    plots_layout = generate_plots(seed_slider.value)
    return create_scatter_plot, generate_plots, plots_layout


@app.cell(hide_code=True)
def _(mo, original_table, plots_layout, shuffled_table):
    # Combine tables and plots
    mo.vstack([
        mo.md("## Dataset Shuffling Demonstration"),
        mo.hstack([original_table, shuffled_table]),
        plots_layout
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    # Key Learning Points
    key_points = mo.md(r"""
    ### 🔑 Key Insights


        - Numpy's `permutation()` creates random indices

        - Shuffling maintains feature-label correspondence

        - Random seed ensures reproducible shuffling
    """)

    # Use a Callout to highlight key points
    mo.callout(key_points, kind="success")
    return (key_points,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


if __name__ == "__main__":
    app.run()
