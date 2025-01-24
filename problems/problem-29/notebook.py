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
        stop=100, 
        step=1, 
        value=42,
        label="Random Seed"
    )
    return (seed_slider,)


@app.cell
def _(mo, np, seed_slider):
    # demonstrzate shuffling
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

        indices = np.random.permutation(len(X))

        return X[indices], y[indices]

    # Example Dataset
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([1, 2, 3, 4])

    X_shuffled, y_shuffled = shuffle_dataset(X, y, seed=seed_slider.value)

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
        original_table,
        shuffle_dataset,
        shuffled_table,
        y,
        y_shuffled,
    )


@app.cell
def _(X, X_shuffled, mo, plt, seed_slider, y, y_shuffled):
    # Visualization of the shuffling
    def create_scatter_plot(X, y, title):
        plt.figure(figsize=(6, 4))
        plt.clf()  # Clear the current figure
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.7)
        plt.title(title)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.colorbar(label="Label")

        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()

        return buf

    # Dynamically generate plots based on current seed value (from slider)
    plots_layout = mo.ui.tabs({
        "Original Dataset": mo.image(create_scatter_plot(X, y, f"Original Dataset")),
        "Shuffled Dataset": mo.image(create_scatter_plot(X_shuffled, y_shuffled, f"Shuffled Dataset (Seed: {seed_slider.value})"))
    })
    return create_scatter_plot, plots_layout


@app.cell
def _(mo, original_table, plots_layout, shuffled_table):
    # Combine tables and plots
    mo.vstack([
        mo.md("## Dataset Shuffling Demonstration"),
        mo.hstack([original_table, shuffled_table]),
        plots_layout
    ])
    return


@app.cell
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
