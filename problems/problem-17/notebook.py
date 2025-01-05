# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "drawdata==0.3.6",
#     "marimo",
#     "numpy==2.2.1",
#     "pandas==2.2.3",
#     "plotly==5.24.1",
#     "scikit-learn==1.6.0",
# ]
# ///

import marimo

__generated_with = "0.10.9"
app = marimo.App()


@app.cell
def _():
    import random

    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    import plotly.express as px
    from drawdata import ScatterWidget
    return KMeans, ScatterWidget, np, pd, px, random


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    method = mo.ui.dropdown(
        options=["Random", "Manual"],
        value="Random",
        label="Generation Method"
    )
    return (method,)


@app.cell
def _(method):
    method
    return


@app.cell
def _(mo):
    points = mo.ui.number(value=200, start=10, stop=1000, label="Number of Points")
    return (points,)


@app.cell
def _(mo):
    k_clusters = mo.ui.number(value=5, start=2, stop=10, label="Number of Clusters")
    k_clusters
    return (k_clusters,)


@app.cell
def _(mo):
    random_button = mo.ui.button(label="Generate new data")
    return (random_button,)


@app.cell
def _(run_button):
    run_button
    return


@app.cell
def _(method, mo, random_button, widget):
    random_button if method.value == "Random" else mo.md(
        f"""
        Draw a dataset of points, then click the run button above!

        {widget}
        """
    )
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run k-means!")
    return (run_button,)


@app.cell
def _(np, random, random_button):
    random_button


    def _generate_data():
        n_clusters = random.randint(2, 10)
        np.random.randn()

        points = []
        for i in range(n_clusters):
            points.append(
                np.random.randn(100, 2) * np.random.uniform(-2, 2)
                + np.random.uniform(-2, 2)
            )
        return np.vstack(points)


    generated_points = _generate_data()
    return (generated_points,)


@app.cell
def _(
    KMeans,
    generated_points,
    k_clusters,
    method,
    np,
    pd,
    px,
    run_button,
    widget,
):
    fig = px.scatter(
        x=generated_points[:, 0], y=generated_points[:, 1], title="Random Points"
    ) if method.value == "Random" else None

    if run_button.value and method.value == "Random":
        kmeans = KMeans(n_clusters=k_clusters.value, random_state=42)
        clusters = kmeans.fit_predict(generated_points)
        df = pd.DataFrame(generated_points, columns=["x", "y"])
        df["cluster"] = clusters

        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="cluster",
            title="K-means Clustering (Random Data)",
            color_continuous_scale="viridis",
        )
    elif run_button.value and method.value == "Manual":
        df = widget.data_as_pandas
        if not df.empty:
            # Drop non-numeric columns (like 'colour')
            numeric_df = df.select_dtypes(include=[np.number])
            kmeans = KMeans(n_clusters=k_clusters.value, random_state=42)
            clusters = kmeans.fit_predict(numeric_df)
            df['cluster'] = clusters
            fig = px.scatter(
                df,
                x='x',
                y='y',
                color='cluster',
                title="K-means Clustering (Manual Data)",
                color_continuous_scale='viridis'
            )

    fig
    return clusters, df, fig, kmeans, numeric_df


@app.cell
def _(ScatterWidget, mo):
    widget = mo.ui.anywidget(ScatterWidget())
    return (widget,)


@app.cell
def _(mo):
    mo.md(
        """
        TODO:

        * Explanatory text about what k-means is
        * What are the centroids?
        * How many iterations does the algorithm run for?
        * What is the loss of the algorithm at each step?
        """
    )
    return


if __name__ == "__main__":
    app.run()
