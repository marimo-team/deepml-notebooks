import marimo

__generated_with = "0.10.9"
app = marimo.App()


@app.cell
def _(mo):
    demo_type = mo.ui.dropdown(
        options=["K-means Interactive Demo", "Iris Dataset Clustering"],
        value="K-means Interactive Demo",
        label="Select Demo"
    )
    return (demo_type,)


@app.cell
def _(demo_type):
    demo_type
    return


@app.cell
def _(mo):
    method = mo.ui.dropdown(
        options=["Random", "Manual"],
        value="Random",
        label="Generation Method"
    )
    return (method,)


@app.cell
def _():
    # method
    return


@app.cell
def _(mo):
    points = mo.ui.number(value=200, start=10, stop=1000, label="Number of Points")
    return (points,)


@app.cell
def _():
    # points
    return


@app.cell
def _(mo):
    data_clusters = mo.ui.number(value=5, start=2, stop=10, label="Data Clusters")
    return (data_clusters,)


@app.cell
def _():
    # data_clusters
    return


@app.cell
def _(mo):
    k_clusters = mo.ui.number(value=5, start=2, stop=10, label="Number of Clusters")
    return (k_clusters,)


@app.cell
def _():
    # k_clusters
    return


@app.cell
def _(mo):
    run_button = mo.ui.button(label="Run")
    return (run_button,)


@app.cell
def _():
    # run_button
    return


@app.cell
def _(mo):
    random_button = mo.ui.button(label="Random")
    return (random_button,)


@app.cell
def _():
    # random_button
    return


@app.cell
def _(
    KMeans,
    ScatterWidget,
    data_clusters,
    datasets,
    demo_type,
    k_clusters,
    method,
    mo,
    np,
    pd,
    points,
    px,
    random_button,
    run_button,
):
    if demo_type.value == "K-means Interactive Demo":
        if method.value == "Manual":
            widget = mo.ui.anywidget(ScatterWidget())
            if run_button.value:
                X = widget.data_as_pandas
                if not X.empty:
                    kmeans = KMeans(n_clusters=k_clusters.value, random_state=42)
                    clusters = kmeans.fit_predict(X)
                    fig = px.scatter(
                        X, 
                        x=X.columns[0], 
                        y=X.columns[1],
                        color=clusters,
                        title="K-means Clustering (Manual Data)"
                    )
                    display = mo.ui.plotly(fig)
                else:
                    display = widget
            else:
                display = widget
        else:
            generated_points = np.random.rand(points.value, 2) * 500
            if run_button.value:
                kmeans = KMeans(n_clusters=k_clusters.value, random_state=42)
                clusters = kmeans.fit_predict(generated_points)
                fig = px.scatter(
                    x=generated_points[:, 0], 
                    y=generated_points[:, 1],
                    color=clusters,
                    title="K-means Clustering (Random Data)"
                )
            else:
                fig = px.scatter(
                    x=generated_points[:, 0], 
                    y=generated_points[:, 1],
                    title="Random Points"
                )
            display = mo.ui.plotly(fig)

        mo.hstack([
            mo.vstack([method, points, data_clusters, k_clusters, run_button, random_button]),
            display
        ])

    elif demo_type.value == "Iris Dataset Clustering":
        iris = datasets.load_iris()
        data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        kmeans = KMeans(n_clusters=k_clusters.value, random_state=42)
        data['cluster'] = kmeans.fit_predict(data)

        fig = px.scatter_3d(
            data, 
            x='sepal length (cm)', 
            y='sepal width (cm)', 
            z='petal length (cm)',
            color='cluster',
            title='K-Means Clustering on Iris Dataset'
        )

        mo.vstack([k_clusters, mo.ui.plotly(fig)])
    return (
        X,
        clusters,
        data,
        display,
        fig,
        generated_points,
        iris,
        kmeans,
        widget,
    )


@app.cell
def _(
    data_clusters,
    demo_type,
    display,
    fig,
    k_clusters,
    method,
    mo,
    points,
    random_button,
    run_button,
):
    [mo.hstack([
            mo.vstack([method, points, data_clusters, k_clusters, run_button, random_button]),
            display
    ]) if demo_type.value == "K-means Interactive Demo" else mo.vstack([k_clusters, mo.ui.plotly(fig)])]
    return


@app.cell
def _():
    # fig
    # display
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn import datasets
    import plotly.express as px
    from drawdata import ScatterWidget
    return KMeans, ScatterWidget, datasets, mo, np, pd, px


if __name__ == "__main__":
    app.run()
