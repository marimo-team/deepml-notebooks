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
    k_clusters = mo.ui.number(value=5, start=2, stop=10, label="Number of Clusters")
    return (k_clusters,)


@app.cell
def _():
    # k_clusters
    return


@app.cell
def _(mo):
    run_button = mo.ui.run_button(label="Run")
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
def _(ScatterWidget, mo):
    widget = mo.ui.anywidget(ScatterWidget())
    return (widget,)


@app.cell
def _():
    # _df = widget.data_as_pandas
    # _df
    return


@app.cell
def _():
    # widget.value
    return


@app.cell
def _(
    KMeans,
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
    widget,
):
    if demo_type.value == "K-means Interactive Demo":
        # if method.value == "Manual":
        #     # widget = mo.ui.anywidget(ScatterWidget())
        #     if run_button.value:
        #         df = widget.data_as_pandas
        #         if not df.empty:
        #             kmeans = KMeans(n_clusters=k_clusters.value, random_state=42)
        #             clusters = kmeans.fit_predict(df)
        #             df['cluster'] = clusters
        #             fig = px.scatter(
        #                 df,
        #                 x='x',
        #                 y='y',
        #                 color='cluster',
        #                 title="K-means Clustering (Manual Data)",
        #                 color_continuous_scale='viridis'
        #             )
        #             display = mo.ui.plotly(fig)
        #         else:
        #             display = widget
        #     else:
        #         display = widget
        if method.value == "Manual":
            # widget = mo.ui.anywidget(ScatterWidget())
            if run_button.value:
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
                df = pd.DataFrame(generated_points, columns=['x', 'y'])
                df['cluster'] = clusters

                fig = px.scatter(
                    df,
                    x='x',
                    y='y',
                    color='cluster',
                    title="K-means Clustering (Random Data)",
                    color_continuous_scale='viridis'
                )
            else:
                fig = px.scatter(
                    x=generated_points[:, 0],
                    y=generated_points[:, 1],
                    title="Random Points"
                )
            display = mo.ui.plotly(fig)

        mo.hstack([
            mo.vstack([method, points, k_clusters, run_button, random_button]),
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
        clusters,
        data,
        df,
        display,
        fig,
        generated_points,
        iris,
        kmeans,
        numeric_df,
    )


@app.cell
def _():
    # [mo.hstack([
    #         mo.vstack([method, points, k_clusters, run_button, random_button]),
    #         display
    # ]) if demo_type.value == "K-means Interactive Demo" else mo.vstack([k_clusters, mo.ui.plotly(fig)])]
    return


@app.cell
def _(
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
    mo.hstack([
        (mo.vstack([method, points, k_clusters, run_button, random_button]),
        display) if demo_type.value == "K-means Interactive Demo" and method.value == "Random"
        else
        (mo.vstack([method, k_clusters, run_button]),
        display) if demo_type.value == "K-means Interactive Demo" and method.value == "Manual"
        else
        (mo.vstack([k_clusters, mo.ui.plotly(fig)]))
    ])
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
