# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy==2.2.1",
#     "scipy==1.13.1",
#     "plotly==5.24.1",
# ]
# ///

import marimo

__generated_with = "0.10.16"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Understanding Kullback-Leibler Divergence for Normal Distributions

        The [Kullback-Leibler (KL) Divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) is a fundamental concept in information theory and probability that measures how one probability distribution differs from a reference distribution.
        """
    ).center()
    return


@app.cell(hide_code=True)
def _(mo):
    # math definition accordion
    definition = mo.md(r"""
    For two normal distributions $P \sim N(\mu_P, \sigma_P^2)$ and $Q \sim N(\mu_Q, \sigma_Q^2)$, 
    the KL Divergence is defined analytically as:

    \[
    D_{KL}(P \parallel Q) = \log\left(\frac{\sigma_Q}{\sigma_P}\right) + 
    \frac{\sigma_P^2 + (\mu_P - \mu_Q)^2}{2\sigma_Q^2} - \frac{1}{2}
    \]

    Key components:

    - $\mu_P$, $\mu_Q$: Mean of distributions $P$ and $Q$

    - $\sigma_P$, $\sigma_Q$: Standard deviation of distributions $P$ and $Q$
    """)

    mo.accordion({"### Mathematical Formulation": definition})
    return (definition,)


@app.cell(hide_code=True)
def _(insights):
    insights
    return


@app.cell(hide_code=True)
def _(mo):
    # insights accordion
    insights = mo.accordion({
        "🔍 Understanding KL Divergence": mo.md("""
        **Key Insights:**

        1. KL Divergence is always non-negative
        2. It measures the information lost when Q is used to approximate P
        3. Not symmetric: D_KL(P || Q) ≠ D_KL(Q || P)

        **Interpretation of Values:**
        
        - 0: Distributions are identical
        
        - Small value: Distributions are very similar
        
        - Large value: Significant difference between distributions
        """),

        "📊 Factors Affecting KL Divergence": mo.md("""
        KL Divergence is influenced by:

        1. **Mean Difference**: Larger difference increases divergence
        2. **Variance Ratio**: Varying standard deviations impacts divergence
        3. **Shape of Distributions**: More spread-out distributions lead to higher divergence
        """)
    })
    return (insights,)


@app.cell
def _(distribution_inputs):
    distribution_inputs
    return


@app.cell(hide_code=True)
def _(mo):
    # distribution inputs
    mu_p = mo.ui.number(
        value=0.0, 
        label="Mean of Distribution P (μP)",
        step=0.1,
        start=-10,
        stop=10
    )

    sigma_p = mo.ui.number(
        value=1.0, 
        label="Standard Deviation of Distribution P (σP)",
        step=0.1,
        start=0.1,
        stop=10
    )

    mu_q = mo.ui.number(
        value=1.0, 
        label="Mean of Distribution Q (μQ)",
        step=0.1,
        start=-10,
        stop=10
    )

    sigma_q = mo.ui.number(
        value=1.0, 
        label="Standard Deviation of Distribution Q (σQ)",
        step=0.1,
        start=0.1,
        stop=10
    )

    # stack layout
    distribution_inputs = mo.vstack([
        mo.md("### Distribution Parameters"),
        mu_p,
        sigma_p,
        mu_q,
        sigma_q
    ])
    return distribution_inputs, mu_p, mu_q, sigma_p, sigma_q


@app.cell
def _(np):
    # KL Divergence calculation function
    def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q):
        """
        Calculate KL Divergence between two normal distributions

        Args:
            mu_p (float): Mean of distribution P
            sigma_p (float): Standard deviation of distribution P
            mu_q (float): Mean of distribution Q
            sigma_q (float): Standard deviation of distribution Q

        Returns:
            float: KL Divergence from P to Q
        """
        return (
            np.log(sigma_q/sigma_p) + 
            (sigma_p**2 + (mu_p - mu_q)**2) / (2 * sigma_q**2) - 
            0.5
        )
    return (kl_divergence_normal,)


@app.cell
def _(run_button):
    run_button
    return


@app.cell
def _(mo):
    # Run button linked to KL Divergence calculation
    run_button = mo.ui.run_button(label="Calculate KL Divergence")
    return (run_button,)


@app.cell
def _(result_display):
    result_display
    return


@app.cell
def _(kl_divergence_normal, mo, mu_p, mu_q, run_button, sigma_p, sigma_q):
    kl_result = 0.0
    if run_button.value:
        kl_result = kl_divergence_normal(
            mu_p.value, 
            sigma_p.value, 
            mu_q.value, 
            sigma_q.value
        )

    result_display = mo.callout(
        mo.md(f"**KL Divergence (P || Q):** {kl_result:.4f}"),
        kind="info"
    )
    return kl_result, result_display


@app.cell
def _(visualize_button):
    visualize_button
    return


@app.cell
def _(mo, mu_p, mu_q, np, pd, px, scipy, sigma_p, sigma_q):
    # Visualization of distributions
    def plot_normal_distributions():
        x = np.linspace(-10, 10, 200)

        # P distribution
        p_dist = scipy.stats.norm.pdf(x, mu_p.value, sigma_p.value)

        # Q distribution
        q_dist = scipy.stats.norm.pdf(x, mu_q.value, sigma_q.value)

        # Plotly DataFrame
        df = pd.DataFrame({
            'x': x,
            'P Distribution': p_dist,
            'Q Distribution': q_dist
        })

        # Plotly plot
        fig = px.line(
            df, 
            x='x', 
            y=['P Distribution', 'Q Distribution'],
            title='Comparison of Normal Distributions',
            labels={'value': 'Probability Density', 'x': 'Value'}
        )

        return fig

    # Visualization button
    visualize_button = mo.ui.run_button(label="Visualize Distributions")
    return plot_normal_distributions, visualize_button


@app.cell
def _(plot_normal_distributions, visualize_button):
    distribution_plot = None
    if visualize_button.value:
        distribution_plot = plot_normal_distributions()
    distribution_plot
    return (distribution_plot,)


@app.cell
def _(conclusion):
    conclusion
    return


@app.cell
def _(mo):
    # Conclusion and next steps
    conclusion = mo.vstack([
        mo.callout(
            mo.md("""
                **Congratulations!** 
                You've explored the Kullback-Leibler Divergence between Normal Distributions. 

                Key takeaways:
                
                - Learned the mathematical formulation
                
                - Calculated KL Divergence interactively
                
                - Visualized distribution differences
            """),
            kind="success"
        ),
        mo.accordion({
            "🎯 Practical Applications": mo.md("""
                
                - Machine Learning Model Evaluation
                
                - Variational Inference
                
                - Probabilistic Generative Models
                
                - Information Theory Research
                
                - Anomaly Detection
            """),
            "🚀 Next Learning Steps": mo.md("""
                
                1. Head over to the Problem Description tag to implement this!
                
                2. Explore KL Divergence for other distributions
                
                3. Understand its role in Variational Autoencoders (VAEs)
                
                4. Compare with other divergence measures
                
            """)
        })
    ])
    return (conclusion,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    # Import necessary libraries
    import numpy as np
    import scipy.stats
    import pandas as pd
    import plotly.express as px
    return np, pd, px, scipy


if __name__ == "__main__":
    app.run()
