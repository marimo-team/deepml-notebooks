# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.0",
#     "numpy==2.2.1",
# ]
# ///

import marimo

__generated_with = "0.10.12"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Understanding the Swish Activation Function

        The [Swish activation function](https://en.wikipedia.org/wiki/Swish_function), introduced by Google researchers, is a self-gated activation function that has shown promising results in deep neural networks. Let's explore it interactively!
        """
    ).center()
    return


@app.cell
def _(mo):
    value = mo.md(r"""
    For an input $x$, the Swish function is defined as:

    \[
    \text{Swish}(x) = x \times \sigma(x)
    \]

    where $\sigma(x)$ is the sigmoid function:

    \[
    \sigma(x) = \frac{1}{1 + e^{-x}}
    \]

    This creates a smooth, non-monotonic function that resembles a combination of ReLU and linear function.
    """)
    mo.accordion({"### Mathematical Definition" : value})
    return (value,)


@app.cell
def _(mo):
    beta = mo.ui.slider(
        start=0.1,
        stop=2.0,
        value=1.0,
        step=0.1,
        label="Beta (β)"
    )

    _callout = mo.callout(
        mo.md("""
            Adjust beta to control the shape of the Swish function.

            **Observe how:**

            - Higher beta values make Swish more like ReLU
            - Lower beta values make the negative part more pronounced
            - β = 1.0 is the standard Swish function
        """),
        kind="info"
    )

    x_range = mo.ui.range_slider(
        start=-10,
        stop=10,
        step=0.5,
        value=[-5, 5],
        label="X-axis Range"
    )

    controls = mo.vstack([
        mo.md("### Adjust Parameters"),
        mo.hstack([
            mo.vstack([
                beta,
                mo.accordion({
                    "About Beta": _callout
                })
            ]),
            mo.vstack([
                x_range,
                mo.accordion({
                    "About Range": "Adjust to see different regions of the function."
                })
            ])
        ])
    ])
    return beta, controls, x_range


@app.cell
def _(mo):
    test_input = mo.ui.number(
        value=0.0,
        start=-10,
        stop=10,
        step=0.1,
        label="Test Input Value"
    )

    input_controls = mo.vstack([
        mo.md("### Test Specific Values"),
        test_input,
        mo.accordion({
            "About Testing": "Enter specific values to see their Swish outputs."
        })
    ])
    return input_controls, test_input


@app.cell
def _(beta, mo):
    formula_display = mo.vstack([
        mo.md(
            f"""
            ### Current Swish Configuration

            With beta parameter $\\beta = {beta.value:.1f}$, the current Swish function is:

            \\[
            f(x) = x \\times \\sigma(\\beta x) = x \\times \\frac{{1}}{{1 + e^{{-{beta.value:.1f}x}}}}
            \\]

            Key Properties:
            - Non-monotonic function
            - Smooth and continuous
            - Bounded below by approximately -0.278
            - Linear scaling for large positive values
            """
        ),
    ])
    return (formula_display,)


@app.cell
def _(formula_display):
    formula_display
    return


@app.cell(hide_code=True)
def _(beta, mo, np, plt, test_input, x_range):
    @mo.cache(pin_modules=True)
    def plot_swish():
        if x_range.value[0] >= x_range.value[1]:
            raise ValueError("Invalid x_range: start value must be less than stop value.")

        x = np.linspace(x_range.value[0], x_range.value[1], 1000)
        sigmoid = 1 / (1 + np.exp(-beta.value * x))
        y = x * sigmoid

        plt.figure(figsize=(12, 7))

        # Plot Swish function
        plt.plot(x, y, 
                label='Swish function', 
                color='blue', 
                linewidth=2)

        # Plot sigmoid component
        plt.plot(x, sigmoid, 
                label='Sigmoid component', 
                color='red', 
                linestyle='--',
                alpha=0.5)

        # Plot test point if within range (adjust slider values accordingly)
        if x_range.value[0] <= test_input.value <= x_range.value[1]:
            test_sigmoid = 1 / (1 + np.exp(-beta.value * test_input.value))
            test_output = test_input.value * test_sigmoid
            plt.scatter([test_input.value], [test_output], 
                       color='green', s=100, 
                       label=f'Test point: f({test_input.value:.2f}) = {test_output:.2f}')

        plt.grid(True, alpha=0.3)
        plt.title(f'Swish Function (β = {beta.value:.1f})')
        plt.xlabel('Input (x)')
        plt.ylabel('Output (Swish(x))')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Add zero lines
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

        plot_display = mo.vstack([
            mo.as_html(plt.gca()),
        ])

        return plot_display
    return (plot_swish,)


@app.cell
def _(controls):
    controls
    return


@app.cell
def _(input_controls):
    input_controls
    return


@app.cell
def _(plot_swish):
    plot_swish()
    return


@app.cell
def _(mo):
    conclusion = mo.vstack([
        mo.callout(
            mo.md("""
                **Key Takeaways!** 
                Through this interactive exploration of the Swish function, you've learned:

                - How Swish combines linear and sigmoid behaviors
                - The effect of the beta parameter on function shape
                - Why Swish can outperform ReLU in deep networks
                - The non-monotonic nature of the function
            """),
            kind="success"
        ),
        mo.accordion({
            "Next Steps": mo.md("""
                1. **Implementation:** Try implementing Swish in your neural networks
                2. **Compare:** See how it performs against ReLU and other activations
                3. **Experiment:** Test different beta values in your models
                4. **Advanced:** Study the gradient behavior of Swish
            """),
            "🎯 Common Applications": mo.md("""
                - Deep neural networks
                - Image classification tasks
                - Natural language processing
                - When ReLU performance plateaus
                - Models requiring smooth activation functions
            """),
        })
    ])
    return (conclusion,)


@app.cell
def _(mo):
    mo.md(f"""
    This interactive learning experience was designed to help you understand the Swish activation function. Hope this helps in your deep learning journey!
    """)
    return


@app.cell
def _(conclusion):
    conclusion
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    return np, plt


if __name__ == "__main__":
    app.run()
