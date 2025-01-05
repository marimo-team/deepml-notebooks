import marimo

__generated_with = "0.10.9"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    # For now, title and intro in LaTeX
    mo.md(
        r"""
        # Understanding the Softmax Activation Function

        The Softmax function transforms a vector of numbers into a probability distribution. It's crucial in neural networks, especially for multi-class classification tasks.

        ### Mathematical Definition

        For an input vector $\mathbf{z}$ with $n$ components, the Softmax function is defined as:

        \[
        \text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}
        \]

        where $i$ ranges from 1 to $n$, and each output $\text{softmax}(z_i)$ represents the probability of class $i$.
        """
    ).center()
    return


@app.cell
def _(mo):
    # Interactive elements for formula exploration
    temperature = mo.ui.slider(
        start=0.5,  # Updated lower bound for stability
        stop=5.0,
        value=1.0,
        step=0.1,
        label="Temperature (τ)"
    )

    vector_size = mo.ui.slider(
        start=2,
        stop=5,
        value=3,
        step=1,
        label="Vector Size (n)"
    )
    return temperature, vector_size


@app.cell(hide_code=True)
def _(mo, temperature, vector_size):
    # Formula text that updates with parameter changes
    formula_md = mo.md(
        f"""
            ### Temperature-Scaled Softmax
            
            With temperature parameter $\\tau = {temperature.value:.1f}$ and vector size $n = {vector_size.value}$, 
            the temperature-scaled Softmax is:
            
            \\[
            \\text{{softmax}}(z_i, \\tau) = \\frac{{e^{{z_i/{temperature.value:.1f}}}}}{{\\sum_{{j=1}}^{{{vector_size.value}}} e^{{z_j/{temperature.value:.1f}}}}}
            \\]
            
            - Higher temperature ($\\tau > 1$) makes the distribution more uniform
            - Lower temperature ($\\tau < 1$) makes the distribution more peaked
            """
    )
    return (formula_md,)


@app.cell
def _(formula_md):
    formula_md
    return


@app.cell
def _(mo, vector_size):
    # Input range control
    x_range = mo.ui.range_slider(
        start=-10,
        stop=10,
        step=0.5,
        value=[-5, 5],
        label="X-axis Range"
    )

    # Array of number inputs dynamically based on vector_size
    input_numbers = [
        mo.ui.number(
            value=1.0,
            start=-10,
            stop=10,
            step=0.1,
            label=f"Value {i+1}"
        )
        for i in range(vector_size.value)
    ]

    array_nums = mo.ui.array(input_numbers)

    # Properly format batch with placeholder
    input_values = mo.ui.batch(
        mo.md("""
        ### Sample Input Values
        Current vector: {inputs}
        """),
        {"inputs": array_nums}
    )
    return array_nums, input_numbers, input_values, x_range


@app.cell(hide_code=True)
def _(array_nums):
    # input_values
    array_nums
    return


@app.cell
def _(array_nums, mo, np, plt, temperature, vector_size, x_range):
    @mo.cache(pin_modules=True)
    def plot_softmax():
        # Validate x_range
        if x_range.value[0] >= x_range.value[1]:
            raise ValueError("Invalid x_range: start value must be less than stop value.")

        x = np.linspace(x_range.value[0], x_range.value[1], 1000)
        scores = np.vstack([x] + [np.zeros_like(x)] * (vector_size.value - 1))

        # Apply temperature scaling
        scaled_scores = scores / temperature.value
        exp_scores = np.exp(scaled_scores)
        softmax_values = exp_scores / np.sum(exp_scores, axis=0)

        plt.figure(figsize=(12, 7))
        for i in range(vector_size.value):
            plt.plot(x, softmax_values[i], 
                     label=f'Class {i+1}', 
                     linewidth=2)

        # Plot current input values
        input_array = np.array([num.value for num in array_nums.elements])
        if len(input_array) >= vector_size.value:
            exp_inputs = np.exp(input_array[:vector_size.value] / temperature.value)
            softmax_inputs = exp_inputs / np.sum(exp_inputs)
            plt.scatter(input_array[:vector_size.value], 
                        softmax_inputs,
                        c='red',
                        s=100,
                        label='Current inputs')

        plt.grid(True, alpha=0.3)
        plt.title(f'Softmax Function (τ = {temperature.value:.1f})')
        plt.xlabel('Input values (z)')
        plt.ylabel('Softmax probabilities')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim(-0.1, 1.1)

        return mo.as_html(plt.gca())

    # All relevant controls and display decent layout
    controls = mo.hstack([
        mo.vstack([
            mo.md("### Function Parameters"),
            temperature,
            vector_size
        ]),
        mo.vstack([
            mo.md("### Input Range"),
            x_range
        ])
    ], justify="space-between")
    return controls, plot_softmax


@app.cell
def _(plot_softmax):
    plot_softmax()
    return


@app.cell
def _(controls):
    controls
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


if __name__ == "__main__":
    app.run()
