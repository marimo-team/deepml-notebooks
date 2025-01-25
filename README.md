﻿# deepml-notebooks

> [!IMPORTANT]  
> The repo is now archived and interactive notebooks will directly be pushed to the [DML-OpenProblem](https://github.com/Open-Deep-ML/DML-OpenProblem) repo.

Steps:
1. `cd` to the appropriate problems/problem directory
2. To run the exported notebook, use:
    ```shell
    python -m http.server --directory docs/
    ```

3. Then navigate to localhost:8000 in your browser to view the rendered html-wasm.
   
> [!NOTE]
> The .html files were created from the .py using the following command
> ``` marimo export html-wasm notebook.py --mode run --no-show-code -o docs/ ```
