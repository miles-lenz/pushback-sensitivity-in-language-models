## 1. Starting & Syncing (The Git Loop)

Use these when setting up or pulling fresh code from your teammate.
*   `uv init` — Initializes a brand new project (creates `pyproject.toml`).
*   `uv sync` — The magic command. Run this after pulling your teammate's code. It reads the `uv.lock` file and instantly makes your local `.venv` exactly match theirs.

## 2. Managing Packages

Use these when you need to change the tools your project uses.
*   `uv add pandas` — Installs a package and saves it to `pyproject.toml` and `uv.lock`.
*   `uv add --dev pytest` — Installs a "development" package. Use `--dev` for tools that help you write code (like Ruff, pre-commit, or testing frameworks) but aren't needed to actually run the app in production.
*   `uv remove requests` — Uninstalls a package and safely cleans it out of your project files.

## 3. Running Your Code

Get out of the habit of typing just `python file.py`. Let `uv` handle the virtual environment for you.
*   `uv run python script.py` — Runs your script securely inside the project's isolated virtual environment. You never have to manually run `source .venv/bin/activate` again.
*   `uv run pytest` — Runs your testing suite using the exact project dependencies.

## 4. Code Quality (Your Setup)

The commands to keep your PRs perfectly formatted and argument-free.
*   `uv run ruff format .` — Instantly formats all code in your project to standard style guidelines.
*   `uv run ruff check . --fix` — Scans your code for bad practices/bugs and automatically fixes the ones it safely can.
*   `uv run pre-commit install` — Run this once per computer. It permanently attaches Ruff to your git commit.