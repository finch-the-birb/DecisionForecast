"""CLI entry point alias for src.training.train."""

import runpy

if __name__ == "__main__":
    runpy.run_module("src.training.train", run_name="__main__")
