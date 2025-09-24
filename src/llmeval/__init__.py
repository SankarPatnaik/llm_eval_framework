"""LLM Evaluation Framework package."""

import sys as _sys

if __name__ != "llmeval":
    # Allow importing the package both as ``llmeval`` and ``src.llmeval``
    # when running without installing the project. This keeps compatibility
    # with users invoking ``python -m src.llmeval...`` by registering the
    # package under the top-level name that the rest of the code expects.
    _sys.modules.setdefault("llmeval", _sys.modules[__name__])

__all__ = []
