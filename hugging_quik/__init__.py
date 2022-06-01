from importlib import import_module

__all__ = [
    "io",
    "model",
    "token",
    "utils",
]
__version__ = "0.0.1"

for submodule in __all__:
    import_module(f"hugging_quik.{submodule}")
