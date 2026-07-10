# loaders package
# Each file format gets its own loader module. factory.py is the single
# entry point the rest of the app should import from.

from loaders.factory import load_any_file

__all__ = ["load_any_file"]
