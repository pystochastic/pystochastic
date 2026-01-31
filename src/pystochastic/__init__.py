from importlib.metadata import version

__version__ = version("pystochastic")

del version  # Clean up namespace

__all__ = [
    __version__,
]
