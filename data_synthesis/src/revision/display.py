"""Notebook display helpers."""

from src.revision.common import *

def display_figure_once(fig):
    """Display a Matplotlib figure in notebooks without the inline backend echo."""
    from IPython.display import display

    display(fig)
    plt.close(fig)

def display_result_once(result):
    """Display a Figure4Result-like object once in notebooks."""
    display_figure_once(result.fig)
    return result
