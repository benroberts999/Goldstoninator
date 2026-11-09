"""
Goldstoninator: Goldstone and Feynman diagrams for many-body perturbation theory, drawn from a line of
text, with the perturbation term that goes with them and its angular reduction.

    import goldstoninator as gd                      # Goldstone diagrams (goldstone.py)
    from goldstoninator import feynmanator as fd     # Feynman diagrams
    from goldstoninator import angularinator as ang  # angular reduction, C++ for ampsci

No dependencies beyond the Python standard library.
"""

from .goldstone import *  # noqa: F401,F403
