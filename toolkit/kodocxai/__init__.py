"""KoDocXAI toolkit - package wrapper.

The modules keep their original flat file names (``kodx_*.py`` etc.); those
names are frozen for reproducibility (the paper's seed strings and audit
manifests reference them verbatim).  This ``__init__`` prepends the package
directory to ``sys.path`` so the modules' original flat imports
(``from kodx_data import ...``) keep working without any code change.
"""
import os as _os
import sys as _sys

_here = _os.path.dirname(_os.path.abspath(__file__))
if _here not in _sys.path:
    _sys.path.insert(0, _here)
