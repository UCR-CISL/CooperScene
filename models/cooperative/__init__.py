import os as _os
import sys as _sys

# Use the in-repo OpenCOOD bundle for `import opencood` (no external install).
_VENDORED_OPENCOOD = _os.path.abspath(
    _os.path.join(_os.path.dirname(__file__), '..', '..', 'vendor'))
if _os.path.isdir(_os.path.join(_VENDORED_OPENCOOD, 'opencood')) \
        and _VENDORED_OPENCOOD not in _sys.path:
    _sys.path.insert(0, _VENDORED_OPENCOOD)

from . import datasets  # noqa: F401,E402
from . import evaluation  # noqa: F401
from . import detectors  # noqa: F401
from . import data_preprocessors  # noqa: F401
from . import sub_modules  # noqa: F401
from . import fuse_modules  # noqa: F401
from .det_head import DetHead  # noqa: F401
