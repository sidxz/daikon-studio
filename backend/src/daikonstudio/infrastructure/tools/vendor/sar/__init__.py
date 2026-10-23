"""Steve Berthel's SAR modules, copied unchanged. See VENDORED.md.

`sar_edit.py` does a bare `import sar_variable as sv`, so this directory must be on
`sys.path` for the files to stay byte-identical to their source.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
# ponytail: sys.path shim so the vendored files stay byte-identical;
# a package-relative rewrite is the upgrade if we ever fork them
if _HERE not in sys.path:
    # Appended, not inserted: this runs at API startup, and in front of the path
    # it would be searched ahead of the stdlib and site-packages for every later
    # import in the process, so the next file vendored in under a name like
    # `codecs.py` or `logging.py` would shadow the real one. The four modules
    # below resolve the same either way.
    sys.path.append(_HERE)

import sar_cliffs  # noqa: E402
import sar_core  # noqa: E402
import sar_edit  # noqa: E402
import sar_variable  # noqa: E402

__all__ = ["sar_cliffs", "sar_core", "sar_edit", "sar_variable"]
