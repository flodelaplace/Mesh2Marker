"""Zero-wheel guardrail: the add-on import path must not pull pydantic.

Run in a fresh interpreter so that other tests importing
:mod:`mesh2marker.validation` cannot pollute ``sys.modules``.
"""

import subprocess
import sys


def test_core_import_path_has_no_pydantic():
    code = (
        "import sys\n"
        "import mesh2marker\n"
        "import mesh2marker.models\n"
        "import mesh2marker.io\n"
        "import mesh2marker.checks\n"
        "leaked = sorted(m for m in sys.modules if 'pydantic' in m)\n"
        "assert not leaked, leaked\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
