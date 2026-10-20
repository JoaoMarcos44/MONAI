"""Run with required dependencies only, reusing their installed files read-only.

Use Python -S so unrelated optional packages from the host are not exposed.
This avoids copying or reinstalling large torch/numpy distributions.
"""

import importlib.machinery
import os
import runpy
import sys

REQUIRED = {
    "torch", "torchgen", "functorch", "numpy", "filelock", "fsspec", "jinja2",
    "markupsafe", "networkx", "sympy", "mpmath", "typing_extensions", "setuptools",
    "packaging", "_distutils_hack", "pkg_resources",
}


class RequiredPackages:
    def find_spec(self, fullname, path=None, target=None):
        if path is None and fullname in REQUIRED:
            return importlib.machinery.PathFinder.find_spec(fullname, [os.environ["MONAI_QA_REQUIRED_SITE"]])
        return None


sys.meta_path.insert(0, RequiredPackages())
arguments = sys.argv[1:]
if arguments[0] == "-m":
    sys.argv = arguments[1:]
    runpy.run_module(arguments[1], run_name="__main__", alter_sys=True)
elif arguments[0] == "-c":
    sys.argv = ["-c"]
    exec(compile(arguments[1], "<string>", "exec"), {"__name__": "__main__"})
else:
    sys.argv = arguments
    runpy.run_path(arguments[0], run_name="__main__")
