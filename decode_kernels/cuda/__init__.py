import glob
import os

BUILD_COMMAND = "python setup.py build_ext --inplace"

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_IMPORT_ERROR = None
try:
    from . import _ext
except (ImportError, OSError) as e:      # not built, or built against a different torch
    _ext = None
    # Python blames a circular import here when the real cause is a missing .so; say which.
    if glob.glob(os.path.join(_PKG_DIR, "_ext*.so")):
        _IMPORT_ERROR = f"a compiled _ext exists but will not load -- {type(e).__name__}: {e}"
    else:
        _IMPORT_ERROR = f"no compiled _ext*.so in {_PKG_DIR}"


def is_available():
    return _ext is not None


def unavailable_reason():
    return _IMPORT_ERROR


def require():
    """The extension module, or a RuntimeError that says how to get one."""
    if _ext is None:
        raise RuntimeError(f"the CUDA extension is not importable -- build it with "
                           f"'{BUILD_COMMAND}'. Underlying error: {_IMPORT_ERROR}")
    return _ext


def extension_path():
    return getattr(_ext, "__file__", None) if _ext is not None else None


def build_info():
    """Compile-time facts from the binary, plus which binary they came from."""
    info = dict(require().build_info())
    path = extension_path()
    info["extension_path"] = path
    info["extension_mtime"] = os.path.getmtime(path) if path else None
    return info


def smoke_fill(tensor, value):
    return require().smoke_fill(tensor, value)


__all__ = ["BUILD_COMMAND", "build_info", "extension_path", "is_available", "require",
           "smoke_fill", "unavailable_reason"]
