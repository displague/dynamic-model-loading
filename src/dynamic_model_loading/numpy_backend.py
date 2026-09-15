"""Explicit thread control for the pinned Windows NumPy/OpenBLAS apparatus."""
import ctypes
import hashlib
from pathlib import Path


def set_blas_threads(count=4):
    import numpy as np
    if type(count) is not int or count <= 0:
        raise ValueError('Positive BLAS thread count required')
    paths = list((Path(np.__file__).resolve().parent.parent/'numpy.libs').glob('*openblas*.dll'))
    if len(paths) != 1:
        raise RuntimeError('Expected the pinned Windows OpenBLAS library')
    library = ctypes.CDLL(str(paths[0]))
    get = getattr(library, 'scipy_openblas_get_num_threads64_')
    set_ = getattr(library, 'scipy_openblas_set_num_threads64_')
    get.argtypes = []
    get.restype = ctypes.c_int
    set_.argtypes = [ctypes.c_int]
    set_.restype = None
    set_(count)
    if get() != count:
        raise RuntimeError('Could not enforce BLAS thread count')
    with paths[0].open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(threads=get(), library=paths[0].name, sha256=digest)
