from io import BytesIO
import sqlite3
import numpy as np
from numpy.typing import NDArray



def adapt_ndarray(array: NDArray) -> bytes:
    f = BytesIO()
    np.save(f, array)
    f.seek(0)
    result = f.read()
    f.close()
    return result

def convert_ndarray(source: bytes) -> NDArray:
    f = BytesIO(source)
    f.seek(0)
    result = np.load(f).astype(np.float32)
    f.close()
    return result

sqlite3.register_adapter(np.ndarray, adapt_ndarray)
sqlite3.register_converter('NDArray', convert_ndarray)
