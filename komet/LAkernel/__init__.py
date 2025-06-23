import ctypes

def LAkernel_eval(x, y, path='komet/LAkernel/LAkernel.so'):
    """
    Compute local alignment kernel between two strings using the compiled C library.

    Args:
        x (str): First sequence.
        y (str): Second sequence.
        path (str): Path to the compiled shared object (.so).

    Returns:
        float: Kernel score.
    """
    lib = ctypes.CDLL(path)
    lib.LAkernel.restype = ctypes.c_double

    x_c = ctypes.c_char_p(x.encode('utf-8'))
    y_c = ctypes.c_char_p(y.encode('utf-8'))

    return lib.LAkernel(x_c, y_c)
