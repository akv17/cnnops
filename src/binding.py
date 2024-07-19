import ctypes


class LibMatmul:

    @classmethod
    def load(cls, fp='libmatmul/libmatmul.so'):
        lib = ctypes.CDLL(fp)
        ob = cls(lib=lib)
        ob._bind()
        return ob
    
    def __init__(self, lib):
        self._lib = lib

    def create_op_spec(self, a, b):
        a_nd = a.ndim
        b_nd = b.ndim
        a_shape = (ctypes.c_int * a_nd)(*a.shape)
        b_shape = (ctypes.c_int * b_nd)(*b.shape)
        spec = self._lib.create_op_spec(a_nd, b_nd, a_shape, b_shape)
        return spec
    
    def matmul(self, spec, a, b, kernel=0):
        a = (ctypes.c_float * a.size)(*a.ravel().tolist())
        b = (ctypes.c_float * b.size)(*b.ravel().tolist())
        c = (ctypes.c_float * spec.c_size)(*([0.0] * spec.c_size))
        self._lib.matmul(kernel, spec, a, b, c)
        return c

    def _bind(self):
        self._lib.create_op_spec.argtypes = (
            ctypes.c_int,
            ctypes.c_int,
            
            ctypes.POINTER(ctypes.c_int),
        )
        self._lib.create_op_spec.restype = _OpSpec
        self._lib.matmul.argtypes = (
            ctypes.c_int,
            _OpSpec,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        )


class _OpSpec(ctypes.Structure):
    _fields_ = (
        ('a_rows', ctypes.c_int),
        ('a_cols', ctypes.c_int),
        ('a_stride', ctypes.c_int),
        ('b_rows', ctypes.c_int),
        ('b_cols', ctypes.c_int),
        ('b_stride', ctypes.c_int),
        ('c_rows', ctypes.c_int),
        ('c_cols', ctypes.c_int),
        ('c_stride', ctypes.c_int),
        ('c_size', ctypes.c_int),
        ('batch_size', ctypes.c_int),
        ('batch_a_stride', ctypes.c_int),
        ('batch_b_stride', ctypes.c_int),
        ('batch_c_stride', ctypes.c_int),
    )
