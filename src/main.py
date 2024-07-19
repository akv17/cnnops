import ctypes
import numpy as np


lib = ctypes.CDLL("libmatmul/libmatmul.so") # Or full path to file  


class OpSpec(ctypes.Structure):
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


lib.create_op_spec.argtypes = (
    ctypes.c_int,
    ctypes.c_int,
    
    ctypes.POINTER(ctypes.c_int),
)
lib.create_op_spec.restype = OpSpec
lib.matmul.argtypes = (
    ctypes.c_int,
    OpSpec,
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
    ctypes.POINTER(ctypes.c_float),
)

a_nd = 2
a_shape = (ctypes.c_int * a_nd)(*[2, 4])
b_nd = 2
b_shape = (ctypes.c_int * b_nd)(*[4, 8])
spec = lib.create_op_spec(a_nd, b_nd, a_shape, b_shape)

a_np = np.random.normal(size=[*a_shape]).astype('float32')
b_np = np.random.normal(size=[*b_shape]).astype('float32')
c_np = np.zeros(shape=[spec.c_rows, spec.c_cols]).astype('float32')
c_np_res = np.dot(a_np, b_np)

a = (ctypes.c_float * a_np.size)(*a_np.ravel().tolist())
b = (ctypes.c_float * b_np.size)(*b_np.ravel().tolist())
c = (ctypes.c_float * c_np.size)(*c_np.ravel().tolist())

lib.matmul(0, spec, a, b, c)

print([*c])
print(c_np_res.ravel())