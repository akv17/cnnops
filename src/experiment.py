import time
import dataclasses

import torch
import numpy as np


@dataclasses.dataclass
class ExperimentResult:
    np_result: 'Any' = dataclasses.field(repr=False)
    lib_result: 'Any' = dataclasses.field(repr=False)


class Experiment:

    def __init__(self, lib, a_shape, b_shape, num_runs=1):
        self.lib = lib
        self.a_shape = a_shape
        self.b_shape = b_shape
        self.num_runs = num_runs

    def compare(self, tol=1e-5):
        a = torch.rand(*self.a_shape).float()
        b = torch.rand(*self.b_shape).float()
        res_torch = self._call_torch(a, b)
        res_lib = self._call_lib(a, b)
        flag = np.allclose(res_torch, res_lib, tol, tol)
        print('compare:')
        print(f'  torch size: {len(res_torch)}')
        print(f'    lib size: {len(res_lib)}')
        print(f'        flag: {flag}')

    def _call_torch(self, a, b):
        c = torch.matmul(a, b).numpy().ravel().tolist()
        return c

    def _call_lib(self, a, b):
        a = a.numpy()
        b = b.numpy()
        spec = self.lib.create_op_spec(a, b)
        c = self.lib.matmul(spec, a, b)
        c = [*c]
        return c
