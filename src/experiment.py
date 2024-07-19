import time
import dataclasses

import torch
import numpy as np


@dataclasses.dataclass
class BenchmarkResult:
    n: int
    torch_times: list = dataclasses.field(repr=False)
    lib_times: list = dataclasses.field(repr=False)

    @property
    def torch_time(self):
        return np.mean(self.torch_times).item()

    @property
    def lib_time(self):
        return np.mean(self.lib_times).item()

    def print(self):
        print('benchmark:')
        print(f'           n: {self.n}')
        print(f'  torch time: {self.torch_time}')
        print(f'    lib time: {self.lib_time}')


class Experiment:

    def __init__(self, lib, a_shape, b_shape):
        self.lib = lib
        self.a_shape = a_shape
        self.b_shape = b_shape

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

    def benchmark(self, n=1):
        a = torch.rand(*self.a_shape).float()
        b = torch.rand(*self.b_shape).float()
        torch_times = self._call_benchmark(n, func=self._call_torch, args=(a, b))
        lib_times = self._call_benchmark(n, func=self._call_lib, args=(a, b))
        res = BenchmarkResult(n=n, torch_times=torch_times, lib_times=lib_times)
        return res

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
    
    def _call_benchmark(self, n, func, args):
        func(*args)  # warmup
        times = []
        for _ in range(n):
            start = time.perf_counter()
            func(*args)
            end = time.perf_counter()
            rt = end - start
            times.append(rt)
        return times
