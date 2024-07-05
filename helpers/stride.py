class Tensor:

    def __init__(self, buffer):
        self.buffer = buffer

    def view(self, shape):
        stride = [None] * len(shape)
        

        pass


def main():
    buf = list(range(1, 7))
    t = Tensor(buf)
    print(t.buffer)


main()

"""
[1 2 3 4 5 6]

3x2
[
    [1 2]
    [3 4]
    [5 6]
]

2x3
[
    [1 2 3]
    [4 5 6]
]
"""