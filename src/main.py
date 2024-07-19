import click

from .binding import LibMatmul
from .experiment import Experiment


@click.group
def main():
    pass


@main.command()
@click.argument('a')
@click.argument('b')
def compare(a, b):
    a_shape = _parse_shape(a)
    b_shape = _parse_shape(b)
    lib = LibMatmul.load()
    exp = Experiment(lib=lib, a_shape=a_shape, b_shape=b_shape)
    exp.compare()


@main.command()
@click.argument('a')
@click.argument('b')
@click.option('--n', type=int, default=1)
def benchmark(a, b, n=1):
    a_shape = _parse_shape(a)
    b_shape = _parse_shape(b)
    lib = LibMatmul.load()
    exp = Experiment(lib=lib, a_shape=a_shape, b_shape=b_shape)
    res = exp.benchmark(n=n)
    res.print()


def _parse_shape(v):
    return tuple(map(int, v.strip().split('x')))


if __name__ == '__main__':
    main()
