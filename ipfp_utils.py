"""
some utility programs used by ipfp_solvers
"""

import numpy as np
import scipy.stats as sts
from typing import Union
from math import sqrt, exp, log
import sys


def print_stars(title: str = None, n: int = 70) -> None:
    """
    prints a starred line, or two around the title

    :param str title:  title

    :param int n: number of stars on line

    :return: nothing
    """
    line_stars = '*' * n
    print()
    print(line_stars)
    if title:
        print(title.center(n))
        print(line_stars)
    print()


def describe_array(v: np.array, name: str = "v"):
    """
    descriptive statistics on an array interpreted as a vector

    :param np.array v: the array

    :param str name: its name

    :return: the `scipy.stats.describe` object
    """
    print_stars(f"{name} has:")
    d = sts.describe(v, None)
    print(f"Number of elements: {d.nobs}")
    print(f"Minimum: {d.minmax[0]}")
    print(f"Maximum: {d.minmax[1]}")
    print(f"Mean: {d.mean}")
    print(f"Stderr: {sqrt(d.variance)}")
    return d


def nprepeat_col(v: np.array, n: int) -> np.array:
    """
    create a matrix with `n` columns equal to `v`

    :param  np.array v: a 1-dim array of size `m`

    :param int n: number of columns requested

    :return: a 2-dim array of shape `(m, n)`
    """
    return np.repeat(v[:, np.newaxis], n, axis=1)


def nprepeat_row(v: np.array, m: int) -> np.array:
    """
    create a matrix with `m` rows equal to `v`

    :param np.array v: a 1-dim array of size `n`

    :param int m: number of rows requested

    :return: a 2-dim array of shape `(m, n)`
    """
    return np.repeat(v[np.newaxis, :], m, axis=0)


def npmaxabs(arr: np.array) -> float:
    """
    maximum absolute value in an array

    :param np.array arr: Numpy array

    :return: a float
    """
    return np.max(np.abs(arr))


def nplog(arr: np.array, eps: float = 1e-30, verbose: bool = False) -> np.array:
    """
    :math:`C^2` extension of  :math:`\\ln(a)` below `eps`

    :param np.array arr: a Numpy array

    :param float eps: lower bound

    :return:  :math:`\\ln(a)` :math:`C^2`-extended below `eps`
    """
    if np.min(arr) > eps:
        return np.log(arr)
    else:
        logarreps = np.log(np.maximum(arr, eps))
        logarr_smaller = log(eps) - (eps - arr) * \
            (3.0 * eps - arr) / (2.0 * eps * eps)
        if verbose:
            n_small_args = np.sum(arr < eps)
            if n_small_args > 0:
                finals = 's' if n_small_args > 1 else ''
                print(
                    f"nplog: {n_small_args} argument{finals} smaller than {eps}: mini = {np.min(arr)}")
        return np.where(arr > eps, logarreps, logarr_smaller)


def der_nplog(arr: np.array, eps: float = 1e-30, verbose: bool = False) -> np.array:
    """
    derivative of :math:`C^2` extension of  :math:`\\ln(a)` below `eps`

    :param np.array arr: a Numpy array

    :param float eps: lower bound

    :return: derivative of  :math:`\\ln(a)` :math:`C^2`-extended below `eps`
    """
    if np.min(arr) > eps:
        return 1.0 / arr
    else:
        der_logarreps = 1.0 / np.maximum(arr, eps)
        der_logarr_smaller = (2.0 * eps - arr) / (eps * eps)
        if verbose:
            n_small_args = np.sum(arr < eps)
            if n_small_args > 0:
                finals = 's' if n_small_args > 1 else ''
                print(
                    f"der_nplog: {n_small_args} argument{finals} smaller than {eps}: mini = {np.min(arr)}")
        return np.where(arr > eps, der_logarreps, der_logarr_smaller)


def npexp(arr: np.array, bigx: float = 30.0, verbose: bool = False) -> np.array:
    """
    :math:`C^2` extension of  :math:`\\exp(a)` above `bigx`

    :param np.array arr: a Numpy array

    :param float bigx: upper bound

    :return:   :math:`\\exp(a)`  :math:`C^2`-extended above `bigx`
    """
    if np.max(arr) < bigx:
        return np.exp(arr)
    else:
        exparr = np.exp(np.minimum(arr, bigx))
        ebigx = exp(bigx)
        darr = arr - bigx
        exparr_larger = ebigx * (1.0 + darr * (1.0 + 0.5 * darr))
        if verbose:
            n_large_args = np.sum(arr > bigx)
            if n_large_args > 0:
                finals = 's' if n_large_args > 1 else ''
                print(
                    f"npexp: {n_large_args} argument{finals} larger than {bigx}: maxi = {np.max(arr)}")
        return np.where(arr < bigx, exparr, exparr_larger)


def der_npexp(arr: np.array, bigx: float = 30.0, verbose: bool = False) -> np.array:
    """
    derivative of :math:`C^2` extension of  :math:`\\exp(a)` above `bigx`

    :param np.array arr: a Numpy array

    :param float bigx: upper bound

    :return: derivative of  :math:`\\exp(a)`  :math:`C^2`-extended above `bigx`
    """
    if np.max(arr) < bigx:
        return np.exp(arr)
    else:
        der_exparr = np.exp(np.minimum(arr, bigx))
        ebigx = exp(bigx)
        darr = arr - bigx
        der_exparr_larger = ebigx * (1.0 + darr)
        if verbose:
            n_large_args = np.sum(arr > bigx)
            if n_large_args > 0:
                finals = 's' if n_large_args > 1 else ''
                print(
                    f"der_npexp: {n_large_args} argument{finals} larger than {bigx}: maxi = {np.max(arr)}")
        return np.where(arr < bigx, der_exparr, der_exparr_larger)


def nppow(a: np.array, b: Union[int, float, np.array]) -> np.array:
    """
    evaluates a**b element-by-element

    :param np.array a: 

    :param Union[int, float, np.array] b: if an array, 
       should have the same shape as `a`

    :return: an array of the same shape as `a`
    """
    mina = np.min(a)
    if mina <= 0:
        print_stars("All elements of a must be positive in nppow!")
        sys.exit(1)

    if isinstance(b, (int, float)):
        return a**b
    else:
        if a.shape != b.shape:
            print_stars(
                "nppow: b is not a number or an array of the same shape as a!")
            sys.exit(1)
        avec = a.ravel()
        bvec = b.ravel()
        a_pow_b = avec**bvec
        return a_pow_b.reshape(a.shape)


def der_nppow(a: np.array, b: Union[int, float, np.array]) -> np.array:
    """
    evaluates the derivatives in a and b of element-by-element a**b 

    :param np.array a:

    :param Union[int, float, np.array] b: if an array, 
       should have the same shape as `a`

    :return: a pair of two arrays of the same shape as `a`
    """

    mina = np.min(a)
    if mina <= 0:
        print_stars("All elements of a must be positive in der_nppow!")
        sys.exit(1)

    if isinstance(b, (int, float)):
        a_pow_b = a ** b
        return (b*a_pow_b/a, a_pow_b*log(a))
    else:
        if a.shape != b.shape:
            print_stars(
                "nppow: b is not a number or an array of the same shape as a!")
            sys.exit(1)
        avec = a.ravel()
        bvec = b.ravel()
        a_pow_b = avec**bvec
        der_wrt_a = a_pow_b*bvec/avec
        der_wrt_b = a_pow_b*nplog(avec)
        return (der_wrt_a.reshape(a.shape), der_wrt_b.reshape(a.shape))


if __name__ == "__main__":
    """ run some tests """

    arr = np.array([-0.001, 0.0, 1e-30, 0.001])
    print(f"x = {arr}")
    print_stars("numpy extended log(x)")
    print(f"   value = {nplog(arr)}")
    print(f"   derivative = {der_nplog(arr)}")

    exponents = np.array([[0.5, 1.0], [1.0, 1.5]])
    eps = 1e-6
    args_pow = np.array([[2.0, 1.0], [3.0, 4.0]])
    print(f"\n\nextended powers")
    print(exponents)
    print(" of")
    print(args_pow)
    print(nppow(args_pow, exponents))
    print(f"\nits first derivatives:")
    print(" should be:")
    print((nppow(args_pow + eps, exponents) -
           nppow(args_pow - eps, exponents)) / (2.0 * eps))
    print(" and")
    print((nppow(args_pow, exponents + eps) -
           nppow(args_pow, exponents - eps)) / (2.0 * eps))
    print("       they are:")
    d1 = der_nppow(args_pow, exponents)
    print(d1[0])
    print("  and")
    print(d1[1])

    arr = np.array(np.arange(6)).reshape((2, 3))
    describe_array(arr, "arr")

    args_exp = np.array([[10.0, 30.0], [32.0, -1.0]])
    print(f"\n\nextended exponential of {args_exp}:")
    print(npexp(args_exp))
    print(f"\nits first derivative should be:")
    print((npexp(args_exp + eps) - npexp(args_exp - eps)) / (2.0 * eps))
    print("       it is:")
    d_exp = der_npexp(args_exp)
    print(d_exp)

    print_stars("Testing rows and columns repeats")
    v = np.arange(3)
    vm = nprepeat_row(v, 2)
    vn = nprepeat_col(v, 4)
    print("v=:")
    print(v)
    print("2 rows of v:")
    print(vm)
    print("4 columns of v:")
    print(vn)
