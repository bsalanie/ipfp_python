"""
Estimation of the `Choo and Siow 2006 <https://www.jstor.org/stable/10.1086/498585?seq=1>`_ model:
in its original version (homoskedastic with singles).

We minimize the :math:`F(u,v,\\lambda)-\\hat{\\mu}\\cdot \\Phi^{\\lambda}` function of Galichon--Salanie (2020, Proposition 5.)

"""
import numpy as np
from math import sqrt
import sys

from typing import Tuple, List
from dataclasses import dataclass

import scipy.optimize as spopt

from ipfp_utils import print_stars
from ipfp_solvers import ipfp_homo_solver


def f_objgrad(pars: np.ndarray, args: List) \
        -> Tuple[float, np.ndarray]:
    """

    """
    muxy, nx, my, bases, gr = args
    ncat_men, _, n_bases = bases.shape
    npars = pars.size

    u, v, l = pars[:ncat_men], pars[ncat_men:-n_bases], \
        pars[-n_bases:]

    f_val = 0.0
    f_grad = np.zeros(npars)

    expmu, expmv = np.exp(-u), np.exp(-v)
    t_u = u + expmu - 1.0
    t_v = v + expmv - 1.0
    Phi_l = bases @ l

    t_xy = np.exp((Phi_l - np.add.outer(u, v))/2.0)
    s_xy = np.sqrt(np.outer(nx, my))
    ts_xy = t_xy*s_xy
    sum_xy = np.sum(ts_xy)

    f_val = np.dot(nx, t_u) + np.dot(my, t_v) \
        + 2.0*sum_xy
    f_val -= np.sum(muxy*Phi_l)

    if gr:
        f_grad[:ncat_men] = nx*(1.0 - expmu) - np.sum(ts_xy, 1)
        f_grad[ncat_men:-n_bases] = my*(1.0 - expmv) - np.sum(ts_xy, 0)
        f_grad[-n_bases:] =  \
            np.einsum('xy,xyl->l', ts_xy - muxy, bases)

    return f_val, f_grad


def estimate_cs_fuvl(muxy: np.ndarray, nx: np.ndarray,
                     my: np.ndarray, bases: np.ndarray) -> spopt.OptimizeResult:
    n_bases = bases.shape[2]
    l_init = np.random.normal(size=n_bases)
    mux0 = nx - np.sum(muxy, 1)
    mu0y = my - np.sum(muxy, 0)
    p_init = np.concatenate((-np.log(mux0/nx),
                             -np.log(mu0y/my), l_init))
    resus = spopt.minimize(f_objgrad, p_init,
                           args=[muxy, nx, my, bases, True],
                           jac=True)  # , options={'disp': True})
    return resus


# for testing purposes

if __name__ == "__main__":
    ncat_men = 3
    ncat_women = 2
    n_bases = 2
    nx = np.array([3, 4, 8])
    my = np.array([4, 6])
    muxy_obs = np.array([[1, 1], [1, 2], [1, 1]])
    bases = np.zeros((ncat_men, ncat_women, n_bases))
    bases[:, :, 0] = 1.0
    bases[:, :, 1] = np.subtract.outer(nx, my)
    l0 = np.array([1.0, -1.0])

    mux0_obs = nx - np.sum(muxy_obs, 1)
    mu0y_obs = my - np.sum(muxy_obs, 0)
    u0 = -np.log(mux0_obs/nx)
    v0 = -np.log(mu0y_obs/my)

    p0 = np.concatenate((u0, v0, l0))

    # checking the gradient
    f0, f_grad = f_objgrad(p0, [muxy_obs, nx, my, bases, True])

    g = np.zeros_like(p0)

    EPS = 1e-6
    for i, p in enumerate(p0):
        p1 = p0.copy()
        p1[i] = p + EPS
        f1, _ = f_objgrad(p1, [muxy_obs, nx, my, bases, False])
        g[i] = (f1-f0)/EPS

    print_stars("checking the gradient: analytic, numeric")
    print(np.column_stack((f_grad, g)))

    # now we simulate the model
    Phi0 = bases @ l0
    (muxy, mux0, mu0y), marg_err_x, marg_err_y \
        = ipfp_homo_solver(Phi0, nx, my)

    # and we estimate it
    resus = estimate_cs_fuvl(muxy, nx, my, bases)

    print("Results of minimization")
    u = resus.x[:ncat_men]
    v = resus.x[ncat_men:-n_bases]
    l = resus.x[-n_bases:]
    print_stars("u vs -log(mu(0|x))")
    print(np.column_stack((u,
                           -np.log(1.0-np.sum(muxy, 1)/nx))))
    print_stars("v vs -log(mu(0|y))")
    print(np.column_stack((v,
                           -np.log(1.0-np.sum(muxy, 0)/my))))
    print_stars("l vs l0")
    print(np.column_stack((l, l0)))
