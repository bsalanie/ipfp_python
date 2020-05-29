"""
Implementations of the IPFP algorithm to solve for equilibrium and do comparative statics
in several variants of the `Choo and Siow 2006 <https://www.jstor.org/stable/10.1086/498585?seq=1>`_ model:

 * homoskedastic with singles (as in CS 2006)
 * homoskedastic without singles
 * gender-heteroskedastic: with a scale parameter on the error term for women
 * gender- and type-heteroskedastic: with a scale parameter on the error term for women

each solver, when fed the joint surplus and margins,
returns the equilibrium matching patterns, the adding-up errors on the margins,
and if requested (gr=True) the derivatives of the matching patterns in all primitives.



"""
import numpy as np
from math import sqrt
import sys

import scipy.linalg as spla

from ipfp_utils import print_stars, npexp, der_npexp, npmaxabs, \
    nplog, nppow, der_nppow, nprepeat_col, nprepeat_row, describe_array


def ipfp_homo_nosingles_solver(Phi, men_margins, women_margins,
                               tol=1e-9, gr=False, verbose=False,
                               maxiter=1000):
    """
    solve for equilibrium in a Choo and Siow market without singles

    given systematic surplus and margins

    :param np.array Phi: matrix of systematic surplus, shape (ncat_men, ncat_women)

    :param np.array men_margins: vector of men margins, shape (ncat_men)

    :param np.array women_margins: vector of women margins, shape (ncat_women)

    :param float tol: tolerance on change in solution

    :param boolean gr: if True, also evaluate derivatives of muxy wrt Phi

    :param boolean verbose: prints stuff

    :param int maxiter: maximum number of iterations

    :return: muxy, marg_err_x, marg_err_y
             and gradients of muxy wrt Phi if gr=True
    """
    ncat_men = men_margins.shape[0]
    ncat_women = women_margins.shape[0]
    n_couples = np.sum(men_margins)


    # check that there are as many men as women
    if np.abs(np.sum(women_margins) - n_couples) > n_couples * tol:
        print_stars(f"{ipfp_homo_nosingles_solver}: there should be as many men as women")

    if Phi.shape != (ncat_men, ncat_women):
        print_stars(
            f"ipfp_hetero_solver: the shape of Phi should be ({ncat_men}, {ncat_women}")
        sys.exit(1)

    ephi2 = npexp(Phi / 2.0)
    ephi2T = ephi2.T

    #############################################################################
    # we solve the equilibrium equations muxy = ephi2 * tx * ty
    #   starting with a reasonable initial point for tx and ty: : tx = ty = bigc
    #   it is important that it fit the number of individuals
    #############################################################################
    bigc = sqrt(n_couples / np.sum(ephi2))
    txi = np.full(ncat_men, bigc)
    tyi = np.full(ncat_women, bigc)

    err_diff = bigc
    tol_diff = tol * err_diff
    niter = 0
    while (err_diff > tol_diff) and (niter < maxiter):
        sx = ephi2 @ tyi
        tx = men_margins / sx
        sy = ephi2T @ tx
        ty = women_margins / sy
        err_x = npmaxabs(tx - txi)
        err_y = npmaxabs(ty - tyi)
        err_diff = err_x + err_y
        txi, tyi = tx, ty
        niter += 1
    muxy = ephi2 * np.outer(txi, tyi)
    marg_err_x = np.sum(muxy, 1) - men_margins
    marg_err_y = np.sum(muxy, 0) - women_margins
    if verbose:
        print(f"After {niter} iterations:")
        print(f"\tMargin error on x: {npmaxabs(marg_err_x)}")
        print(f"\tMargin error on y: {npmaxabs(marg_err_y)}")
    if not gr:
        return muxy, marg_err_x, marg_err_y
    else:
        sxi = ephi2 @ tyi
        syi = ephi2T @ txi
        n_sum_categories = ncat_men + ncat_women
        n_prod_categories = ncat_men * ncat_women

        # start with the LHS of the linear system
        lhs = np.zeros((n_sum_categories, n_sum_categories))
        lhs[:ncat_men, :ncat_men] = np.diag(sxi)
        lhs[:ncat_men, ncat_men:] = ephi2 * txi.reshape((-1, 1))
        lhs[ncat_men:, ncat_men:] = np.diag(syi)
        lhs[ncat_men:, :ncat_men] = ephi2T * tyi.reshape((-1, 1))

        # now fill the RHS
        n_cols_rhs = n_prod_categories
        rhs = np.zeros((n_sum_categories, n_cols_rhs))

        #  to compute derivatives of (txi, tyi) wrt Phi
        der_ephi2 = der_npexp(Phi / 2.0) / \
                    (2.0 * ephi2)  # 1/2 with safeguards
        ivar = 0
        for iman in range(ncat_men):
            rhs[iman, ivar:(ivar + ncat_women)] = - \
                                                      muxy[iman, :] * der_ephi2[iman, :]
            ivar += ncat_women
        ivar1 = ncat_men
        ivar2 = 0
        for iwoman in range(ncat_women):
            rhs[ivar1, ivar2:n_cols_rhs:ncat_women] = - \
                                                          muxy[:, iwoman] * der_ephi2[:, iwoman]
            ivar1 += 1
            ivar2 += 1
        # solve for the derivatives of txi and tyi
        dt_dT = spla.solve(lhs, rhs)
        dt = dt_dT[:ncat_men, :]
        dT = dt_dT[ncat_men:, :]
        # now construct the derivatives of muxy
        dmuxy = np.zeros((n_prod_categories, n_cols_rhs))
        ivar = 0
        for iman in range(ncat_men):
            dt_man = dt[iman, :]
            dmuxy[ivar:(ivar + ncat_women),
            :] = np.outer((ephi2[iman, :] * tyi), dt_man)
            ivar += ncat_women
        for iwoman in range(ncat_women):
            dT_woman = dT[iwoman, :]
            dmuxy[iwoman:n_prod_categories:ncat_women,
            :] += np.outer((ephi2[:, iwoman] * txi), dT_woman)
        # add the term that comes from differentiating ephi2
        muxy_vec2 = (muxy * der_ephi2).reshape(n_prod_categories)
        dmuxy += np.diag(muxy_vec2)
        return muxy, marg_err_x, marg_err_y, dmuxy


def ipfp_homo_solver(Phi, men_margins, women_margins, tol=1e-9,
                     gr=False, verbose=False, maxiter=1000):
    """
    solve for equilibrium in a Choo and Siow market

    given systematic surplus and margins

    :param np.array Phi: matrix of systematic surplus, shape (ncat_men, ncat_women)

    :param np.array men_margins: vector of men margins, shape (ncat_men)

    :param np.array women_margins: vector of women margins, shape (ncat_women)

    :param float tol: tolerance on change in solution

    :param boolean gr: if True, also evaluate derivatives of muxy wrt Phi

    :param boolean verbose: prints stuff

    :param int maxiter: maximum number of iterations

    :return: (muxy, mux0, mu0y), errors on margins marg_err_x, marg_err_y,
             and gradients of (muxy, mux0, mu0y)
             wrt (men_margins, women_margins, Phi) if gr=True
    """

    ncat_men = men_margins.size
    ncat_women = women_margins.size
    if Phi.shape != (ncat_men, ncat_women):
        print_stars(
            f"ipfp_homo_solver: the shape of Phi should be ({ncat_men}, {ncat_women}")
        sys.exit(1)

    ephi2 = npexp(Phi / 2.0)

    #############################################################################
    # we solve the equilibrium equations muxy = ephi2 * tx * ty
    #   where mux0=tx**2  and mu0y=ty**2
    #   starting with a reasonable initial point for tx and ty: tx = ty = bigc
    #   it is important that it fit the number of individuals
    #############################################################################

    ephi2T = ephi2.T
    nindivs = np.sum(men_margins) + np.sum(women_margins)
    bigc = sqrt(nindivs / (ncat_men + ncat_women + 2.0 * np.sum(ephi2)))
    txi = np.full(ncat_men, bigc)
    tyi = np.full(ncat_women, bigc)

    err_diff = bigc
    tol_diff = tol * bigc
    niter = 0
    while (err_diff > tol_diff) and (niter < maxiter):
        sx = ephi2 @ tyi
        tx = (np.sqrt(sx * sx + 4.0 * men_margins) - sx) / 2.0
        sy = ephi2T @ tx
        ty = (np.sqrt(sy * sy + 4.0 * women_margins) - sy) / 2.0
        err_x = npmaxabs(tx - txi)
        err_y = npmaxabs(ty - tyi)
        err_diff = err_x + err_y
        txi = tx
        tyi = ty
        niter += 1
    mux0 = txi * txi
    mu0y = tyi * tyi
    muxy = ephi2 * np.outer(txi, tyi)
    marg_err_x = mux0 + np.sum(muxy, 1) - men_margins
    marg_err_y = mu0y + np.sum(muxy, 0) - women_margins
    if verbose:
        print(f"After {niter} iterations:")
        print(f"\tMargin error on x: {npmaxabs(marg_err_x)}")
        print(f"\tMargin error on y: {npmaxabs(marg_err_y)}")
    if not gr:
        return (muxy, mux0, mu0y), marg_err_x, marg_err_y
    else:  # we compute the derivatives
        sxi = ephi2 @ tyi
        syi = ephi2T @ txi
        n_sum_categories = ncat_men + ncat_women
        n_prod_categories = ncat_men * ncat_women
        # start with the LHS of the linear system
        lhs = np.zeros((n_sum_categories, n_sum_categories))
        lhs[:ncat_men, :ncat_men] = np.diag(2.0 * txi + sxi)
        lhs[:ncat_men, ncat_men:] = ephi2 * txi.reshape((-1, 1))
        lhs[ncat_men:, ncat_men:] = np.diag(2.0 * tyi + syi)
        lhs[ncat_men:, :ncat_men] = ephi2T * tyi.reshape((-1, 1))
        # now fill the RHS
        n_cols_rhs = n_sum_categories + n_prod_categories
        rhs = np.zeros((n_sum_categories, n_cols_rhs))
        #  to compute derivatives of (txi, tyi) wrt men_margins
        rhs[:ncat_men, :ncat_men] = np.eye(ncat_men)
        #  to compute derivatives of (txi, tyi) wrt women_margins
        rhs[ncat_men:n_sum_categories,
        ncat_men:n_sum_categories] = np.eye(ncat_women)
        #  to compute derivatives of (txi, tyi) wrt Phi
        der_ephi2 = der_npexp(Phi / 2.0) / \
                    (2.0 * ephi2)  # 1/2 with safeguards
        ivar = n_sum_categories
        for iman in range(ncat_men):
            rhs[iman, ivar:(ivar + ncat_women)] = - \
                                                      muxy[iman, :] * der_ephi2[iman, :]
            ivar += ncat_women
        ivar1 = ncat_men
        ivar2 = n_sum_categories
        for iwoman in range(ncat_women):
            rhs[ivar1, ivar2:n_cols_rhs:ncat_women] = - \
                                                          muxy[:, iwoman] * der_ephi2[:, iwoman]
            ivar1 += 1
            ivar2 += 1
        # solve for the derivatives of txi and tyi
        dt_dT = spla.solve(lhs, rhs)
        dt = dt_dT[:ncat_men, :]
        dT = dt_dT[ncat_men:, :]
        # now construct the derivatives of the mus
        dmux0 = 2.0 * (dt * txi.reshape((-1, 1)))
        dmu0y = 2.0 * (dT * tyi.reshape((-1, 1)))
        dmuxy = np.zeros((n_prod_categories, n_cols_rhs))
        ivar = 0
        for iman in range(ncat_men):
            dt_man = dt[iman, :]
            dmuxy[ivar:(ivar + ncat_women), :] = np.outer((ephi2[iman, :] * tyi), dt_man)
            ivar += ncat_women
        for iwoman in range(ncat_women):
            dT_woman = dT[iwoman, :]
            dmuxy[iwoman:n_prod_categories:ncat_women, :] += np.outer((ephi2[:, iwoman] * txi), dT_woman)
        # add the term that comes from differentiating ephi2
        muxy_vec2 = (muxy * der_ephi2).reshape(n_prod_categories)
        dmuxy[:, n_sum_categories:] += np.diag(muxy_vec2)
        return (muxy, mux0, mu0y), marg_err_x, marg_err_y, (dmuxy, dmux0, dmu0y)


def ipfp_hetero_solver(Phi, men_margins, women_margins, tau, tol=1e-9,
                       gr=False, verbose=False, maxiter=1000):
    """
    solve for equilibrium in a  in a gender-heteroskedastic Choo and Siow market

    given systematic surplus and margins and a scale parameter dist_params[0]

    :param np.array Phi: matrix of systematic surplus, shape (ncat_men, ncat_women)

    :param np.array men_margins: vector of men margins, shape (ncat_men)

    :param np.array women_margins: vector of women margins, shape (ncat_women)

    :param float tau: a positive scale parameter for the error term on women

    :param float tol: tolerance on change in solution

    :param boolean gr: if True, also evaluate derivatives of muxy wrt Phi

    :param boolean verbose: prints stuff

    :param int maxiter: maximum number of iterations

    :param np.array dist_params: array of one positive number (the scale parameter for women)

    :return: (muxy, mux0, mu0y), errors on margins marg_err_x, marg_err_y,
             and gradients of (muxy, mux0, mu0y)
             wrt (men_margins, women_margins, Phi, dist_params[0]) if gr=True
    """

    ncat_men = men_margins.shape[0]
    ncat_women = women_margins.shape[0]
    if Phi.shape != (ncat_men, ncat_women):
        print_stars(
            f"ipfp_hetero_solver: the shape of Phi should be ({ncat_men}, {ncat_women}")
        sys.exit(1)

    if tau <= 0:
        print_stars("ipfp_hetero_solver needs a positive tau")
        sys.exit(1)

    #############################################################################
    # we use ipfp_heteroxy_solver with sigma_x = 1 and tau_y = tau
    #############################################################################

    sigma_x = np.ones(ncat_men)
    tau_y = np.full(ncat_women, tau)

    if gr:
        mus, marg_err_x, marg_err_y, dmus_hxy = \
            ipfp_heteroxy_solver(Phi, men_margins, women_margins,
                                 sigma_x, tau_y, tol=tol, gr=True,
                                 maxiter=maxiter, verbose=verbose)
        dmus_xy, dmus_x0, dmus_0y = dmus_hxy
        n_sum_categories = ncat_men + ncat_women
        n_prod_categories = ncat_men * ncat_women
        n_cols = n_sum_categories + n_prod_categories
        itau_y = n_cols + ncat_men
        dmuxy = np.zeros((n_prod_categories, n_cols + 1))
        dmuxy[:, :n_cols] = dmus_xy[:, :n_cols]
        dmuxy[:, -1] = np.sum(dmus_xy[:, itau_y:], 1)
        dmux0 = np.zeros((ncat_men, n_cols + 1))
        dmux0[:, :n_cols] = dmus_x0[:, :n_cols]
        dmux0[:, -1] = np.sum(dmus_x0[:, itau_y:], 1)
        dmu0y = np.zeros((ncat_women, n_cols + 1))
        dmu0y[:, :n_cols] = dmus_0y[:, :n_cols]
        dmu0y[:, -1] = np.sum(dmus_0y[:, itau_y:], 1)
        return (muxy, mux0, mu0y), marg_err_x, marg_err_y, (dmuxy, dmux0, dmu0y)

    else:
        return ipfp_heteroxy_solver(Phi, men_margins, women_margins,
                                    sigma_x, tau_y, tol=tol, gr=False,
                                    maxiter=maxiter, verbose=verbose)



def ipfp_heteroxy_solver(Phi, men_margins, women_margins,
                         sigma_x, tau_y, tol=1e-9,
                         gr=False, maxiter=1000, verbose=False):
    """
    solve for equilibrium in a  in a gender- and type-heteroskedastic Choo and Siow market

    given systematic surplus and margins and a scale parameter dist_params[0]

    :param np.array Phi: matrix of systematic surplus, shape (ncat_men, ncat_women)

    :param np.array men_margins: vector of men margins, shape (ncat_men)

    :param np.array women_margins: vector of women margins, shape (ncat_women)

    :param np.array sigma_x: an array of positive numbers of shape (ncat_men)

    :param np.array tau_y: an array of positive numbers of shape (ncat_women)

    :param float tol: tolerance on change in solution

    :param boolean gr: if True, also evaluate derivatives of muxy wrt Phi

    :param boolean verbose: prints stuff

    :param int maxiter: maximum number of iterations

    :return: (muxy, mux0, mu0y), errors on margins marg_err_x, marg_err_y,
              and gradients of (muxy, mux0, mu0y)
              wrt (men_margins, women_margins, Phi, dist_params) if gr=True
    """

    ncat_men, ncat_women = men_margins.size, women_margins.size
    if Phi.shape != (ncat_men, ncat_women):
        print_stars(
            f"ipfp_heteroxy_solver: the shape of Phi should be ({ncat_men}, {ncat_women}")
        sys.exit(1)

    if np.min(sigma_x) <= 0.0:
        print_stars(
            "ipfp_heteroxy_solver: all elements of sigma_x must be positive")
        sys.exit(1)
    if np.min(tau_y) <= 0.0:
        print_stars(
            "ipfp_heteroxy_solver: all elements of tau_y must be positive")
        sys.exit(1)
    sumxy1 = 1.0 / np.add.outer(sigma_x, tau_y)
    ephi2 = npexp(Phi * sumxy1)

    #############################################################################
    # we solve the equilibrium equations muxy = ephi2 * tx * ty
    #   with tx = mux0^(sigma_x/(sigma_x + tau_max))
    #   and ty = mu0y^(tau_y/(sigma_max + tau_y))
    #   starting with a reasonable initial point for tx and ty: tx = ty = bigc
    #   it is important that it fit the number of individuals
    #############################################################################

    nindivs = np.sum(men_margins) + np.sum(women_margins)
    bigc = nindivs / (ncat_men + ncat_women + 2.0 * np.sum(ephi2))
    # we find the largest values of sigma_x and tau_y
    xmax = np.argmax(sigma_x)
    sigma_max = sigma_x[xmax]
    ymax = np.argmax(tau_y)
    tau_max = tau_y[ymax]
    # we use tx = mux0^(sigma_x/(sigma_x + tau_max))
    #    and ty = mu0y^(tau_y/(sigma_max + tau_y))
    sig_taumax = sigma_x + tau_max
    txi = np.power(bigc, sigma_x / sig_taumax)
    sigmax_tau = tau_y + sigma_max
    tyi = np.power(bigc, tau_y / sigmax_tau)
    err_diff = bigc
    tol_diff = tol * bigc
    tol_newton = tol
    niter = 0
    while (err_diff > tol_diff) and (niter < maxiter):
        # Newton iterates for men
        err_newton = bigc
        txin = txi.copy()
        mu0y_in = np.power(np.power(tyi, sigmax_tau), 1.0 / tau_y)
        while err_newton > tol_newton:
            txit = np.power(txin, sig_taumax)
            mux0_in = np.power(txit, 1.0 / sigma_x)
            out_xy = np.outer(np.power(mux0_in, sigma_x),
                              np.power(mu0y_in, tau_y))
            muxy_in = ephi2 * np.power(out_xy, sumxy1)
            errxi = mux0_in + np.sum(muxy_in, 1) - men_margins
            err_newton = npmaxabs(errxi)
            txin -= errxi / (sig_taumax * (mux0_in / sigma_x
                                           + np.sum(sumxy1 * muxy_in, 1)) / txin)
        tx = txin

        # Newton iterates for women
        err_newton = bigc
        tyin = tyi.copy()
        mux0_in = np.power(np.power(tx, sig_taumax), 1.0 / sigma_x)
        while err_newton > tol_newton:
            tyit = np.power(tyin, sigmax_tau)
            mu0y_in = np.power(tyit, 1.0 / tau_y)
            out_xy = np.outer(np.power(mux0_in, sigma_x),
                              np.power(mu0y_in, tau_y))
            muxy_in = ephi2 * np.power(out_xy, sumxy1)
            erryi = mu0y_in + np.sum(muxy_in, 0) - women_margins
            err_newton = npmaxabs(erryi)
            tyin -= erryi / (sigmax_tau * (mu0y_in / tau_y
                                           + np.sum(sumxy1 * muxy_in, 0)) / tyin)

        ty = tyin

        err_x = npmaxabs(tx - txi)
        err_y = npmaxabs(ty - tyi)
        err_diff = err_x + err_y

        txi = tx
        tyi = ty

        niter += 1

    mux0 = mux0_in
    mu0y = mu0y_in
    muxy = muxy_in
    marg_err_x = mux0 + np.sum(muxy, 1) - men_margins
    marg_err_y = mu0y + np.sum(muxy, 0) - women_margins

    if verbose:
        print(f"After {niter} iterations:")
        print(f"\tMargin error on x: {npmaxabs(marg_err_x)}")
        print(f"\tMargin error on y: {npmaxabs(marg_err_y)}")
    if not gr:
        return (muxy, mux0, mu0y), marg_err_x, marg_err_y
    else:  # we compute the derivatives
        n_sum_categories = ncat_men + ncat_women
        n_prod_categories = ncat_men * ncat_women
        # we work directly with (mux0, mu0y)
        sigrat_xy = sumxy1 * sigma_x.reshape((-1, 1))
        taurat_xy = 1.0 - sigrat_xy
        mux0_mat = nprepeat_col(mux0, ncat_women)
        mu0y_mat = nprepeat_row(mu0y, ncat_men)
        # muxy = axy * bxy * ephi2
        axy = nppow(mux0_mat, sigrat_xy)
        bxy = nppow(mu0y_mat, taurat_xy)
        der_axy1, der_axy2 = der_nppow(mux0_mat, sigrat_xy)
        der_bxy1, der_bxy2 = der_nppow(mu0y_mat, taurat_xy)
        der_axy1_rat, der_axy2_rat = der_axy1 / axy, der_axy2 / axy
        der_bxy1_rat, der_bxy2_rat = der_bxy1 / bxy, der_bxy2 / bxy

        # start with the LHS of the linear system on (dmux0, dmu0y)
        lhs = np.zeros((n_sum_categories, n_sum_categories))
        lhs[:ncat_men, :ncat_men] = np.diag(
            1.0 + np.sum(muxy * der_axy1_rat, 1))
        lhs[:ncat_men, ncat_men:] = muxy * der_bxy1_rat
        lhs[ncat_men:, ncat_men:] = np.diag(
            1.0 + np.sum(muxy * der_bxy1_rat, 0))
        lhs[ncat_men:, :ncat_men] = (muxy * der_axy1_rat).T

        # now fill the RHS (derivatives wrt men_margins, then men_margins,
        #    then Phi, then sigma_x and tau_y)
        n_cols_rhs = n_sum_categories + n_prod_categories + ncat_men + ncat_women
        rhs = np.zeros((n_sum_categories, n_cols_rhs))

        #  to compute derivatives of (mux0, mu0y) wrt men_margins
        rhs[:ncat_men, :ncat_men] = np.eye(ncat_men)
        #  to compute derivatives of (mux0, mu0y) wrt women_margins
        rhs[ncat_men:,
        ncat_men:n_sum_categories] = np.eye(ncat_women)

        #   the next line is sumxy1 with safeguards
        sumxy1_safe = sumxy1 * der_npexp(Phi * sumxy1) / ephi2

        big_a = muxy * sumxy1_safe
        big_b = der_axy2_rat - der_bxy2_rat
        b_mu_s = big_b * muxy * sumxy1
        a_phi = Phi * big_a
        big_c = sumxy1 * (a_phi - b_mu_s * tau_y)
        big_d = sumxy1 * (a_phi + b_mu_s * sigma_x.reshape((-1, 1)))

        #  to compute derivatives of (mux0, mu0y) wrt Phi
        ivar = n_sum_categories
        for iman in range(ncat_men):
            rhs[iman, ivar:(ivar + ncat_women)] = -big_a[iman, :]
            ivar += ncat_women
        ivar1 = ncat_men
        ivar2 = n_sum_categories
        iend_phi = n_sum_categories + n_prod_categories
        for iwoman in range(ncat_women):
            rhs[ivar1, ivar2:iend_phi:ncat_women] = -big_a[:, iwoman]
            ivar1 += 1
            ivar2 += 1

        #  to compute derivatives of (mux0, mu0y) wrt sigma_x
        iend_sig = iend_phi + ncat_men
        der_sigx = np.sum(big_c, 1)
        rhs[:ncat_men, iend_phi:iend_sig] = np.diag(der_sigx)
        rhs[ncat_men:, iend_phi:iend_sig] = big_c.T
        #  to compute derivatives of (mux0, mu0y) wrt tau_y
        der_tauy = np.sum(big_d, 0)
        rhs[ncat_men:, iend_sig:] = np.diag(der_tauy)
        rhs[:ncat_men, iend_sig:] = big_d

        # solve for the derivatives of mux0 and mu0y
        dmu0 = spla.solve(lhs, rhs)
        dmux0 = dmu0[:ncat_men, :]
        dmu0y = dmu0[ncat_men:, :]

        # now construct the derivatives of muxy
        dmuxy = np.zeros((n_prod_categories, n_cols_rhs))
        der1 = ephi2 * der_axy1 * bxy
        ivar = 0
        for iman in range(ncat_men):
            dmuxy[ivar:(ivar + ncat_women), :] \
                = np.outer(der1[iman, :], dmux0[iman, :])
            ivar += ncat_women
        der2 = ephi2 * der_bxy1 * axy
        for iwoman in range(ncat_women):
            dmuxy[iwoman:n_prod_categories:ncat_women, :] \
                += np.outer(der2[:, iwoman], dmu0y[iwoman, :])

        # add the terms that comes from differentiating ephi2
        #  on the derivative wrt Phi
        i = 0
        j = n_sum_categories
        for iman in range(ncat_men):
            for iwoman in range(ncat_women):
                dmuxy[i, j] += big_a[iman, iwoman]
                i += 1
                j += 1
        #  on the derivative wrt sigma_x
        ivar = 0
        ix = iend_phi
        for iman in range(ncat_men):
            dmuxy[ivar:(ivar + ncat_women), ix] -= big_c[iman, :]
            ivar += ncat_women
            ix += 1
        # on the derivative wrt tau_y
        iy = iend_sig
        for iwoman in range(ncat_women):
            dmuxy[iwoman:n_prod_categories:ncat_women, iy] -= big_d[:, iwoman]
            iy += 1

        return (muxy, mux0, mu0y), marg_err_x, marg_err_y, (dmuxy, dmux0, dmu0y)


def print_simulated_ipfp(muxy, marg_err_x, marg_err_y):
    print("    simulated matching:")
    print(muxy[:4, :4])
    print(f"margin error on x: {npmaxabs(marg_err_x)}")
    print(f"             on y: {npmaxabs(marg_err_y)}")


if __name__ == "__main__":

    do_test_gradient_hetero = True
    do_test_gradient_heteroxy = False

    # we generate a Choo and Siow homo matching
    ncat_men = ncat_women = 25
    n_sum_categories = ncat_men + ncat_women
    n_prod_categories = ncat_men * ncat_women

    mu, sigma = 0.0, 1.0
    n_bases = 4
    bases_surplus = np.zeros((ncat_men, ncat_women, n_bases))
    x_men = (np.arange(ncat_men) - ncat_men / 2.0) / ncat_men
    y_women = (np.arange(ncat_women) - ncat_women / 2.0) / ncat_women

    bases_surplus[:, :, 0] = 1
    for iy in range(ncat_women):
        bases_surplus[:, iy, 1] = x_men
    for ix in range(ncat_men):
        bases_surplus[ix, :, 2] = y_women
    for ix in range(ncat_men):
        for iy in range(ncat_women):
            bases_surplus[ix, iy, 3] = \
                (x_men[ix] - y_women[iy]) * (x_men[ix] - y_women[iy])

    men_margins = np.random.uniform(1.0, 10.0, size=ncat_men)
    women_margins = np.random.uniform(1.0, 10.0, size=ncat_women)

    # np.random.normal(mu, sigma, size=n_bases)
    true_surplus_params = np.array([3.0, -1.0, -1.0, -2.0])
    true_surplus_matrix = bases_surplus @ true_surplus_params

    print_stars("Testing ipfp homo:")
    mus, marg_err_x, marg_err_y = \
        ipfp_homo_solver(true_surplus_matrix, men_margins,
                         women_margins, tol=1e-12)
    muxy, mux0, mu0y = mus
    print("    checking matching:")
    print(" true matching:")
    print(muxy[:4, :4])
    print_simulated_ipfp(muxy, marg_err_x, marg_err_y)

    # and we test ipfp hetero for tau = 1
    tau = 1.0
    print_stars("Testing ipfp hetero for tau = 1:")
    mus_tau, marg_err_x_tau, marg_err_y_tau = \
        ipfp_hetero_solver(true_surplus_matrix, men_margins,
                           women_margins, tau)
    print("    checking matching:")
    print(" true matching:")
    print(muxy[:4, :4])
    muxy_tau, _, _ = mus_tau
    print_simulated_ipfp(muxy_tau, marg_err_x_tau, marg_err_y_tau)

    # and we test ipfp heteroxy for sigma = tau = 1
    print_stars("Testing ipfp heteroxy for sigma_x and tau_y = 1:")

    sigma_x = np.ones(ncat_men)
    tau_y = np.ones(ncat_women)

    mus_hxy, marg_err_x_hxy, marg_err_y_hxy = \
        ipfp_heteroxy_solver(true_surplus_matrix, men_margins, women_margins,
                             sigma_x, tau_y)
    muxy_hxy, _, _ = mus_hxy
    print_simulated_ipfp(muxy_hxy, marg_err_x_hxy, marg_err_y_hxy)

    # and we test ipfp homo w/o singles
    print_stars("Testing ipfp homo w/o singles:")
    # we need as many women as men
    women_margins_nosingles = women_margins * \
                              (np.sum(men_margins) / np.sum(women_margins))
    muxy_nos, marg_err_x_nos, marg_err_y_nos = \
        ipfp_homo_nosingles_solver(true_surplus_matrix,
                                   men_margins, women_margins_nosingles, gr=False)
    print_simulated_ipfp(muxy_nos, marg_err_x_nos, marg_err_y_nos)

    # check the gradient
    iman = 3
    iwoman = 17

    GRADIENT_STEP = 1e-6

    if do_test_gradient_heteroxy:
        mus_hxy, marg_err_x_hxy, marg_err_y_hxy, dmus_hxy = \
            ipfp_heteroxy_solver(true_surplus_matrix, men_margins, women_margins,
                                 sigma_x, tau_y, gr=True)
        muij = mus_hxy[0][iman, iwoman]
        muij_x0 = mus_hxy[1][iman]
        muij_0y = mus_hxy[2][iwoman]
        gradij = dmus_hxy[0][iman * ncat_women + iwoman, :]
        gradij_x0 = dmus_hxy[1][iman, :]
        gradij_0y = dmus_hxy[2][iwoman, :]
        n_cols_rhs = n_prod_categories + 2 * n_sum_categories
        gradij_numeric = np.zeros(n_cols_rhs)
        gradij_numeric_x0 = np.zeros(n_cols_rhs)
        gradij_numeric_0y = np.zeros(n_cols_rhs)
        icoef = 0
        for ix in range(ncat_men):
            men_marg = men_margins.copy()
            men_marg[ix] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = \
                ipfp_heteroxy_solver(true_surplus_matrix, men_marg, women_margins,
                                     sigma_x, tau_y)
            gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
            gradij_numeric_x0[icoef] = (mus[1][iman] - muij_x0) / GRADIENT_STEP
            gradij_numeric_0y[icoef] = (mus[2][iwoman] - muij_0y) / GRADIENT_STEP
            icoef += 1
        for iy in range(ncat_women):
            women_marg = women_margins.copy()
            women_marg[iy] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = \
                ipfp_heteroxy_solver(true_surplus_matrix, men_margins, women_marg,
                                     sigma_x, tau_y)
            gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
            gradij_numeric_x0[icoef] = (mus[1][iman] - muij_x0) / GRADIENT_STEP
            gradij_numeric_0y[icoef] = (mus[2][iwoman] - muij_0y) / GRADIENT_STEP
            icoef += 1
        for i1 in range(ncat_men):
            for i2 in range(ncat_women):
                surplus_mat = true_surplus_matrix.copy()
                surplus_mat[i1, i2] += GRADIENT_STEP
                mus, marg_err_x, marg_err_y = \
                    ipfp_heteroxy_solver(surplus_mat, men_margins, women_margins,
                                         sigma_x, tau_y)
                gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
                gradij_numeric_x0[icoef] = (mus[1][iman] - muij_x0) / GRADIENT_STEP
                gradij_numeric_0y[icoef] = (mus[2][iwoman] - muij_0y) / GRADIENT_STEP
                icoef += 1
        for ix in range(ncat_men):
            sigma = sigma_x.copy()
            sigma[ix] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = \
                ipfp_heteroxy_solver(true_surplus_matrix, men_margins, women_margins,
                                     sigma, tau_y)
            gradij_numeric[icoef] \
                = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
            gradij_numeric_x0[icoef] = (mus[1][iman] - muij_x0) / GRADIENT_STEP
            gradij_numeric_0y[icoef] = (mus[2][iwoman] - muij_0y) / GRADIENT_STEP
            icoef += 1
        for iy in range(ncat_women):
            tau = tau_y.copy()
            tau[iy] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = \
                ipfp_heteroxy_solver(true_surplus_matrix, men_margins, women_margins,
                                     sigma_x, tau)
            gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
            gradij_numeric_x0[icoef] = (mus[1][iman] - muij_x0) / GRADIENT_STEP
            gradij_numeric_0y[icoef] = (mus[2][iwoman] - muij_0y) / GRADIENT_STEP
            icoef += 1

        diff_gradients = gradij_numeric - gradij
        error_gradient = np.abs(diff_gradients)

        describe_array(error_gradient, "error on the numerical gradient, heteroxy")

        diff_gradients_x0 = gradij_numeric_x0 - gradij_x0
        error_gradient_x0 = np.abs(diff_gradients_x0)

        describe_array(error_gradient_x0, "error on the numerical gradient x0, heteroxy")

        diff_gradients_0y = gradij_numeric_0y - gradij_0y
        error_gradient_0y = np.abs(diff_gradients_0y)

        describe_array(error_gradient_0y, "error on the numerical gradient 0y, heteroxy")

    if do_test_gradient_hetero:
        tau = 1.0
        mus_h, marg_err_x_h, marg_err_y_h, dmus_h = \
            ipfp_hetero_solver(true_surplus_matrix, men_margins, women_margins,
                               tau, gr=True)
        muij = mus_h[0][iman, iwoman]
        gradij = dmus_h[0][iman * ncat_women + iwoman, :]
        n_cols_rhs = n_prod_categories + n_sum_categories + 1
        gradij_numeric = np.zeros(n_cols_rhs)
        icoef = 0
        for ix in range(ncat_men):
            men_marg = men_margins.copy()
            men_marg[ix] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = \
                ipfp_hetero_solver(true_surplus_matrix, men_marg, women_margins,
                                   tau)
            gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
            icoef += 1
        for iy in range(ncat_women):
            women_marg = women_margins.copy()
            women_marg[iy] += GRADIENT_STEP
            mus, marg_err_x, marg_err_y = \
                ipfp_hetero_solver(true_surplus_matrix, men_margins, women_marg,
                                   tau)
            gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
            icoef += 1
        for i1 in range(ncat_men):
            for i2 in range(ncat_women):
                surplus_mat = true_surplus_matrix.copy()
                surplus_mat[i1, i2] += GRADIENT_STEP
                mus, marg_err_x, marg_err_y = \
                    ipfp_hetero_solver(surplus_mat, men_margins, women_margins,
                                       tau)
                gradij_numeric[icoef] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP
                icoef += 1
        tau_plus = tau + GRADIENT_STEP
        mus, marg_err_x, marg_err_y = \
            ipfp_hetero_solver(true_surplus_matrix, men_margins, women_margins,
                               tau_plus)
        gradij_numeric[-1] = (mus[0][iman, iwoman] - muij) / GRADIENT_STEP

        error_gradient = np.abs(gradij_numeric - gradij)

        describe_array(error_gradient, "error on the numerical gradient, hetero")
