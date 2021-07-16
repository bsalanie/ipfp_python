""" 
Interactive Streamlit application that solves for the stable matchinf and estimates the parameters of the joint surplus \
    in a `Choo and Siow 2006 <https://www.jstor.org/stable/10.1086/498585?seq=1>`_ model \
        (homoskedastic model with singles)
"""


import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from ipfp_utils import nprepeat_col, nprepeat_row
from ipfp_solvers import ipfp_homo_solver
from estimate_cs_fuvl import estimate_cs_fuvl


def plot_heatmap(mat, str_tit):
    ncat_men, ncat_women = mat.shape
    mat_arr = np.empty((mat.size, 4))
    mat_min, mat_max = np.min(mat), np.max(mat)
    i = 0
    for ix in range(ncat_men):
        for iy in range(ncat_women):
            m = mat[ix, iy]
            s = m - mat_min + 1
            mat_arr[i, :] = np.array([ix, iy, m, s])
            i += 1

    mat_df = pd.DataFrame(mat_arr, columns=['Men', 'Women', 'Value', 'Size'])
    mat_df = mat_df.astype(dtype={'Men': int, 'Women': int, 'Value':  float,
                                  'Size': float})
    base = alt.Chart(mat_df).encode(
        x='Men:O',
        y=alt.Y('Women:O', sort="descending")
    )
    mat_map = base.mark_circle(opacity=0.4).encode(
        size=alt.Size('Size:Q', legend=None,
                      scale=alt.Scale(range=[1000, 10000])),
        color=alt.Color('Value:Q'),
        # tooltip=alt.Tooltip('Value', format=".2f")
    )
    text = base.mark_text(baseline='middle', fontSize=16).encode(
        text=alt.Text('Value:Q', format=".2f"),
    )
    both = (mat_map + text).properties(title=str_tit, width=500, height=500)
    return both


st.title("Solving for matching equilibrium with IPFP")

st.markdown("This solves for equilibrium in a [Choo and Siow 2006](https://www.jstor.org/stable/10.1086/498585?seq=1) matching model with transferable utilities. It relies on the IPFP algorithm in [Galichon and Salanie 2021](http://bsalanie.com/wp-content/uploads/2021/06/2021-06-1_Cupids.pdf).")


st.header("Generating primitives for a marriage market")
st.subheader("First, choose numbers of categories of men and women")
ncat_men = st.radio("Number of categories of men", list(range(2, 5)))
ncat_women = st.radio("Number of categories of women", list(range(2, 5)))

nx = np.zeros(ncat_men)
my = np.zeros(ncat_women)
st.subheader("Second, choose the numbers of men and women in each category")
for iman in range(ncat_men):
    nx[iman] = st.slider(f"Number of men in category {iman+1}",
                         min_value=1, max_value=10, step=1)
for iwoman in range(ncat_women):
    my[iwoman] = st.slider(f"Number of women in category {iwoman+1}",
                           min_value=1, max_value=10, step=1)

st.subheader(
    "Now we generate the joint surpluses for all potential couples")

npars = (ncat_men-1)*(ncat_women - 1)
st.markdown(
    f"Since you chose {ncat_men} types for men and {ncat_women} for women, we can estimate at most ({ncat_men}-1) ({ncat_women} - 1)= {npars} parameters.")
terms_deg = [1, 3, 6, 10]
for ideg in range(4):
    if npars < terms_deg[ideg]:
        break
max_deg = ideg - 1
n_bases = terms_deg[max_deg]
if max_deg == 0:
    st.write("Choose a coefficient for the basis function")
    st.write("$\Phi_{xy}=c_0$")
    c0 = st.slider("c0",
                   min_value=-5.0, max_value=5.0, value=0.0)
    coeffs = np.array([c0])
    coeff_names = ["c0"]
elif max_deg == 1:
    st.write("Choose a coefficient for the 3 basis functions")
    st.write("$\Phi_{xy}=c_0+c_1 x + c_2 y$")
    c0 = st.slider("c0",
                   min_value=-5.0, max_value=5.0, value=0.0)
    c1 = st.slider("c1",
                   min_value=-1.0, max_value=1.0, value=0.0)
    c2 = st.slider("c2",
                   min_value=-1.0, max_value=1.0, value=0.0)
    coeffs = np.array([c0, c1, c2])
    coeff_names = ["c0", "c1", "c2"]
elif max_deg == 2:
    st.write("Choose a coefficient for the 6 basis functions")
    st.write("$\Phi_{xy}=c_0+c_1 x + c_2 y + c_3 x^2 + c_4 x y + c_5 y^2$")
    c0 = st.slider("c0",
                   min_value=-5.0, max_value=5.0, value=0.0)
    c1 = st.slider("c1",
                   min_value=-1.0, max_value=1.0, value=0.0)
    c2 = st.slider("c2",
                   min_value=-1.0, max_value=1.0, value=0.0)
    c3 = st.slider("c3",
                   min_value=-1.0, max_value=1.0, value=0.0)
    c4 = st.slider("c4",
                   min_value=-1.0, max_value=1.0, value=0.0)
    c5 = st.slider("c5",
                   min_value=-1.0, max_value=1.0, value=0.0)
    coeffs = np.array([c0, c1, c2, c3, c4, c5])
    coeff_names = ["c0", "c1", "c2", "c3",
                   "c4", "c5"]
else:
    st.wrote("PROBLEM!!!")


xvals = np.arange(ncat_men) + 1
yvals = np.arange(ncat_women) + 1

bases = np.zeros((ncat_men, ncat_women, n_bases))
bases[:, :, 0] = 1.0
if max_deg >= 1:
    xvals_mat = nprepeat_col(xvals, ncat_women)
    yvals_mat = nprepeat_row(yvals, ncat_men)
    bases[:, :, 1] = xvals_mat
    bases[:, :, 2] = yvals_mat
if max_deg == 2:
    bases[:, :, 3] = xvals_mat*xvals_mat
    bases[:, :, 4] = np.outer(xvals, yvals)
    bases[:, :, 5] = yvals_mat*yvals_mat
Phi = bases @ coeffs


st.write("Here is your joint surplus by categories")

str_Phi = "Joint surplus"
st.altair_chart(plot_heatmap(Phi, str_Phi))


st.subheader("And here are the stable matching patterns:")

(muxy, mux0, mu0y), marg_err_x, marg_err_y\
    = ipfp_homo_solver(Phi, nx, my)
row_names = ['men %d' % i for i in xvals]
col_names = ['women %d' % i for i in yvals]
df_muxy = pd.DataFrame(muxy, index=row_names,
                       columns=col_names)

str_muxy = "Stable Matching Patterns"
st.altair_chart(plot_heatmap(muxy, str_muxy))

st.write("Singles:")
df_mux0 = pd.DataFrame({'Single men': mux0}, index=row_names)
st.table(df_mux0)
df_mu0y = pd.DataFrame({'Single women': mu0y}, index=col_names)
st.table(df_mu0y)

st.subheader(
    "Estimating the parameters.")
st.write("Now we use the stable matching as data,  and we estimate the coefficients of the basis functions.")
st.markdown(
    "We do this by minimizing the globally convex function given in equation (3.14)  of [Galichon and Salanie 2021](http://bsalanie.com/wp-content/uploads/2021/06/2021-06-1_Cupids.pdf).")
st.write(
    "It will also give us the estimates of the expected utilities $u_x$ and $v_y$.")

data_muxy = muxy
data_nx = mux0 + np.sum(muxy, 1)
data_my = mu0y + np.sum(muxy, 0)

resus = estimate_cs_fuvl(muxy, nx, my, bases)
u = resus.x[:ncat_men]
v = resus.x[ncat_men:-n_bases]
l = resus.x[-n_bases:]
st.write(resus.x)
st.write("The coefficients are:")
df_coeffs_estimates = pd.DataFrame({'Estimated': l,
                                    'True': coeffs},
                                   index=coeff_names)
st.table(df_coeffs_estimates)
st.write("The expected utilities are:")
df_u_estimates = pd.DataFrame({'Estimated': u,
                               'True': -np.log(mux0/nx)},
                              index=row_names)
st.table(df_u_estimates)
df_v_estimates = pd.DataFrame({'Estimated': v,
                               'True': -np.log(mu0y/my)},
                              index=col_names)
st.table(df_v_estimates)
