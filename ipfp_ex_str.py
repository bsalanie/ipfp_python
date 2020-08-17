
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import plotly.express as px
from ipfp_solvers import ipfp_homo_solver
import random


def plot_heatmap(mat, str_tit):
    ncat_men, ncat_women = mat.shape
    mat_arr = np.empty((mat.size, 4))
    mat_min, mat_max = np.min(mat), np.max(mat)
    i = 0
    for ix in range(ncat_men):
        for iy in range(ncat_women):
            m = mat[ix, iy]
            s = 10.0*(0.5+2.0*(m - mat_min)/(mat_max - mat_min))
            mat_arr[i, :] = np.array([ix, iy, m, s])
            i += 1
            
    mat_df = pd.DataFrame(mat_arr, columns=['Men', 'Women', 'Value', 'Size'])
    mat_df = mat_df.astype(dtype={'Men': int, 'Women': int, 'Value':  float,
                                  'Size': float  })
    base = alt.Chart(mat_df).encode(
    x='Men:O',
    y=alt.Y('Women:O', sort="descending")
        )
    mat_map = base.mark_circle(opacity=0.4).encode(
    size=alt.Size('Size:Q', legend=None,
        scale=alt.Scale(range=[1000, 10000])),
        color=alt.Color('Value:Q'),
    #tooltip=alt.Tooltip('Value', format=".2f")
    )
    text = base.mark_text(baseline='middle', fontSize=16).encode(
        text=alt.Text('Value:Q', format=".2f"),
        )
    both = (mat_map + text).properties(title=str_tit, width=500, height=500)
    return both


st.title("Solving for matching equilibrium with IPFP")

st.markdown("This solves for equilibrium in a [Choo and Siow 2006](https://www.jstor.org/stable/10.1086/498585?seq=1) matching model with transferable utilities. It relies on the IPFP algorithm in [Galichon and Salanie 2020](https://econ.columbia.edu/working-paper/cupids-invisible-hand-social-surplus-and-identification-in-matching-models-2).")


st.header("Generating primitives for a marriage market")
st.subheader("First, choose numbers of categories of men and women")
ncat_men = st.radio("Number of categories of men", [2, 3, 4])
ncat_women = st.radio("Number of categories of women", [2, 3, 4])

nx = np.zeros(ncat_men)
my = np.zeros(ncat_women)
st.subheader("Second, choose the numbers of men and women in each category")
for iman in range(ncat_men):
    nx[iman] = st.slider(f"Number of men in category {iman+1}",
                         min_value=10, max_value=100, step=10)
for iwoman in range(ncat_women):
    my[iwoman] = st.slider(f"Number of women in category {iwoman+1}",
                           min_value=10, max_value=100, step=10)

# st.subheader("Let me allocate these men and women in categories")
# men_cats = np.array(random.choices(range(ncat_men), k=nmen))
# nx = np.zeros(ncat_men)
# for ix in range(ncat_men):
#     nx[ix] = np.sum(men_cats == ix)
# st.write("Here is the allocation of men")
# st.write(nx)
# women_cats = np.array(random.choices(range(ncat_women), k=nwomen))
# my = np.zeros(ncat_women)
# for iy in range(ncat_women):
#     my[iy] = np.sum(women_cats == iy)
# st.write("Here is the allocation of women")
# st.write(my)

st.subheader(
    "OK, now we generate the joint surpluses for all potential couples")

st.write("Choose coefficients for the six basis functions")
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

xvals = np.arange(ncat_men) + 1
yvals = np.arange(ncat_women) + 1

Phi = np.zeros((ncat_men, ncat_women))
Phi += c0
Phi += ((c1 * xvals+c3*xvals*xvals).reshape((-1, 1)))
Phi += (c2 * yvals+c4*yvals*yvals)
Phi += (c5*np.outer(xvals, yvals))

st.write("Here is your joint surplus by categories")
# st.write(Phi)

str_Phi = "Joint surplus"
st.altair_chart(plot_heatmap(Phi, str_Phi))


st.subheader("Time to solve for the equilibrium!")
solve_it = st.button("Solve")
if solve_it:
    (muxy, mux0, mu0y), marg_err_x, marg_err_y \
        = ipfp_homo_solver(Phi, nx, my)
    row_names = ['men %d' % i for i in xvals]
    col_names = ['women %d' % i for i in yvals]
    df_muxy = pd.DataFrame(muxy, index=row_names,
                           columns=col_names)
#    st.write("Matches by cell")
#    st.table(df_muxy)
    str_muxy = "Equilibrium Matching Patterns"
    st.altair_chart(plot_heatmap(muxy, str_muxy))

    st.write("Single men")
    df_mux0 = pd.DataFrame(mux0, index=row_names)
    st.table(df_mux0)
    st.write("Single women")
    df_mu0y = pd.DataFrame(mu0y, index=col_names)
    st.table(df_mu0y)

    # st.write("Would you like to check that this is indeed an equilibrium?")
    # if st.button("Check"):
    #     st.write("Errors on margins for men")
    #     st.write(marg_err_x)
    #     st.write("Errors on margins for women")
    #     st.write(marg_err_y)
    #     st.subheader("Checking that the Choo and Siow formula holds")
    #     impliedPhi = np.log(muxy*muxy/np.outer(mux0, mu0y))
    #     st.write("The errors on the implied joint surplus are:")
    #     df_Phi = pd.DataFrame(impliedPhi - Phi, index=row_names,
    #                           columns=col_names)
    #     st.table(df_Phi)
