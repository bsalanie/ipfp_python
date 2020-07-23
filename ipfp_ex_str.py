
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from ipfp_solvers import ipfp_homo_solver
import random

st.title("Solving for matching equilibrium with IPFP")

st.markdown("This solves for equilibrium in a [Choo and Siow 2006](https://www.jstor.org/stable/10.1086/498585?seq=1) matching model with transferable utilities. It relies on the IPFP algorithm in [Galichon and Salanie 2020](https://econ.columbia.edu/working-paper/cupids-invisible-hand-social-surplus-and-identification-in-matching-models-2).")


st.header("Generating primitives for a marriage market")
st.subheader("First, choose numbers of categories of men and women")
ncat_men = st.slider("Number of categories of men", min_value=2, max_value=10)
ncat_women = st.slider("Number of categories of women",  min_value=2, max_value=10)

st.subheader("How many men and women in total?")
nmen = st.slider("Total number of men", min_value=ncat_men, max_value=100)
nwomen = st.slider("Total number of women", min_value=ncat_women, max_value=100)


st.subheader("Let me allocate these men and women in categories")
men_cats = np.array(random.choices(range(ncat_men), k=nmen))
nx = np.zeros(ncat_men)
for ix in range(ncat_men):
    nx[ix] = np.sum(men_cats == ix)
st.write("Here is the allocation of men")
st.write(nx)
women_cats = np.array(random.choices(range(ncat_women), k=nwomen))
my = np.zeros(ncat_women)
for iy in range(ncat_women):
    my[iy] = np.sum(women_cats == iy) 
st.write("Here is the allocation of women")
st.write(my)

st.subheader("OK, now we generate the joint surpluses from all potential couples")


Phi = np.random.normal(size=ncat_men*ncat_women).reshape((ncat_men,ncat_women))
st.write("'Systematic' joint surplus by categories")
st.write(Phi)


st.subheader("Time to solve for equilibrium")
if st.button("Solve"):
    (muxy, mux0, mu0y), marg_err_x, marg_err_y \
        = ipfp_homo_solver(Phi, nx, my)
    row_names  = ['men %d' % i for i in range(ncat_men)]
    col_names = ['women %d' % i for i in range(ncat_women)]
    df_muxy = pd.DataFrame(muxy, index=row_names,
                           columns=col_names)
    st.write("Matches by cell")
    st.table(df_muxy)
    st.write("Errors on margins for men")
    st.write(marg_err_x)
    st.write("Errors on margins for women")
    st.write(marg_err_y)
    st.subheader("Checking that the Choo and Siow formula holds")
    impliedPhi = np.log(muxy*muxy/np.outer(mux0, mu0y))
    st.write("THe errors on the implied joint surplus are:")
    df_Phi = pd.DataFrame(impliedPhi - Phi, index=row_names,
                           columns=col_names)
    st.table(df_Phi)
    



