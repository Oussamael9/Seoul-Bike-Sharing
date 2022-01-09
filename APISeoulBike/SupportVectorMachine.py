import streamlit as st
from sklearn.svm import SVR

def svm_param_selector(seed):
    C = st.number_input("C", 0.1, 5.0, 1.0, 0.1)
    epsilon = st.number_input("epsilon", 0.0, 2.0, 0.1, 0.1)
    kernel = st.selectbox("kernel", {"linear", "poly", "rbf", "sigmoid", "precomputed"})
    degree = st.number_input("degree", 1, 6, 3, 1)


    params = {
        "C":C,
        "epsilon":epsilon,
        "kernel":kernel,
        "degree":degree
    }
    model = SVR(**params)
    return model