import streamlit as st
from sklearn.linear_model import LinearRegression

def lr_param_selector(seed):
    params = {

    }
    model = LinearRegression(**params)
    return model