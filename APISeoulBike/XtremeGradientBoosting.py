import streamlit as st
from xgboost import XGBRegressor

def xgb_param_selector(seed):
    n_estimators = st.number_input("n_estimators", 100, 1000, 100, 50)
    learning_rate = st.number_input("learning_rate (%)", 1, 100, 10, 1) ## Afficher la lerning rate en %
    max_depth = st.number_input("max_depth", 1, 10, 3, 1)

    params = {
        "n_estimators": n_estimators,
        "learning_rate":learning_rate/100,
        "max_depth": max_depth,
    }
    
    model = XGBRegressor(**params)
    return model
