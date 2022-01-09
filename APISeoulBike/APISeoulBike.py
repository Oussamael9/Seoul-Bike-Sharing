import streamlit as st

from RandomForest import rf_param_selector
from DecisionTree import dt_param_selector
from XtremeGradientBoosting import xgb_param_selector
from LinearRegression import lr_param_selector
from SupportVectorMachine import svm_param_selector

from functions import *

st.set_page_config(page_title="Seoul Bike", page_icon= "🚲", layout="wide")

st.markdown(""" <style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

RANDOM_SEED = 1

st.title("**API Seoul Bike** 🚴")

st.write("This is an app where you can train a model for predicting the number of bike rented in a day.")
st.write("""
Currently Rental bikes are introduced in many urban cities for the enhancement of mobility comfort.
 It is important to make the rental bike available and accessible to the public at the right time as it lessens the waiting time. 
 Eventually, providing the city with a stable supply of rental bikes becomes a major concern. 
 The crucial part is the prediction of bike count required at each hour for the stable supply of rental bikes.

 ** Data source: ** [Seoul Bike Sharing Demand Data Set](https://archive.ics.uci.edu/ml/datasets/Seoul+Bike+Sharing+Demand)
 """)

st.header("***Dataset Loading*** 💻")

with st.expander("Data informations", expanded=True):
     st.write("""
     The dataset must have these columns names in this order.

        - **Date** : Format DD/MM/YYYY
        - **Rented Bike Count** : The number of bikes who will be rented
        - **Hour** : From 0 to 23
        - **Temperature(°C)** : float value
        - **Humidity(%)** : float value
        - **Wind speed (m/s)** : float value
        - **Visibility (10m)** : float value
        - **Dew point temperature(°C)** : float value
        - **Solar Radiation (MJ/m2)** : float value
        - **Rainfall(mm)** : float value
        - **Snowfall (cm)** : float value
        - **Seasons** : Winter, Spring, Summer, Automn
        - **Holiday** : Holiday or No Holiday
        - **Functioning Day** : Yes or No
     """)

uploaded_file = st.file_uploader("Import a csv file.", "csv")

drive_file = st.checkbox("Import the file directly from google drive (option not added yet)")

if drive_file:
    pass
else:
    if uploaded_file:
        dataset = load_datas(uploaded_file, encoding="latin-1")
        st.write(dataset.sample(9, random_state=RANDOM_SEED))


### Visualisation

st.header("***Data Visualisation*** 🔎")
st.write("You can visualise some of the correlation between the features in your datas here. Load a dataset to use this part.")

st.subheader("***Statistics***")
st.write("Here you can see some statistics of your variables like the mean or the standard deviation.")
if drive_file or uploaded_file:
    col1, _ , col3 = st.columns([3, 1, 3])

    numerical_cols, categorical_cols = find_numerical_categorical_cols(dataset)

    feature_selected = col1.selectbox("Name of the feature", [None] + numerical_cols)
    if feature_selected:
        col3.write("Decription of the feature")
        col3.write(dataset[feature_selected].describe())



st.subheader("***Plots***")
st.write("Here you can visualize some plots to see the different repartitions or correlations between variables")

if drive_file or uploaded_file:
    distribution_plot(dataset)
    violin_strip_plot(dataset)
    scatter_features(dataset)


### Sidebar components

st.sidebar.header("ML Model Creation")

side_expander_split = st.sidebar.expander("Split the dataset", expanded=True)
test_size = side_expander_split.slider("test set size (%)", 0, 100, 20)

models = {
    "Linear regression":lr_param_selector,
    "Support Vector Machine":svm_param_selector,
    "Decision Tree":dt_param_selector,
    "Random Forest":rf_param_selector,
    "Xtreme Gradient Boosting":xgb_param_selector,
}

side_expander_train = st.sidebar.expander("Train a model", expanded=True)
model_selected = side_expander_train.selectbox("Choose a model", models.keys())

side_expender_tune = st.sidebar.expander("Tune the hyperparameters", expanded=True)
with side_expender_tune:
    model = models[model_selected](RANDOM_SEED)

button = st.sidebar.button("Train model")
###


### Feature Engineering
if drive_file or uploaded_file:
    preprocessed_dataset = preprocess_dataset(dataset)
    X, y = seperate_X_y(preprocessed_dataset)
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size/100, RANDOM_SEED)
    X_train_scaled, X_test_scaled, scaler = scaling_datas(X_train, X_test)

###


### Validation

st.header("***Validation of the model*** ✔️")
st.write("You must create a ML model with the sidebar for this section.")
st.write("You can verify the accuracy of your model and avoid underfitting and overfitting in this part.")

if (drive_file or uploaded_file) and button:
    model = create_model(model, X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse, r2 = print_metrics(y_test, y_pred)

    col1, col3 = st.columns(2)

    result_visu(y_test, y_pred, col1)
    r2_score_plt(r2, col3)

###

