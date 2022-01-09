import streamlit as st
import pandas as pd

import plotly.express as px
import plotly.graph_objs as go
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler


BASE_COLUMNS = ['Date', 'Rented Bike Count', 'Hour', 'Temperature', 'Humidity',
       'Wind speed', 'Visibility', 'Dew point temperature',
       'Solar Radiation', 'Rainfall', 'Snowfall', 'Seasons',
       'Holiday', 'Functioning Day']

NEW_COLUMNS = ['Date', 'Hour', 'Temperature', 'Dew point temperature', 
       'Wind speed', 'Humidity', 'Visibility',
       'Solar Radiation', 'Rainfall', 'Snowfall', 'Seasons',
       'Holiday', 'Functioning Day', 'Rented Bike Count']


def load_datas(file, encoding):
    dataset = pd.read_csv(file, encoding=encoding)
    if "Rented Bike Count" in dataset.columns:
        dataset.columns = BASE_COLUMNS
        dataset["Rented Bike Count"] = dataset["Rented Bike Count"].apply(float)
        dataset = dataset.reindex(NEW_COLUMNS, axis=1)
    else:
        dataset.columns = BASE_COLUMNS[0:1] + BASE_COLUMNS[2:]
        dataset = dataset.reindex(NEW_COLUMNS[:-1], axis=1)

    dataset["Humidity"] = dataset["Humidity"].apply(float)
    dataset["Visibility"] = dataset["Visibility"].apply(float)
    
    return dataset


def find_numerical_categorical_cols(dataset):
    numerical_cols = [col for col in dataset.columns if dataset[col].dtype == float]
    categorical_cols = [col for col in dataset.columns if dataset[col].dtype != float] #(object, int, str)]
    return numerical_cols, categorical_cols


def distribution_plot(dataset):
    st.write('**Distribution analysis**')

    open = st.checkbox("See the plot", key=2)

    if open:
        numerical_cols, _ = find_numerical_categorical_cols(dataset)

        select = st.selectbox("Which variable's distribution are you interested to see?", dataset[numerical_cols].columns)

        fig = px.histogram(dataset, x=select)
        st.plotly_chart(fig, use_container_width=True)


def violin_strip_plot(dataset):
    st.write('**Violin plot analysis**')

    open = st.checkbox("See the plot", key=3)

    if open:
        _, categorical_cols = find_numerical_categorical_cols(dataset)
        categorical_cols.remove("Date")

        select = st.selectbox("Which variable are you interested to compare it to the Rented Bike Count variable?", categorical_cols)

        fig=px.violin(dataset, y=dataset["Rented Bike Count"], x=dataset[select],hover_data = categorical_cols, color=dataset[select])
        st.plotly_chart(fig, use_container_width=True)
    


def scatter_features(dataset):
    st.write('**Scatterplot analysis**')

    open = st.checkbox("See the plot", key=4)

    if open:
        selected_x_var = st.selectbox('What do you want the x variable to be?', dataset.columns)
        selected_y_var = st.selectbox('What about the y?', dataset.columns, index= len(dataset.columns)-1)
        fig = px.scatter(dataset, x = dataset[selected_x_var], y = dataset[selected_y_var])
        st.plotly_chart(fig, use_container_width=True)


def preprocess_dataset(dataset):
    dataset = dataset.copy()

    dataset["Month"] = dataset["Date"].apply(lambda x: int(x.split("/")[1]))

    dataset["Date"] = pd.to_datetime(dataset["Date"])
    dataset["WeekEnd"] = dataset["Date"].apply(lambda x: 1 if x.weekday() >= 5 else 0)

    dataset.drop(dataset[dataset["Functioning Day"] == "No"].index, inplace=True)
    dataset.drop("Functioning Day", axis = 1, inplace=True)

    holiday_no_holiday = {
    "No Holiday":0,
    "Holiday":1
    }
    dataset["Holiday"] = dataset["Holiday"].map(holiday_no_holiday)

    dataset.drop("Date", axis = 1, inplace=True)

    dataset = pd.get_dummies(dataset)
    return dataset


def seperate_X_y(dataset):
    X = dataset.drop("Rented Bike Count", axis=1)
    y = dataset["Rented Bike Count"]

    return X, y


def split_dataset(X, y, test_size, seed):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size)
    return X_train, X_test, y_train, y_test


def scaling_datas(X_train, X_test):
    numerical_cols, categorical_cols = find_numerical_categorical_cols(X_train)

    scaler = StandardScaler()

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train[numerical_cols]), columns=numerical_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test[numerical_cols]), columns=numerical_cols, index=X_test.index)

    X_train_scaled = pd.concat([X_train[categorical_cols], X_train_scaled], axis=1)
    X_test_scaled = pd.concat([X_test[categorical_cols], X_test_scaled], axis=1)

    return X_train_scaled, X_test_scaled, scaler


def create_model(model, X_train, y_train):
    model = model.fit(X_train, y_train)
    return model



def print_metrics(y_test, y_pred):
    return mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)


def result_visu(y_test,y_pred, column):
    diff = y_test - y_pred
    fig = px.histogram(      
        diff,  
        x = diff,
        #title = "Histogram of prediction errors",
    )
    fig.update_layout(xaxis_title='Rented bikes prediction error', yaxis_title="Frequency")
    column.plotly_chart(fig, use_container_width=True)


def r2_score_plt(r2_score, column):
    fig = go.Figure(
        go.Indicator(
        mode="gauge+number+delta",
        value=r2_score,
        title={"text": f"R Squared Score"},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={"axis": {"range": [0, 1]}},
    ))

    column.plotly_chart(fig, use_container_width=True)


