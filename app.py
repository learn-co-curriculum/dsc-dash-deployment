########## IMPORTS ##########

from flask import Flask, request
import joblib
import json

from dash import Dash, html, dcc, dash_table, Input, Output
import dash_bootstrap_components as dbc

import pandas as pd
from sklearn.datasets import load_iris

########## SETTING UP THE APPS ##########

flask_app = Flask(__name__)
external_stylesheets = [dbc.themes.BOOTSTRAP]
dash_app = Dash(__name__, external_stylesheets=external_stylesheets, server=flask_app)

########## HELPER FUNCTIONS ##########

def create_sliders(X):
    slider_items = []
    for column in X:
        label = html.H5(column)
        
        lower_bound = X[column].min()
        upper_bound = X[column].max()
        value = X[column].median()

        slider = dcc.Slider(
            lower_bound,
            upper_bound,
            value=value, # set median as default
            marks=None,
            tooltip={"always_visible": True},
            id=column # set id based on column name
        )

        item = dbc.ListGroupItem(children=[
            label,
            slider
        ])
        slider_items.append(item)
    return dbc.ListGroup(slider_items)

def create_list_group(selected_row_data):
    return dbc.ListGroup([
        dbc.ListGroupItem(f"{k}: {v}") for k, v in selected_row_data.items()
    ])

def create_image_card(selected_row_data):
    iris_class = selected_row_data["class"]
    if iris_class == 0:
        name = "Iris setosa "
        img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/180px-Kosaciec_szczecinkowaty_Iris_setosa.jpg"
        img_source = "https://commons.wikimedia.org/wiki/File:Kosaciec_szczecinkowaty_Iris_setosa.jpg"
    elif iris_class == 1:
        name = "Iris versicolor "
        img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/320px-Iris_versicolor_3.jpg"
        img_source = "https://commons.wikimedia.org/wiki/File:Iris_versicolor_3.jpg"
    else:
        name = "Iris virginica "
        img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/295px-Iris_virginica.jpg"
        img_source = "https://commons.wikimedia.org/wiki/File:Iris_virginica.jpg"

    return dbc.Card(children=[
        dbc.CardImg(src=img_url),
        dbc.CardBody(children=[
            html.Em(name),
            html.Small(html.A("(image source)", href=img_source, target="blank_"))
        ])
    ])

def iris_prediction(sepal_length, sepal_width, petal_length, petal_width):
    """
    Given sepal length, sepal width, petal length, and petal width,
    predict the class of iris
    """
    with open("model.pkl", "rb") as f:
        model = joblib.load(f)
    X = [[sepal_length, sepal_width, petal_length, petal_width]]
    predictions = model.predict(X)
    # model.predict takes a list of records and returns a list of predictions
    # but we are only making a single prediction
    prediction = int(predictions[0])
    return {"predicted_class": prediction}

def check_prediction(selected_row_data):
    """
    Return an Alert component with information about the model's prediction
    vs. the true class value
    """
    data_copy = selected_row_data.copy()
    actual_class = data_copy.pop("class")
    # remove " (cm)" from labels
    data_cleaned = {k.split(" (cm)")[0].replace(" ", "_"):v for k, v in data_copy.items()}
    result = iris_prediction(**data_cleaned)
    predicted_class = result["predicted_class"]
    correct_prediction = predicted_class == actual_class
    if correct_prediction:
        color = "success"
    else:
        color = "danger"
    return dbc.Alert(f"Predicted class: {predicted_class}", color=color)

########## DECLARING LAYOUT COMPONENTS ##########

data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="class")
full_dataset = pd.concat([X, y], axis=1)

prediction_layout = html.Div(children=[
    create_sliders(X),
    dbc.Alert("Prediction will go here", color="info", id="prediction-output")
]) 

markdown = dcc.Markdown("""
## Iris Training Dataset

Below is a DataTable showing a sample of 20 records from the
 [Iris Dataset](https://en.wikipedia.org/wiki/Iris_flower_data_set).

Select any record to view more information!
""")

table = dash_table.DataTable(
    data=full_dataset.sample(10, random_state=4).to_dict(orient="records"),
    row_selectable="single",
    cell_selectable=False,
    id="tbl"
)

modal = dbc.Modal(children=[
    dbc.ModalHeader(dbc.ModalTitle("Iris Information")),
    dbc.ModalBody(id="modal-body")
],
                  id="modal",
                  is_open=False
                 )

past_data_layout = html.Div(children=[
    html.Div(markdown),
    html.Div(table),
    modal
])

tabs = dbc.Tabs(children=[
        dbc.Tab(prediction_layout, label="Generate Predictions on New Data"),
        dbc.Tab(past_data_layout, label="Analyze Performance on Past Data")
    ])

dash_app.layout = dbc.Container(children=[
    html.H1("Iris Classification Model"),
    tabs
])

########## CALLBACKS ##########

@dash_app.callback(
    Output("prediction-output", "children"),
    [
        # list comprehension to specify all of the input columns
        Input(column, "value") for column in X.columns
    ]
)
def generate_user_input_prediction(*args):
    return f"Predicted class: {iris_prediction(*args)['predicted_class']}"

@dash_app.callback(Output("modal", "is_open"), Input("tbl", "selected_rows"))
def toggle_modal(selected_rows):
    if selected_rows:
        return True
    else:
        return False

@dash_app.callback(
    Output("modal-body", "children"),
    [Input("tbl", "derived_virtual_data"), Input("tbl", "selected_rows")])
def render_information(rows, selected_rows):
    if selected_rows:
        # selection is set to "single" so there will be exactly 1 selected row
        selected_row_data = rows[selected_rows[0]]      
        return html.Div(dbc.Row(children=[
            dbc.Col(create_image_card(selected_row_data)),
            dbc.Col(children=[
                create_list_group(selected_row_data),
                html.Hr(),
                check_prediction(selected_row_data)
            ])
        ]))

########## ROUTES ##########

@flask_app.route('/predict', methods=['POST'])
def predict():
    request_json = request.get_json()
    result = iris_prediction(**request_json)
    return json.dumps(result)
