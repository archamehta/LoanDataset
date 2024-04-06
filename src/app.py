import dash
from dash import dcc, html, Input,Output,State
import seaborn as sns
import plotly.express as px
from dash.exceptions import PreventUpdate
import io
import base64
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load the dataset
my_df = pd.read_csv('https://raw.githubusercontent.com/archamehta/LoanDataset/main/src/loan_data.csv')

# Get the categorical columns
categorical_columns = my_df.select_dtypes(include=['object', 'category']).columns.tolist()

# Get the numeric columns
numeric_columns = my_df.select_dtypes(include=['number']).columns.tolist()

# Define dropdowns for each graph
target_dropdown1 = html.Div(className="dropdown2_div", children=[
    html.P("Select Target 1: "),
    dcc.Dropdown(id='target_dropdown_id1', options=[{'label': col, 'value': col} for col in numeric_columns], style=dict(width=150, marginLeft=2))
])

category_dropdown1 = html.Div(className="dropdown2_div", children=[
    html.P("Select Category 1: "),
    dcc.Dropdown(id='cat_dropdown_id1', options=[{'label': col, 'value': col} for col in categorical_columns], style=dict(width=150, marginLeft=2))
])

target_dropdown2 = html.Div(className="dropdown2_div", children=[
    html.P("Select Target 2: "),
    dcc.RadioItems(id='target_dropdown_id2', options=[{'label': col, 'value': col} for col in numeric_columns], style=dict(width=150, marginLeft=2))
])

category_dropdown2 = html.Div(className="dropdown2_div", children=[
    html.P("Select Category 2: "),
    dcc.RadioItems(id='cat_dropdown_id2', options=[{'label': col, 'value': col} for col in numeric_columns], style=dict(width=150, marginLeft=2))
])
target_dropdown3 = html.Div(className="dropdown2_div", children=[
    html.P("Select Target 3: "),
    dcc.Dropdown(id='target_dropdown_id3', options=[{'label': col, 'value': col} for col in numeric_columns], style=dict(width=150, marginLeft=2))
])

category_dropdown3 = html.Div(className="dropdown2_div", children=[
    html.P("Select Category 3: "),
    dcc.Dropdown(id='cat_dropdown_id3', options=[{'label': col, 'value': col} for col in categorical_columns], style=dict(width=150, marginLeft=2))
])

target_dropdown4 = html.Div(className="dropdown2_div", children=[
    html.P("Select Target 4: "),
    dcc.Dropdown(id='target_dropdown_id4', options=[{'label': col, 'value': col} for col in numeric_columns], style=dict(width=150, marginLeft=2))
])

category_dropdown4 = html.Div(className="dropdown2_div", children=[
    html.P("Select Category 4: "),
    dcc.Dropdown(id='cat_dropdown_id4', options=[{'label': col, 'value': col} for col in numeric_columns], style=dict(width=150, marginLeft=2))
])

check_list = dcc.Checklist(id="row3_checklist", options=[{'label': col, 'value': col} for col in my_df.columns], value=[], inline=True)

app = dash.Dash(__name__)
server = app.server

# Add heading with custom background color
heading = html.H1("Loan Prediction Dataset", style={'text-align': 'center', 'color': 'white', 'background-color': 'blue'})

app.layout = html.Div([
    heading,  # Add the heading here
    html.Div(className="parent_container", children=[
        html.Div(id="row2", children=[
            html.Div(className="row2_child", children=[
                html.Div([category_dropdown1, target_dropdown1]),
                html.Div(dcc.Graph(id='graph1'), style=dict(width="50", display="inline-block")),
                html.Div([category_dropdown2, target_dropdown2]),
                html.Div(dcc.Graph(id='graph2'), style=dict(width="50%", display="inline-block")),
            ]),
            html.Div(className="row2_child", children=[
                html.Div([category_dropdown3, target_dropdown3]),
                html.Div(dcc.Graph(id='graph3'), style=dict(width="50%", display="inline-block")),
                html.Div([category_dropdown4, target_dropdown4]),
                html.Div(dcc.Graph(id='graph4'), style=dict(width="50", display="inline-block")),
            ]),
        ]),
        html.Div(id="row3", children=[
            html.Div(id="row3_child1", children=[check_list, html.Button('Train', id='train_button_id', n_clicks=0)])
        ]),
        html.Div(id="row4", children=[]),
        html.Div(id="row5", children=[
            dcc.Input(id="row4_input_id", type="text", placeholder="input type"),
            html.Button('Predict', id='predict_button_id', n_clicks=0),
            html.P(id="prediction_holder", children=[""])
        ]),
    ])
])

# Define callback function for updating graph 1
@app.callback(Output('graph1', 'figure'), [Input('cat_dropdown_id1', 'value'), Input('target_dropdown_id1', 'value')])
def update_graph1(cat_dropdown_val, target_variable):
    if target_variable is not None and cat_dropdown_val is not None:
        data = my_df.groupby(cat_dropdown_val)[target_variable].mean().reset_index()
        figure = px.pie(data, names=cat_dropdown_val, values=target_variable, title=f'Pie Chart: {target_variable} by {cat_dropdown_val}')
        return figure
    else:
        return {}

# Define callback function for updating graph 2
@app.callback(Output('graph2', 'figure'), [Input('cat_dropdown_id2', 'value'), Input('target_dropdown_id2', 'value')])
def update_graph2(cat_dropdown_val, target_variable):
    if target_variable is not None and cat_dropdown_val is not None:
        data = my_df.groupby(cat_dropdown_val)[target_variable].mean().reset_index()
        figure = px.scatter(data, x=cat_dropdown_val, y=target_variable, title=f'Box Plot: {target_variable} by {cat_dropdown_val}')
        return figure
    else:
        return {}

# Define callback function for updating graph 3
@app.callback(Output('graph3', 'figure'), [Input('cat_dropdown_id3', 'value'), Input('target_dropdown_id3', 'value')])
def update_graph3(cat_dropdown_val, target_variable):
    if target_variable is not None and cat_dropdown_val is not None:
        data = my_df.groupby(cat_dropdown_val)[target_variable].mean().reset_index()
        figure = px.bar(data, x=cat_dropdown_val, y=target_variable, title=f'Bar Chart: {target_variable} by {cat_dropdown_val}')
        return figure
    else:
        return {}

# Define callback function for updating graph 4
@app.callback(Output('graph4', 'figure'), [Input('cat_dropdown_id4', 'value'), Input('target_dropdown_id4', 'value')])
def update_graph4(cat_dropdown_val, target_variable):
    if target_variable is not None and cat_dropdown_val is not None:
        figure = px.scatter(my_df, x=cat_dropdown_val, y=target_variable, title=f'Scatter Plot: {target_variable} by {cat_dropdown_val}')
        return figure
    else:
        return {}

# store the features and the target
@app.callback(Output('row5', 'children'),[State('target_dropdown_id1', 'value'),State("row3_checklist","value"),Input('train_button_id', 'n_clicks')])
def train_model(target,selected_features, n_clicks):
    if(target is None or len(selected_features)==0):
        raise PreventUpdate
    global model_pipeline
    X = my_df[selected_features]
    y = my_df[target]
    selected_categorical_features = my_df[selected_features].select_dtypes(include=['object', 'category']).columns.tolist()
    selected_numerical_features =my_df[selected_features].select_dtypes(include=['number']).columns.tolist()

    numeric_transformer = Pipeline(steps=[('scaler', MinMaxScaler())]) #Define the preprocessing pipeline for numerical features
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))]) #Define the preprocessing pipeline for categorical features
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, selected_numerical_features),('cat', categorical_transformer, selected_categorical_features)]) # Combine preprocessing steps
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),('regressor', LinearRegression())]) # Create the complete pipeline

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model_pipeline.fit(X_train, y_train) #Train the model
    y_pred = model_pipeline.predict(X_test) #Predict on the test set
    r2 = r2_score(y_test, y_pred) #Evaluate the model
    return "The R2 score is: "+str(round(r2,2))


# store the features and the target
@app.callback(Output('row4_input_id', 'placeholder'),[Input("row3_checklist","value")])
def change_input_placeholder(selected_features):
    if(len(selected_features)==0):
        raise PreventUpdate
    else:
        return ",".join(selected_features)


@app.callback(Output('prediction_holder', 'children'),[State("row3_checklist","value"),State("row4_input_id","value"),Input('predict_button_id', 'n_clicks')])
def predict(checklist_items,input_val,n_clicks):
    if (input_val is None):
        raise PreventUpdate

    df=pd.DataFrame([input_val.split(",")], columns=checklist_items)
    return "prediction is: " + str(round(model_pipeline.predict(df)[0],2))



if __name__ == '__main__':
    app.run_server(debug=True,port=5001)
