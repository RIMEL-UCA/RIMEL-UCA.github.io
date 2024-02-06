import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import json
import sys
import dash_bootstrap_components as dbc



def getColorCategory(category):
    color = None
    if category == 'Beginner':
        color = 'green'
    elif category == 'Intermediate':
        color = 'orange'
    else:
        color = 'red'
    return color

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP,'./assets/style.css'])

if(len(sys.argv) != 2):
    print("Usage: python script.py nom_projet")
    sys.exit(1)

nom_du_projet = sys.argv[1]

json_file_path = f'./metrics/{nom_du_projet.split("/")[1]}--metrics.json'

with open(json_file_path, "r") as json_file:
    data = json.load(json_file)

communication_rate_map = data["communication_rate_map"]
services = list(communication_rate_map.keys())
producers = [communication_rate_map[service]["producers"] for service in services]
consumers = [communication_rate_map[service]["consumers"] for service in services]


topics_diversity = data["topics_diversity"]
labels = ["Producers", "Consumers", "Services"]
values = [data["producers_number"], data["consumers_number"], data["services_number"]]

initial_state = {f'service-{i}': False for i in range(len(data["services_names"]))}


# Création du layout du tableau de bord
app.layout = html.Div([
    html.H1(f"Metrics dashboard of {nom_du_projet}", style={'text-align': 'center', 'color': 'black'}),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Metrics Board', value='tab-1'),
        dcc.Tab(label='Information Board', value='tab-2'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        return html.Div(children=[
    html.Div([
    
    html.Div([
    html.H2("Topics diversity"),
    html.Div([
        html.Div([
            html.P(data["topics_number"], style={'margin-left': '40px', 'font-size': '50px', 'font-weight': 'bold'}),
            html.P("topic(s)", style={'position': 'relative', 'top': '30px', 'left': '10px', 'font-size': '25px'}),
            html.Hr(style={'height': '200px', 'background': 'white', 'position': 'relative', 'top': '5px', 'left': '-40px', 'transform': 'rotate(65deg)', 'border': '3!important', 'opacity': '1 !important'}),
            html.Div([
                'for',
                html.Div(data['services_number'], style={'font-size': '45px', 'margin': '0 10px'}),
                "service(s)"], style={'left': '-70px', 'top': '40px', 'position': 'relative', 'display': 'flex', 'align-items': 'center', 'font-size': '15px'}),
        ], style={'width': '350px', 'display': 'flex', 'justify-content': 'center', 'background': 'rgb(52, 73, 94)', 'color': 'white', 'border-radius': '10px'})]
                ,style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center',})]
             ,style={'display': 'flex', 'justify-content': 'space-evenly', 'align-items': 'center',"flex-direction": "column"}),
    
        html.Div([
        html.H2("Producer/Consumers Ratio"),

        html.Div([
            html.P(data["producers_number"], style={'margin-left': '40px', 'font-size': '50px', 'font-weight': 'bold'}),
            html.P("producer(s)", style={'position': 'relative', 'top': '30px', 'left': '10px', 'font-size': '25px'}),
            html.Hr(style={'height': '200px', 'background': 'white', 'position': 'relative', 'top': '5px', 'left': '-40px', 'transform': 'rotate(65deg)'}),
            html.Div([
                'for',
                html.Div(data["consumers_number"], style={'font-size': '45px', 'margin': '0 10px'}),
                "consumer(s)"], style={'left': '-70px', 'top': '40px', 'position': 'relative', 'display': 'flex', 'align-items': 'center', 'font-size': '15px'}),
        ], style={'width': '350px', 'display': 'flex', 'justify-content': 'center', 'background': 'rgb(52, 73, 94)', 'color': 'white', 'border-radius': '10px'})]
                ,style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', "flex-direction": "column"}),
    ], style={'display': 'flex', 'justify-content': 'space-evenly', 'align-items': 'center', 'margin': '30px'}),
    
    dcc.Graph(
        id='Consumer-Producer-Services-pie',
        figure={
            'data': [
                go.Pie(labels=labels, values=values, hole=0.4)
            ],
            'layout': go.Layout(
                title='Ratio of Producers, Consumers and Services',
            )
        }
    ),
    dcc.Graph(
        id='producer-consumer-bar',
        figure={
            'data': [
                go.Bar(x=services, y=producers, name='Producers'),
                go.Bar(x=services, y=consumers, name='Consumers')
            ],
            'layout': go.Layout(
                title='Number of Producers and Consumers per Service',
                xaxis=dict(title='Services'),
                yaxis=dict(title='Count'),
                barmode='group'
            )
        }
    ),
], style={'margin': '50px'})
    elif tab == 'tab-2':
        return html.Div([
            html.Div([
                html.Div([
                    html.H4("Number of producers", style={'color': 'white'}),
                    html.Div(data["producers_number"], style={'font-size': '36px', 'font-weight': 'bold', 'color': 'white'}),
                ], style={'background-color': '#34495e', 'padding': '20px', 'border-radius': '10px', 'text-align': 'center'}),
                html.Div([
                    html.H4("Number of consumers", style={'color': 'white'}),
                    html.Div(data["consumers_number"], style={'font-size': '36px', 'font-weight': 'bold', 'color': 'white'}),
                ], style={'background-color': '#34495e', 'padding': '20px', 'border-radius': '10px', 'text-align': 'center'}),
                html.Div([
                    html.H4("Number of services", style={'color': 'white'}),
                    html.Div(data["services_number"], style={'font-size': '36px', 'font-weight': 'bold', 'color': 'white'}),
                ], style={'background-color': '#34495e', 'padding': '20px', 'border-radius': '10px', 'text-align': 'center'}),
                html.Div([
                    html.H4("Number of topics", style={'color': 'white'}),
                    html.Div(data["topics_number"], style={'font-size': '36px', 'font-weight': 'bold', 'color': 'white'}),
                ], style={'background-color': '#34495e', 'padding': '20px', 'border-radius': '10px', 'text-align': 'center'}),
            ], style={'display': 'flex', 'justify-content': 'space-evenly', 'align-items': 'center'}),
                html.Div(
                    [
                        dbc.Card(
                            dbc.CardBody([
                                    html.H2("Categorization of the project", className="mb-2"),
                                    html.P([
                                        html.Span(data["categorization"]["difficulty"], style={'color': 'white', 'background-color': getColorCategory(data["categorization"]["difficulty"]), 'padding': '5px'})
                                    ])
                                ]),
                            className="mb-3 mt-3",
                        ),
                    ]
                ),
            html.Div([
                html.H2("All Services Names"),
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            [
                                html.Div(service_name)
                                for service_name in data["services_names"]
                            ],
                            title="Services",
                        ),
                    ],
                    className="accordion",
                    style={'margin-bottom': '30px'}
                ),   
                html.H2("All Topic Names"),
                dbc.Accordion(
                    [
                        dbc.AccordionItem(
                            [
                                html.Div(service_name)
                                for service_name in data["topics_names"]
                            ],
                            title="Topics",
                        ),
                    ],
                    className="accordion",
                ),
            ], style={"display": 'flex', 'justify-content': 'space-evenly', 'margin': '30px', "flex-direction": "column"})
        ],style={'margin': '50px'})
    
if __name__ == '__main__':
    app.run_server(debug=True)
