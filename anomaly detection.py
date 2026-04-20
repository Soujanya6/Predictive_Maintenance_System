#importing importent library what i use in this dashboard file

import random
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from dash import Dash,dcc,html,dash_table
from dash.dependencies import Input,Output
import plotly.graph_objs as go

from sklearn.preprocessing import StandardScaler

#Defining data columns
data = pd.DataFrame(columns=['Time', 'BlastTemp', 'BlastPressure', 'O2Percent', 'PCI_Rate',
       'TopGas_CO', 'TopGas_CO2', 'TuyereTemp', 'CoolingWater_DeltaT',
       'ShellTemp', 'Vibration' ])
# Genarating data over time
def generate_data():
    return {
        "Time": datetime.now(),
        "BlastTemp": random.uniform(1065, 1257),
        "BlastPressure": random.uniform(2, 3),
        "O2Percent": random.uniform(19, 23),
        "PCI_Rate": random.uniform(130, 170),
        "TopGas_CO": random.uniform(18, 26),
        "TopGas_CO2": random.uniform(14, 22),
        "TuyereTemp": random.uniform(1740, 2063),
        "CoolingWater_DeltaT": random.uniform(11, 28),
        "ShellTemp": random.uniform(285, 370),
        "Vibration": random.uniform(1, 7)

    }

# Process data as training 
def pre_process(x):
    td_x = x.drop(columns=["Time"])
    scaler = StandardScaler()
    scale_x = scaler.fit_transform(td_x)
    return scale_x


# Initialize dash and its layout
app = Dash(__name__)

app.layout = html.Div([

    html.H1("🔍 Anomaly Detection Dashboard", style={'textAlign': 'center'}),

   
    html.Div(id='kpi-cards', style={'display': 'flex', 'justifyContent': 'space-around'}),

    
    dcc.Graph(id='live-graph'),

    dcc.Graph(id ='plotly'),

    
    dash_table.DataTable(
        id='table',
        page_size=10,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'}
    ),

   
    dcc.Interval(id='interval', interval=2000, n_intervals=0)

])

#Defining callback for Input Output
@app.callback(
    [Output('live-graph', 'figure'),
     Output('table', 'data'),
     Output('table', 'columns'),
     Output('kpi-cards', 'children')],
    [Input('interval', 'n_intervals')]
)

# Now here is the main part there i using the model what i build to generate anomaly data. 
# After Prepocess here simple StandardScaler is used.
# and the kpi's are saying actual calculation on tha data on other site percentage of data also.
def update_dashboard(n):

    global data

    
    new_row = generate_data()
    data = pd.concat([data, pd.DataFrame([new_row])], ignore_index=True)

  
    data = data.tail(200)

    if len(data) > 20:
        model = joblib.load("/Users/soujanyadutta/Predictive_Maintenance_System/anomaly_model.pkl")
        features = pre_process(data)
        data["anomaly"] = model.fit_predict(features)
    else:
        data["anomaly"] = 1

    normal = data[data["anomaly"] == 1]
    anomaly = data[data["anomaly"] == -1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=normal["Time"], y=normal["BlastTemp"],
        mode='lines', name='Normal'
    ))

    fig.add_trace(go.Scatter(
        x=anomaly["Time"], y=anomaly["BlastTemp"],
        mode='markers', name='Anomaly',
        marker=dict(color='red', size=8)
    ))

    fig.update_layout(title="Blast Temperature with Anomalies")

    table_data = data.to_dict('records')
    columns = [{"name": i, "id": i} for i in data.columns]

    total = len(data)
    anomalies = len(anomaly)
    rate = round((anomalies / total) * 100, 2) if total > 0 else 0

    kpis = [
        html.Div([html.H4("Total"), html.H2(total)]),
        html.Div([html.H4("Anomalies"), html.H2(anomalies)]),
        html.Div([html.H4("Anomaly %"), html.H2(f"{rate}%")])
    ]

    return fig, table_data, columns, kpis

if __name__ == '__main__':
    app.run(debug=True)