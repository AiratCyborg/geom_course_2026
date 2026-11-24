"""Приложение Dash для ДЗ №5."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "homeworks" / "hw05" / "pressure_long.csv"

table = pd.read_csv(DATA_PATH)
required_columns = {
    "time_h",
    "radius_m",
    "well_id",
    "pressure_mpa",
}
assert required_columns.issubset(table.columns)

time_values = sorted(table["time_h"].unique())

app = Dash(__name__)
app.layout = html.Div(
    [
        html.H1("Давление и расстояние"),
        dcc.Slider(
            id="time-slider",
            min=0,
            max=len(time_values) - 1,
            step=1,
            value=0,
            marks={
                index: str(time)
                for index, time in enumerate(time_values)
            },
        ),
        dcc.Graph(id="pressure-graph"),
    ]
)


@app.callback(
    Output("pressure-graph", "figure"),
    Input("time-slider", "value"),
)
def update_graph(time_index):
    selected_time = time_values[time_index]
    current = table[table["time_h"] == selected_time].sort_values("radius_m")

    fig = go.Figure(
        data=[
            go.Scatter(
                x=current["radius_m"],
                y=current["pressure_mpa"],
                mode="lines+markers",
                text=current["well_id"],
                hovertemplate=(
                    "Скважина %{text}<br>"
                    "r = %{x:.2f} м<br>"
                    "p = %{y:.3f} МПа<extra></extra>"
                ),
            )
        ]
    )

    fig.update_layout(
        title=f"Время: {selected_time:.2f} ч",
        xaxis_title="Радиус, м",
        yaxis_title="Давление, МПа",
        xaxis_type="log",
        margin=dict(l=60, r=20, t=60, b=50),
    )

    return fig


if __name__ == "__main__":
    app.run(debug=True)