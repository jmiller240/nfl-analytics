

import plotly.graph_objects as go

dict = {
    "layout": {
        "font": {"family": "Helvetica", "size": 10, "color": "#323232"},
        "paper_bgcolor": "#fafafa",
        "plot_bgcolor": "white",
        # "plot_bgcolor": "rgba(0,0,0,0)",
        "height": 500,
        "width": 700,
        "margin": {
            "t": 75,
            "l": 50,
            "r": 25,
            "b": 50
        },
        "title": {
            "font": {"size": 18, "color": "#1a1a1a"},
            "xref": "container",
            "x": 0.07,
            "yref": "container",
            "y": 0.95},
        "xaxis": {
            "visible": True,
            "zeroline": False,
            "linecolor": "#f0f0f0",
            "linewidth": 1,
            "mirror": True,
            "gridcolor": "#d1d1d1", 
            "griddash": "solid",
            "gridwidth": 1,
            "ticks": "",
            "title": {
                "font": {"weight": "bold"},
                "standoff": 5
            },
        },
        "yaxis": {
            "visible": True,
            "zeroline": False,
            "linecolor": "#f0f0f0", 
            "linewidth": 1,
            "mirror": True,
            "gridcolor": "#d1d1d1", 
            "griddash": "solid",
            "gridwidth": 1,
            "ticks": "",
            "title": {
                "font": {"weight": "bold"},
                "standoff": 5
            },
        },
        "colorway": ["#636efa", "#ef553b", "#00cc96", "#ab63fa", "#ffa15a"], # Custom color palette
    },
    "data": {
        "scatter": [
            {"marker": {"symbol": "circle", "size": 8}},
            {"line": {"width": 2}},
        ],
        "bar": [
            {"marker": {"line": {"width": 1, "color": "#ffffff"}}},
        ],
    },
}



nfl_template = go.layout.Template(dict)
# xaxis = go.layout.XAxis(
#     linecolor='#f0f0f0',
#     linewidth=1,
#     mirror=True,
#     gridcolor='red'
# )
# nfl_template.layout.update(xaxis=xaxis)
