
nfl_template = {
    "layout": {
        "font": {"family": "Helvetica", "size": 10, "color": "#000000"},
        # "paper_bgcolor": "#f0f0f0",
        "paper_bgcolor": "white",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "height": 500,
        "width": 900,
        "margin": {
            "t": 75,
            "l": 50,
            "r": 50,
            "b": 50
        },
        "title": {
            "font": {"size": 18, "weight": "bold", "color": "#1a1a1a"},
            "xref": "container",
            "x": 0.07,
            "yref": "container",
            "y": 0.95},
        "xaxis": {
            "zeroline": False,
            "gridcolor": "#d1d1d1", 
            "linecolor": "#d1d1d1", 
            "griddash": "solid",
            "gridwidth": 1,
            "ticks": "",
            "tickcolor": "#d1d1d1",
            "title": {
                "font": {"weight": "bold"},
                "standoff": 5
            },
        },
        "yaxis": {
            "zeroline": False,
            "gridcolor": "#d1d1d1", 
            "linecolor": "#d1d1d1", 
            "griddash": "solid",
            "gridwidth": 1,
            "ticks": "",
            "tickcolor": "#d1d1d1",
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
