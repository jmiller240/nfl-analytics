'''
Jack Miller
Oct. 2025
'''

import pandas as pd
import numpy as np
from scipy import stats
import math
from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.colors as cl

from PIL import Image

## Theme
from resources.plotly_theme import nfl_template

pio.templates["nfl_template"] = nfl_template




''' Helper Funcs '''

def b_from_point_and_slope(P0, m):
    x0, y0 = P0

    # Slope-intercept form: y = mx + b
    b = y0 - m * x0
    return b



''' Charts '''


def tier_chart(data_frame: pd.DataFrame,
               x_col: str,
               y_col: str,
               logos_col: str,
               title: str,
               n_tiers: int = 6,
               x_reversed: bool = False,
               y_reversed: bool = False) -> go.Figure:
    
    ## Init
    fig = go.Figure()

    ## Inputs
    X = data_frame[x_col].to_numpy()
    Y = data_frame[y_col].to_numpy()
    LOGOS = data_frame[logos_col].to_numpy()

    x_range = X.max() - X.min()
    y_range = Y.max() - Y.min()

    ## Scatter
    fig.add_trace(
        go.Scatter(
            x=X,
            y=Y,
            mode='markers',
        ),
    )

    ## Best fit line

    # Regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
    # slope, intercept = (1, 0)
    print(f'y = {slope:.5f}x + {intercept}')

    # Calculate y-values for the regression line
    line_y = (slope * X) + intercept

    # Add the regression line
    best_fit_line = go.Scatter(
        x=X, 
        y=line_y, 
        mode='lines',
        line=dict(color='#b8b8b8', width=1.5, dash='dash')
    )
    # fig.add_trace(best_fit_line)

    ## Add tier lines
    slope_sign = -1 if slope < 0 else 1
    slope = slope_sign * (y_range / x_range)
    print(f'{x_range = }')
    print(f'{y_range = }')
    print(f'{slope = }')

    # Tier lines slope
    recip_slope = (-1/slope)
    # recip_slope = -1

    # Create evenly spaced lines through data

    first_tier = X.min() + (x_range*.1)
    last_tier = X.max() - (x_range*.1)
    tier_spacing = (last_tier - first_tier) / (n_tiers - 2)
    tiers = [first_tier+(i*tier_spacing) for i in range(n_tiers - 1)]

    for i in tiers:
        # Point on best fit line
        P0 = (i, (slope*i) + intercept)

        # Y intercept of perp. line that goes thru P0
        b = b_from_point_and_slope(P0, recip_slope)

        # Tier line
        tier_line = (recip_slope * X) + b
        # print(f'y = {recip_slope:.5f}x + {b}')

        fig.add_trace(
            go.Scatter(
                x=X, 
                y=tier_line, 
                mode='lines',
                line=dict(color='#a7a7a7', width=1.2)
            )
        )

    ## Format
    BUFFER = 0.1
    VIZ_X_RANGE = ()
    if x_reversed:
        VIZ_X_RANGE = (X.max() + (x_range*BUFFER), X.min() - (x_range*BUFFER))
    else:
        VIZ_X_RANGE = (X.min() - (x_range*BUFFER), X.max() + (x_range*BUFFER))

    VIZ_Y_RANGE = ()
    if y_reversed:
        VIZ_Y_RANGE = (Y.max() + (y_range*BUFFER), Y.min() - (y_range*BUFFER))
    else:
        VIZ_Y_RANGE = (Y.min() - (y_range*BUFFER), Y.max() + (y_range*BUFFER))

    fig.add_vline(x=X.mean(), line_width=1, line_dash="dash", line_color="#CB4747", layer='above')
    fig.add_hline(y=Y.mean(), line_width=1, line_dash="dash", line_color="#CB4747", layer='above')

    fig.update_traces(
        marker=dict(opacity=0)
    )
    fig.update_yaxes(
        title=y_col,
        range=VIZ_Y_RANGE
    )
    fig.update_xaxes(
        title=x_col,
        range=VIZ_X_RANGE
    )
    fig.update_layout(
        template="nfl_template",
        margin=dict(t=50, r=25),
        title=dict(
            text=title
        ),
        showlegend=False,
    )

    ## Logos
    logo_size_x = math.ceil((VIZ_X_RANGE[1] - VIZ_X_RANGE[0]) / 10)
    logo_size_y = math.ceil((VIZ_Y_RANGE[1] - VIZ_Y_RANGE[0]) / 10)
    print(logo_size_x)
    print(logo_size_y)

    for i in range(len(X)):
        fig.add_layout_image(
            source=LOGOS[i],  # The loaded image
            xref="x",    # Reference x-coordinates to the x-axis
            yref="y",    # Reference y-coordinates to the y-axis
            x=X[i], # X-coordinate of the image's center
            y=Y[i], # Y-coordinate of the image's center
            sizex=7,   # Width of the image in data units
            sizey=7,   # Height of the image in data units
            xanchor="center", # Anchor the image by its center horizontally
            yanchor="middle", # Anchor the image by its middle vertically
            layer="above", # Place image above other plot elements
            opacity=0.9
        )

    ## Credits
    fig.add_annotation(
        text=f'Figure: @clankeranalytic | Data: nfl_data_py | {datetime.today().strftime("%Y-%m-%d")}',
        showarrow=False,
        xref='paper',
        yref='paper',
        y=-0.1, 
        x=1,
        align='left'
    )


    return fig