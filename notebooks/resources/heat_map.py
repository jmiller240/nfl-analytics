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



def heat_map(x: list[str], 
             y: list, 
             z: list, 
             text: list,
             title: str,
             color_name: str,
             color_scale: str = 'Greens') -> go.Figure:
    
    # Heatmap
    fig = go.Figure()

    fig.add_trace(
        go.Heatmap(
            x=x,
            y=y,
            z=z,
            text=text,
            texttemplate="%{text}",
            textfont={"size":10},
            colorscale=color_scale,
            colorbar=dict(
                title=dict(
                    text=color_name,
                    side='right',
                    font=dict(weight='bold')
                ),
                tickformat='0.0%',
            ),
            ygap=1,
            xgap=1
        )      
    )
    fig.update_xaxes(
        side="top",
        # tickfont=dict(textcase='upper'),
    )
    fig.update_yaxes(
        # tickfont=dict(textcase='upper')
    )
    fig.update_layout(
        template='nfl_template',
        title=dict(
            text=title,
        ),
        yaxis=dict(
            tickformat=".0%",
            categoryorder='array',
            # categoryarray=sl.index,
            autorange='reversed'
        ),
        height=500, width=900
    )
    
    # Credits
    fig.add_annotation(
        text=f'Figure: @clankeranalytic | Data: nfl_data_py | {datetime.today():%Y-%m-%d}',
        showarrow=False,
        xref='paper',
        yref='paper',
        y=-0.15, 
        x=1,
        align='left'
    )

    return fig