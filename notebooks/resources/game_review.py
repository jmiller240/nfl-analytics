
from datetime import datetime
import requests
from io import BytesIO
from PIL import Image
import os

import pandas as pd
import numpy as np
from scipy.stats import percentileofscore

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.colors as cl
from plotly.subplots import make_subplots

from resources.plotly_theme import nfl_template
from resources.get_nfl_data import get_pbp_data, get_team_info, get_player_info, get_matchups
from resources.player_stats import get_player_stats

pio.templates['nfl_template'] = nfl_template


''' Helpers '''

VISUALS_FOLDER = '/Users/jmiller/Documents/Fun/nfl/visuals'

qtr_mapper_obj = {
    1: 'Q1',
    2: 'Q2',
    3: 'Q3',
    4: 'Q4',
    5: 'OT'
}
def qtr_mapper(down):
    return qtr_mapper_obj[down]

down_mapper_obj = {
    1: '1st',
    2: '2nd',
    3: '3rd',
    4: '4th'
}
def down_mapper(down):
    return down_mapper_obj[down]


''' Visuals '''

def production_by_down(team_data: pd.DataFrame, league_data: pd.DataFrame, home_team: str, away_team: str, week: int, export: bool = False) -> go.Figure:
    
    ## Data ##
    # League by down
    league_adv_offense = league_data.loc[(league_data['Offensive Snap']) & (~league_data['Is Special Teams Play']),:].copy()

    league_sr_down = league_adv_offense.groupby(['posteam', 'game_id', 'down']).aggregate(
        Plays=('posteam', 'size'),
        Yards=('yards_gained', 'sum'),
        Successes=('success', 'sum'),
        EPA=('epa', 'sum'),
        WPA=('wpa', 'sum')
    )
    league_sr_down['Yards / Play'] = league_sr_down['Yards'] / league_sr_down['Plays']
    league_sr_down['Success Rate'] = league_sr_down['Successes'] / league_sr_down['Plays']
    league_sr_down['EPA / Play'] = league_sr_down['EPA'] / league_sr_down['Plays']
    league_sr_down['WPA / Play'] = (league_sr_down['WPA']*100) / league_sr_down['Plays']
    league_sr_down['WPA'] = (league_sr_down['WPA']*100)

    # This game by down
    game_data = league_data.loc[(league_data['week'] == week) & (league_data['away_team'] == away_team) & (league_data['home_team'] == home_team),:]
    game_adv_offense = game_data.loc[(game_data['Offensive Snap']) & (~game_data['Is Special Teams Play']),:].copy()

    game_sr_down = game_adv_offense.groupby(['posteam', 'down']).aggregate(
        Plays=('posteam', 'size'),
        Yards=('yards_gained', 'sum'),
        Successes=('success', 'sum'),
        EPA=('epa', 'sum'),
        WPA=('wpa', 'sum')
    )
    game_sr_down['Yards / Play'] = game_sr_down['Yards'] / game_sr_down['Plays']
    game_sr_down['Success Rate'] = game_sr_down['Successes'] / game_sr_down['Plays']
    game_sr_down['EPA / Play'] = game_sr_down['EPA'] / game_sr_down['Plays']
    game_sr_down['WPA / Play'] = (game_sr_down['WPA']*100) / game_sr_down['Plays']
    game_sr_down['WPA'] = (game_sr_down['WPA']*100)

    # Percentile the down performances
    cols = ['Yards / Play', 'Success Rate', 'EPA / Play', 'WPA']
    for col in cols:
        perc_col = f'{col} Percentile'
        game_sr_down[perc_col] = 0
        for idx,row in game_sr_down.iterrows():
            down = idx[1]
            
            col_val = row[col]
            league_vals = league_sr_down.loc[league_sr_down.index.get_level_values(2) == down, col]

            percentile = percentileofscore(league_vals, col_val) / 100
            game_sr_down.loc[idx, perc_col] = percentile

    ## Figure ##
    teams = [away_team, home_team]
    wordmarks = [team_data.loc[team_data.index == teams[0], 'team_wordmark'].values[0], team_data.loc[team_data.index == teams[1], 'team_wordmark'].values[0]]
    away_score = game_data['away_score'].tail(1).values[0]
    home_score = game_data['home_score'].tail(1).values[0]

    heat_maps = []

    for team in teams:
        sl = game_sr_down.loc[game_sr_down.index.get_level_values(0) == team, :].copy()
        sl['Yards / Play'] = round(sl['Yards / Play'], 1)
        sl['Success Rate'] = round(sl['Success Rate'] * 100, 1)
        sl['WPA'] = round(sl['WPA'], 1)
        sl['EPA / Play'] = round(sl['EPA / Play'], 2)

        cols_perc = [f'{col} Percentile' for col in cols]
        x = cols
        y = list(map(lambda x: down_mapper(x), sl.index.get_level_values(1).tolist()))
        z = sl[cols_perc].values
        act_vals = sl[cols].values
        text = []
        for i in range(len(y)):
            row = []
            for j in range(len(x)):
                act_val = f'{act_vals[i][j]:.2f}' if x[j] == 'EPA / Play' else f'{act_vals[i][j]:.1f}'
                if x[j] in ['Success Rate', 'WPA']:
                    act_val += '%'
                row.append(f'<span style="font-size: 14px">{act_val}</span><br><span style="font-size: 10px">({z[i][j]:.0%})</span>')
            text.append(row)

        h = go.Heatmap(
            x=x,
            y=y,
            z=z,
            text=text,
            texttemplate="%{text}",
            textfont={"size":12},
            coloraxis='coloraxis1',
            ygap=2,
            xgap=2
        )  
        heat_maps.append(h)

    # Figure
    H_SPACING = 0.15
    fig = make_subplots(rows=2, cols=1, horizontal_spacing=H_SPACING,
                        shared_xaxes=True)

    for r in range(len(heat_maps)):
        fig.add_trace(
            heat_maps[r],
            row=r + 1, col=1,
        )

    # Wordmarks
    for i in range(len(wordmarks)):
        response = requests.get(wordmarks[i])
        logo_img = Image.open(BytesIO(response.content))

        col_wid = (1 / 2) - (H_SPACING / 2)
        fig.add_layout_image(
            x=0,
            y=1.075 - ((0.475 + H_SPACING)*i),
            sizex=.15,
            sizey=.15,
            xanchor='left',
            yanchor='bottom',
            xref='paper', 
            yref='paper',
            source=logo_img,
        )
        fig.add_annotation(
            text=f' - {away_score if i == 0 else home_score}',
            font=dict(weight='bold', size=20),
            x=.135,
            y=1.075 - ((0.475 + H_SPACING)*i),
            xref='paper', 
            yref='paper',
            showarrow=False,
            xanchor='left',
            yanchor='bottom',
        )

    fig.update_xaxes(
        side='top',
        anchor='y',
        tickfont=dict(weight='bold', size=12),
        showgrid=False,
        linecolor='#f0f0f0', mirror=True
    )
    fig.update_yaxes(
        type='category',
        showgrid=False,
        autorange='reversed',
        tickfont=dict(weight='bold', size=12),
        linecolor='#f0f0f0', mirror=True
    )
    fig.update_coloraxes(
        cmin=0, cmax=1,
        colorscale='PRGn',
        showscale=False
    )
    fig.update_layout(
        template='nfl_template',
        title=dict(
            text=f'<b>Production by Down</b><br><sup>Week {week}: {home_team} vs. {away_team}',
            x=0.5
        ),
        margin_pad=5,
        margin=dict(t=85, l=50, b=50)
    )
    # Credits
    fig.add_annotation(
        text=f'Percentile (in paren.) of all single-game offensive performances in 2025; <span style="color: rgba(64, 0, 75, 0.8); font-weight: bold;">purple</span> colors = lower percentiles, <span style="color: rgba(0, 68, 27, 0.8); font-weight: bold;">green</span> colors = higher percentiles<br>Figure: @clankeranalytic | Data: nfl_data_py | {datetime.today().strftime("%Y-%m-%d")}',
        showarrow=False,
        xref='paper',
        yref='paper',
        y=-0.1, 
        x=1,
        align='right'
    )

    ## Export ##
    if export:
        pio.write_image(fig, f'{VISUALS_FOLDER}/week {week}/game review/{home_team}-{away_team}/Production by Down - Week {week} - {home_team} vs {away_team}.png',
                                scale=6, width=900, height=500)

    return fig


def production_by_qtr(team_data: pd.DataFrame, league_data: pd.DataFrame, home_team: str, away_team: str, week: int, export: bool = False) -> go.Figure:
    
    ## Data ##
    # League by qtr
    league_adv_offense = league_data.loc[(league_data['Offensive Snap']) & (~league_data['Is Special Teams Play']),:].copy()

    league_prod_by_qrt = league_adv_offense.groupby(['posteam', 'game_id', 'qtr']).aggregate(
        Plays=('posteam', 'size'),
        Yards=('yards_gained', 'sum'),
        Successes=('success', 'sum'),
        EPA=('epa', 'sum'),
        WPA=('wpa', 'sum')
    )
    league_prod_by_qrt['Yards / Play'] = league_prod_by_qrt['Yards'] / league_prod_by_qrt['Plays']
    league_prod_by_qrt['Success Rate'] = league_prod_by_qrt['Successes'] / league_prod_by_qrt['Plays']
    league_prod_by_qrt['EPA / Play'] = league_prod_by_qrt['EPA'] / league_prod_by_qrt['Plays']
    league_prod_by_qrt['WPA / Play'] = (league_prod_by_qrt['WPA']*100) / league_prod_by_qrt['Plays']
    league_prod_by_qrt['WPA'] = (league_prod_by_qrt['WPA']*100)

    # This game by qtr
    game_data = league_data.loc[(league_data['week'] == week) & (league_data['away_team'] == away_team) & (league_data['home_team'] == home_team),:]
    game_adv_offense = game_data.loc[(game_data['Offensive Snap']) & (~game_data['Is Special Teams Play']),:].copy()

    game_prod_by_qtr = game_adv_offense.groupby(['posteam', 'qtr']).aggregate(
        Plays=('posteam', 'size'),
        Yards=('yards_gained', 'sum'),
        Successes=('success', 'sum'),
        EPA=('epa', 'sum'),
        WPA=('wpa', 'sum')
    )
    game_prod_by_qtr['Yards / Play'] = game_prod_by_qtr['Yards'] / game_prod_by_qtr['Plays']
    game_prod_by_qtr['Success Rate'] = game_prod_by_qtr['Successes'] / game_prod_by_qtr['Plays']
    game_prod_by_qtr['EPA / Play'] = game_prod_by_qtr['EPA'] / game_prod_by_qtr['Plays']
    game_prod_by_qtr['WPA / Play'] = (game_prod_by_qtr['WPA']*100) / game_prod_by_qtr['Plays']
    game_prod_by_qtr['WPA'] = (game_prod_by_qtr['WPA']*100)

    # Percentile the qtr performances
    cols = ['Yards / Play', 'Success Rate', 'EPA / Play', 'WPA']
    for col in cols:
        perc_col = f'{col} Percentile'
        game_prod_by_qtr[perc_col] = 0
        for idx,row in game_prod_by_qtr.iterrows():
            down = idx[1]
            
            col_val = row[col]
            league_vals = league_prod_by_qrt.loc[league_prod_by_qrt.index.get_level_values(2) == down, col]

            percentile = percentileofscore(league_vals, col_val) / 100
            game_prod_by_qtr.loc[idx, perc_col] = percentile


    ## Figure ##
    teams = [away_team, home_team]
    wordmarks = [team_data.loc[team_data.index == teams[0], 'team_wordmark'].values[0], team_data.loc[team_data.index == teams[1], 'team_wordmark'].values[0]]
    away_score = game_data['away_score'].tail(1).values[0]
    home_score = game_data['home_score'].tail(1).values[0]

    heat_maps = []

    for team in teams:
        sl = game_prod_by_qtr.loc[game_prod_by_qtr.index.get_level_values(0) == team, :].copy()
        sl['Yards / Play'] = round(sl['Yards / Play'], 1)
        sl['Success Rate'] = round(sl['Success Rate'] * 100, 1)
        sl['WPA'] = round(sl['WPA'], 1)
        sl['EPA / Play'] = round(sl['EPA / Play'], 2)

        cols_perc = [f'{col} Percentile' for col in cols]
        x = list(map(lambda x: qtr_mapper(x), sl.index.get_level_values(1).tolist()))
        y = cols
        z = sl[cols_perc].transpose().values
        act_vals = sl[cols].transpose().values
        text = []

        for i in range(len(y)):
            row = []
            for j in range(len(x)):
                act_val = f'{act_vals[i][j]:.2f}' if y[i] == 'EPA / Play' else f'{act_vals[i][j]:.1f}'
                if y[i] in ['Success Rate', 'WPA']:
                    act_val += '%'
                row.append(f'<span style="font-size: 14px">{act_val}</span><br><span style="font-size: 10px">({z[i][j]:.0%})</span>')
            text.append(row)

        h = go.Heatmap(
            x=x,
            y=y,
            z=z,
            text=text,
            texttemplate="%{text}",
            textfont={"size":12},
            coloraxis='coloraxis1',
            ygap=2,
            xgap=2
        )  
        heat_maps.append(h)

    # Figure
    H_SPACING = 0.15
    fig = make_subplots(rows=2, cols=1, horizontal_spacing=H_SPACING,
                        shared_xaxes=True)

    for r in range(len(heat_maps)):
        fig.add_trace(
            heat_maps[r],
            row=r + 1, col=1,
        )

    # Wordmarks
    for i in range(len(wordmarks)):
        response = requests.get(wordmarks[i])
        logo_img = Image.open(BytesIO(response.content))

        col_wid = (1 / 2) - (H_SPACING / 2)
        fig.add_layout_image(
            x=0,
            y=1.075 - ((0.475 + H_SPACING)*i),
            sizex=.15,
            sizey=.15,
            xanchor='left',
            yanchor='bottom',
            xref='paper', 
            yref='paper',
            source=logo_img,
        )
        fig.add_annotation(
            text=f' - {away_score if i == 0 else home_score}',
            font=dict(weight='bold', size=20),
            x=.135,
            y=1.075 - ((0.475 + H_SPACING)*i),
            xref='paper', 
            yref='paper',
            showarrow=False,
            xanchor='left',
            yanchor='bottom',
        )

    fig.update_xaxes(
        side='top',
        anchor='y',
        tickfont=dict(weight='bold', size=12),
        showgrid=False,
        linecolor='#f0f0f0', mirror=True
    )
    fig.update_yaxes(
        type='category',
        showgrid=False,
        autorange='reversed',
        tickfont=dict(weight='bold', size=12),
        linecolor='#f0f0f0', mirror=True
    )
    fig.update_coloraxes(
        cmin=0, cmax=1,
        colorscale='PRGn',
        showscale=False
    )
    fig.update_layout(
        template='nfl_template',
        title=dict(
            text=f'<b>Production by Quarter</b><br><sup>Week {week}: {home_team} vs. {away_team}',
            x=0.5
        ),
        margin_pad=5,
        margin=dict(t=85, l=100, b=50)
    )
    # Credits
    fig.add_annotation(
        text=f'Percentile (in paren.) of all single-game offensive performances in 2025; <span style="color: rgba(64, 0, 75, 0.8); font-weight: bold;">purple</span> colors = lower percentiles, <span style="color: rgba(0, 68, 27, 0.8); font-weight: bold;">green</span> colors = higher percentiles<br>Figure: @clankeranalytic | Data: nfl_data_py | {datetime.today().strftime("%Y-%m-%d")}',
        showarrow=False,
        xref='paper',
        yref='paper',
        y=-0.1, 
        x=1,
        align='right'
    )

    ## Export ##
    if export: 
        pio.write_image(fig, f'{VISUALS_FOLDER}/week {week}/game review/{home_team}-{away_team}/Production by Qtr - Week {week} - {home_team} vs {away_team}.png',
                                scale=6, width=900, height=500)

    return fig


def receiver_sr_down_distance(league_data: pd.DataFrame, player_info: pd.DataFrame, home_team: str, away_team: str, week: int, min_plays: int = 2, export: bool = False) -> go.Figure:
    
    ## Data ##
    DOWNS = ['1st Down', '2nd & Short', '2nd & Medium', '2nd & Long', 
        '3rd & Short', '3rd & Medium', '3rd & Long', '4th & Short', '4th & Medium']
    
    pass_data = league_data.loc[(league_data['pass'] == 1) & 
                               (league_data['week'] == week) & (league_data['away_team'] == away_team) & (league_data['home_team'] == home_team), :].copy()
    pass_data.loc[pass_data['Down & Distance'].str.contains('1st'), 'Down & Distance'] = '1st Down'

    # Group by down/distance
    receiver_sr_by_down = pass_data.groupby(['receiver', 'Down & Distance']).aggregate(
        Plays=('pass', 'sum'),
        Successes=('success', 'sum'),
        player_id=('receiver_player_id', 'first'),
    ).reset_index()
    receiver_sr_by_down['Success Rate'] = receiver_sr_by_down['Successes'] / receiver_sr_by_down['Plays']

    # Pivot so down/distance is columns
    col_sort = pd.MultiIndex.from_product([DOWNS, ['Successes', 'Plays', 'Success Rate']])

    down_distance_pivot = receiver_sr_by_down.loc[(receiver_sr_by_down['Down & Distance'] != ''),:].pivot(
            index='receiver',
            columns=['Down & Distance'],
            values=['Plays', 'Successes', 'Success Rate']
        )\
        .swaplevel(0, 1, axis=1).sort_index(axis=1).reset_index().set_index('receiver')\
        .reindex(columns=col_sort)

    # Filter pivot to just success rate
    idx = pd.IndexSlice
    viz_slice = pd.DataFrame(down_distance_pivot.loc[:, idx[:, ('Success Rate')]])
    viz_slice.columns = viz_slice.columns.get_level_values(0)

    # Merge in total attempts
    receiver_total_plays = pass_data.groupby('receiver')['pass'].sum()
    viz_slice['Plays'] = viz_slice.index.map(receiver_total_plays)
    viz_slice = viz_slice.loc[viz_slice['Plays'] >= min_plays, :].sort_values(by='Plays', ascending=False)

    # Add player info
    player_ids = receiver_sr_by_down.reset_index().groupby('receiver')['player_id'].first()
    viz_slice['player_id'] = viz_slice.index.map(player_ids)
    viz_slice['headshot'] = viz_slice['player_id'].map(player_info.set_index('gsis_id')['headshot'])

    # Annotations
    annot_df = viz_slice[DOWNS].copy() # Initialize with data, convert to string
    for r in annot_df.index:
        for c in DOWNS: #down_distance_pivot.columns.get_level_values(0).unique():
            plays = down_distance_pivot.loc[r, (c, 'Plays')].astype(float)
            successes = down_distance_pivot.loc[r, (c, 'Successes')].astype(float)
            success_rate = down_distance_pivot.loc[r, (c, 'Success Rate')].astype(float) * 100
            if pd.isna(plays):
                annot_df.loc[r,c] = ''
            else:
                annot_df.loc[r, c] = f"{success_rate:,.1f}%<br>({successes:,.0f} / {plays:,.0f})"

    ## Figure ##
    x = [down.replace(' ', '<br>') for down in DOWNS]
    y = viz_slice.index
    z = viz_slice[DOWNS].values.tolist()
    text=annot_df.values.tolist()
    headshots = viz_slice['headshot'].tolist()
    n_players = len(headshots)

    heat_map = go.Heatmap(
        x=x,
        y=y,
        z=z,
        text=text,
        texttemplate="%{text}",
        textfont={"size":10},
        colorscale=px.colors.sequential.Greens,
        ygap=1,
        xgap=1,
        showscale=False
    )

    # Figure
    fig = go.Figure(
        data=[heat_map]
    )

    # Headshots
    for i in range(n_players):
        if not headshots[i]: continue

        response = requests.get(headshots[i])
        headshot = Image.open(BytesIO(response.content))

        hs_size = (1/n_players) * 0.9
        y = (1 - ((1/n_players)/2)) - ((1/n_players)*i)
        fig.add_layout_image(
            source=headshot,
            xref='paper', 
            yref='paper',
            x=-.145,
            y=y,
            sizex=hs_size,
            sizey=hs_size,
            xanchor='center',
            yanchor='middle',
        )

    # Format
    fig.update_yaxes(
        showgrid=False,
        tickfont=dict(size=10),
        tickformat=".0%",
        categoryorder='array',
        autorange='reversed',
        linecolor='#f0f0f0', mirror=True
    )
    fig.update_xaxes(
        side='top',
        linecolor='#f0f0f0', mirror=True
    )
    fig.update_layout(
        template='nfl_template',
        title=f'<b>Receiving Success Rate by Down & Distance</b><br><sup>Week {week}: {home_team} vs. {away_team}; min {min_plays} plays</sup>',
        margin=dict(t=100, l=150, r=25, b=50, pad=5),
        coloraxis=dict(showscale=False),
        width=900,
        height=700
    )

    # Credits
    fig.add_annotation(
        text=f'<span style="font-weight: bold; color: rgb(0, 101, 41);">Darker</span> colors indicate a higher success rate, <span style="font-weight: bold; color: rgb(184, 227, 177);">lighter</span> colors a lower success rate<br>Figure: @clankeranalytic | Data: nfl_data_py | {datetime.today():%Y-%m-%d}',
        showarrow=False,
        xref='paper',
        yref='paper',
        y=-0.075, 
        x=1,
        align='right'
    )

    ## Export ##
    if export: 
        pio.write_image(fig, f"{VISUALS_FOLDER}/week {week}/game review/{home_team}-{away_team}/Receiving SR by Down & Distance - Week {week} - {home_team} vs {away_team}.png", height=700, width=900, scale=6)

    return fig

def rusher_sr_down_distance(league_data: pd.DataFrame, player_info: pd.DataFrame, home_team: str, away_team: str, week: int, min_attempts: int = 1, export: bool = False) -> go.Figure:
    
    ## Data ##
    DOWNS = ['1st Down', '2nd & Short', '2nd & Medium', '2nd & Long', 
        '3rd & Short', '3rd & Medium', '3rd & Long', '4th & Short', '4th & Medium']
    
    run_data = league_data.loc[(league_data['rush'] == 1) & 
                               (league_data['week'] == week) & (league_data['away_team'] == away_team) & (league_data['home_team'] == home_team), :].copy()
    run_data.loc[run_data['Down & Distance'].str.contains('1st'), 'Down & Distance'] = '1st Down'

    # Group by down/distance
    rusher_by_down_distance = run_data.groupby(['rusher', 'Down & Distance']).aggregate(
        Attempts=('rush_attempt', 'sum'),
        Successes=('success', 'sum'),
        player_id=('rusher_player_id', 'first'),
    ).reset_index()
    rusher_by_down_distance['Success Rate'] = rusher_by_down_distance['Successes'] / rusher_by_down_distance['Attempts']
    
    # print(rusher_by_down_distance.to_string())

    # Pivot so down/distances are columns
    col_sort = pd.MultiIndex.from_product([DOWNS, ['Successes', 'Attempts', 'Success Rate']])

    down_distance_pivot = rusher_by_down_distance.loc[(rusher_by_down_distance['Down & Distance'] != ''),:].pivot(
            index='rusher',
            columns=['Down & Distance'],
            values=['Attempts', 'Successes', 'Success Rate']
        )\
        .swaplevel(0, 1, axis=1).sort_index(axis=1).reset_index().set_index('rusher')\
        .reindex(columns=col_sort)

    # print(down_distance_pivot.to_string())

    # Filter pivot to just success rate
    idx = pd.IndexSlice
    viz_slice = pd.DataFrame(down_distance_pivot.loc[:, idx[:, ('Success Rate')]])
    viz_slice.columns = viz_slice.columns.get_level_values(0)

    # Merge in total attempts
    total_attempts_by_rusher = run_data.groupby('rusher')['rush_attempt'].sum()
    viz_slice['Attempts'] = viz_slice.index.map(total_attempts_by_rusher)

    # Filter to min attempts
    viz_slice = viz_slice.loc[viz_slice['Attempts'] >= min_attempts, :].sort_values(by='Attempts', ascending=False)

    # Add player info
    player_ids = rusher_by_down_distance.reset_index().groupby('rusher')['player_id'].first()
    # print(player_ids)
    viz_slice['player_id'] = viz_slice.index.map(player_ids)
    viz_slice['headshot'] = viz_slice['player_id'].map(player_info.set_index('gsis_id')['headshot'])
    # print(viz_slice.to_string())

    # Annotations
    annot_df = viz_slice[DOWNS].copy() # Initialize with data, convert to string
    for r in annot_df.index:
        for c in DOWNS: #down_distance_pivot.columns.get_level_values(0).unique():
            attempts = down_distance_pivot.loc[r, (c, 'Attempts')].astype(float)
            successes = down_distance_pivot.loc[r, (c, 'Successes')].astype(float)
            success_rate = down_distance_pivot.loc[r, (c, 'Success Rate')].astype(float) * 100
            if pd.isna(attempts):
                annot_df.loc[r,c] = ''
            else:
                annot_df.loc[r, c] = f"{success_rate:,.1f}%<br>({successes:,.0f} / {attempts:,.0f})"
    
    # print(annot_df.to_string())

    ## Figure ##
    
    # Heatmap
    x = [down.replace(' ', '<br>') for down in DOWNS]
    y = viz_slice.index
    z = viz_slice[DOWNS].values.tolist()    
    text=annot_df.values.tolist()
    headshots = viz_slice['headshot'].tolist()
    n_players = len(headshots)
    # print(x)
    # print(y)
    # print(z)
    # print(text)
    # print(headshots)
    # print(f'{n_players = }')

    heat_map = go.Heatmap(
        x=x,
        y=y,
        z=z,
        text=text,
        texttemplate="%{text}",
        textfont={"size":10},
        colorscale=px.colors.sequential.Greens,
        ygap=1,
        xgap=1,
        showscale=False
    )

    # Figure
    fig = go.Figure(
        data=[heat_map]
    )

    # Headshots
    # print(headshots)
    for i in range(n_players):
        if not headshots[i]: continue

        response = requests.get(headshots[i])
        headshot = Image.open(BytesIO(response.content))

        hs_size = (1/n_players) * 0.8
        y = (1 - ((1/n_players)/2)) - ((1/n_players)*i)
        fig.add_layout_image(
            source=headshot,
            xref='paper', 
            yref='paper',
            x=-.145,
            y=y,
            sizex=hs_size,
            sizey=hs_size,
            xanchor='center',
            yanchor='middle',
        )
    # print('headshots done')

    # Format
    fig.update_yaxes(
        showgrid=False,
        tickfont=dict(size=10),
        tickformat=".0%",
        categoryorder='array',
        autorange='reversed',
        linecolor='#f0f0f0', mirror=True
    )
    fig.update_xaxes(
        side='top',
        linecolor='#f0f0f0', mirror=True
    )
    fig.update_layout(
        template='nfl_template',
        title=f'<b>Rushing Success Rate by Down & Distance</b><br><sup>Week {week}: {home_team} vs. {away_team}; designed rushes (not incl. scrambles); min {min_attempts} attempts</sup>',
        margin=dict(t=100, l=150, r=25, b=50, pad=5),
        coloraxis=dict(showscale=False),
        width=900,
        height=700
    )

    # Credits
    fig.add_annotation(
        text=f'<span style="font-weight: bold; color: rgb(0, 101, 41);">Darker</span> colors indicate a higher success rate, <span style="font-weight: bold; color: rgb(184, 227, 177);">lighter</span> colors a lower success rate<br>Figure: @clankeranalytic | Data: nfl_data_py | {datetime.today():%Y-%m-%d}',
        showarrow=False,
        xref='paper',
        yref='paper',
        y=-0.075, 
        x=1,
        align='right'
    )

    # fig.show()

    ## Export ##
    if export: 
        # print('exporting')
        pio.write_image(fig, f"{VISUALS_FOLDER}/week {week}/game review/{home_team}-{away_team}/Rushing SR by Down & Distance - Week {week} - {home_team} vs {away_team}.png", 
                        scale=6, height=700, width=900)

    return fig


def epa_box_score(league_data: pd.DataFrame, team_data: pd.DataFrame, home_team: str, away_team: str, week: int, export: bool = False) -> go.Figure:
    
    ## Data ##
    game_data = league_data.loc[(league_data['week'] == week) & (league_data['away_team'] == away_team) & (league_data['home_team'] == home_team), :]
    
    player_stats = get_player_stats(pbp_data=game_data)
    player_stats = player_stats.sort_values(by='Total EPA', ascending=False)

    ## Figure ##
    teams = [away_team, home_team]
    wordmarks = [team_data.loc[team_data.index == away_team, 'team_wordmark'].values[0], team_data.loc[team_data.index == home_team, 'team_wordmark'].values[0]]
    units = ['Passing', 'Receiving', 'Rushing']

    # Create positional bars for each team
    bar_charts = []
    x_ranges = []
    for unit in units:
        plays_col = f'{unit} Plays'
        epa_col = f'{unit} EPA'

        # Get position slice
        unit_sl = player_stats.loc[player_stats[plays_col] > 0, :].copy()
        unit_sl = unit_sl.sort_values(by=plays_col, ascending=False)

        unit_sl['text'] = unit_sl[epa_col].round(2).astype(str) + ' (' + unit_sl[plays_col].astype(int).astype(str) + ')'

        # X axis range (EPA)
        max_abs_val = unit_sl[epa_col].abs().max()
        r = max_abs_val + (max_abs_val*0.05)
        x_ranges.append([-1*r, r])

        # Make a bar for both teams
        for team in teams:
            team_sl = unit_sl.loc[unit_sl.index.get_level_values(0) == team, :]

            x = team_sl[epa_col].tolist()
            y = team_sl.index.get_level_values(1)
            text = team_sl['text'].tolist()
            colors = team_sl['team_color'].tolist()

            bar = px.bar(
                x=x,
                y=y,
                text=text,
                color_discrete_sequence=colors,
                opacity=0.7
            )

            bar_charts.append(bar)

    # Create Figure
    N_ROWS = 3
    N_COLS = 2

    H_SPACING = .25/N_COLS
    V_SPACING = .2/N_ROWS
    fig = make_subplots(rows=N_ROWS, cols=N_COLS, 
                        row_heights=[1,4,3], column_widths=[1,1],
                        horizontal_spacing=H_SPACING, vertical_spacing=V_SPACING,
                        row_titles=units, x_title='Total EPA')

    fig.for_each_annotation(lambda a: a.update(x=-.135, textangle=-90, font=dict(weight='bold', size=14)) if a.text in units else())
    fig.for_each_annotation(lambda a: a.update(y=.015, font=dict(weight='bold', size=14)) if a.text == 'Total EPA' else())

    # Add bars to subplots
    i = 0
    for row in range(1, N_ROWS+1):
        for col in range(1, N_COLS+1):
            bar = bar_charts[i]
            
            for trace in bar.data:
                fig.add_trace(trace, row=row, col=col)
            
            i += 1

    # Wordmarks
    for i in range(len(wordmarks)):
        response = requests.get(wordmarks[i])
        logo_img = Image.open(BytesIO(response.content))

        col_wid = (1 / N_COLS) - (H_SPACING / 2)
        fig.add_layout_image(
            x=(col_wid / 2) if i == 0 else 1 - (col_wid / 2),
            y=1.05,
            sizex=.2,
            sizey=.2,
            xanchor='center',
            yanchor='middle',
            xref='paper', 
            yref='paper',
            source=logo_img,
        )

    # Format
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=2, color='#323232')))
    fig.update_yaxes(
        autorange='reversed',
        linecolor='#f0f0f0', mirror=True

    )
    # Ranges of each position bar
    for i in range(N_ROWS):
        rng = x_ranges[i][1] - x_ranges[i][0]
        fig.update_xaxes(
            range=x_ranges[i], 
            dtick=rng // 7, 
            linecolor='#f0f0f0', 
            mirror=True,
            row=i+1
        )
    fig.update_layout(
        template='nfl_template',
        title=dict(text=f'<b>EPA Box Score</b><br><sup>Week {week}: {home_team} vs. {away_team}</sup>'),
        height=700, width=700,
        margin=dict(t=100, r=15, b=65, l=90)
    )

    # Credits
    fig.add_annotation(
        text=f'Sorted by number of plays (bar text)<br>Figure: @clankeranalytic | Data: nfl_data_py | {datetime.today().strftime("%Y-%m-%d")}',
        showarrow=False,
        xref='paper',
        yref='paper',
        y=-0.1, 
        x=1,
        align='right'
    )

    # Export
    if export: 
        pio.write_image(fig, f'{VISUALS_FOLDER}/week {week}/game review/{home_team}-{away_team}/EPA Box Score - Week {week} - {home_team} vs {away_team}.png', scale=6, width=700, height=700)

    return fig


''' Main '''

def run_game_review(league_data: pd.DataFrame, team_data: pd.DataFrame, player_info: pd.DataFrame, home_team: str, away_team: str, week: int):
    print(f'Generating visuals: {home_team} vs. {away_team}')

    # Create folder
    folder = f'{VISUALS_FOLDER}/week {week}/game review/{home_team}-{away_team}'

    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        print(f'{home_team}-{away_team}: already done!')
        return

    # Visuals
    print(f'prod by down')
    production_by_down(team_data=team_data, league_data=league_data, home_team=home_team, away_team=away_team, week=week, export=True)
    print(f'prod by qtr')
    production_by_qtr(team_data=team_data, league_data=league_data, home_team=home_team, away_team=away_team, week=week, export=True)
    print(f'receiver sr')
    receiver_sr_down_distance(player_info=player_info, league_data=league_data, home_team=home_team, away_team=away_team, week=week, min_plays=2, export=True)
    print(f'rusher sr')
    rusher_sr_down_distance(player_info=player_info, league_data=league_data, home_team=home_team, away_team=away_team, week=week, min_attempts=1, export=True)
    print(f'epa box score')
    epa_box_score(team_data=team_data, league_data=league_data, home_team=home_team, away_team=away_team, week=week, export=True)

    print(f'Done.')


def main(season: int, week: int):
    # Import data
    team_data = get_team_info()
    
    league_data = get_pbp_data(years=[season])
    league_data = league_data.loc[(league_data['week'] <= week), :]

    player_info = get_player_info()

    # Get matchups
    matchups = get_matchups(years=[season])
    matchups = matchups.loc[matchups['week'] == week, ['home_team', 'away_team']].to_dict(orient='records')

    # Run for each game
    for game in matchups:
        # if game['home_team'] != 'ATL': continue

        run_game_review(league_data=league_data, team_data=team_data, player_info=player_info, home_team=game['home_team'], away_team=game['away_team'], week=week)
