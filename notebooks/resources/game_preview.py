
''' IMPORTS '''

import pandas as pd
import numpy as np
from scipy import stats
import math
import requests
from io import BytesIO
from datetime import datetime
import os

import nfl_data_py as nfl
from scipy.stats import percentileofscore, rankdata

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import plotly.colors as cl
from plotly.subplots import make_subplots

from PIL import Image

from resources.tier_chart import tier_chart
from resources.plotly_theme import nfl_template
from resources.get_nfl_data import get_team_info, get_player_info, get_pbp_data, get_matchups
from resources.team_stats import get_team_stats

pio.templates['nfl_template'] = nfl_template



## Constants ##

LEAGUE_LOGO = 'https://raw.githubusercontent.com/nflverse/nflverse-pbp/master/NFL.png'
PLOT_BUFFER = 0.1

VISUALS_FOLDER = '/Users/jmiller/Documents/Fun/nfl/visuals'


''' Helpers '''

pass_len_mapper = {
    'Short': '0 to 10 yds',
    'Medium': '10 to 20 yds',
    'Long': '20+ yds'
}
def pass_len_mapper_func(pass_len):
    return pass_len_mapper[pass_len]


''' Functions '''

def team_form(team_data: pd.DataFrame, league_data: pd.DataFrame, home_team: str, away_team: str, week: int, export: bool = False) -> go.Figure:
    ''' Team Form '''

    ## Data ##
    league_adv_offense_pbp = league_data.loc[(league_data['Offensive Snap']) &
                                         (~league_data['Is Special Teams Play']), :].copy()

    matchup_teams = [away_team, home_team]

    league_av_epa = league_adv_offense_pbp['epa'].mean()
    print(f'League av EPA / Play: {league_av_epa:,.2f}')

    for team in matchup_teams:
        for unit in ['posteam', 'defteam']:
            opp_unit = 'defteam' if unit == 'posteam' else 'posteam'

            # Wrangle
            team_sl = league_adv_offense_pbp.loc[league_adv_offense_pbp[unit] == team, ['game_id', 'start_time', 'posteam', 'defteam', 'epa']]

            # Rolling EPA
            team_sl['Rolling EPA / Play'] = team_sl.rolling(window=75, closed='left')['epa'].mean()

            # Set play numbers
            team_sl = team_sl.reset_index(drop=True)
            team_sl.index = team_sl.index + 1

            # Filter to Last 8 games
            last_8_games = team_sl[['game_id', 'start_time']].drop_duplicates().tail(8)['game_id'].tolist()
            team_sl = team_sl.loc[team_sl['game_id'].isin(last_8_games), :].copy()
            play_num_range = team_sl.index.max() - team_sl.index.min()

            # Team info
            team_sl = team_sl.merge(team_data[['team_color', 'team_logo_espn']], 
                                left_on=opp_unit, right_index=True, how='left').rename(columns={
                                    'team_color': 'opp_color',
                                    'team_logo_espn': 'opp_logo'
                                })
            
            # Data
            x = team_sl.index
            y = team_sl['Rolling EPA / Play'].to_numpy()
            opp = team_sl[opp_unit].to_numpy()
            colors = team_sl['opp_color'].tolist()
            color_map = team_data['team_color'].to_dict()
            team_wordmark = team_data.loc[team, 'team_wordmark']

            # Games
            games = team_sl['game_id'].unique().tolist()
            opp_logos = team_sl.drop_duplicates(subset='game_id')['opp_logo'].tolist()
            opp_colors = team_sl.drop_duplicates(subset='game_id')['opp_color'].tolist()
            game_endpoints = []
            game_midpoints = []
            for g in games:
                sl = team_sl.loc[team_sl['game_id'] == g, :]
                midpoint_play = (sl.index[-1] + sl.index[0]) / 2
                game_midpoints.append(midpoint_play)
                game_endpoints.append((sl.index[0],sl.index[-1]))

            # Figure
            fig = px.line(
                data_frame=team_sl,
                x=x,
                y=y,
                color_discrete_sequence=['#323232']
            )

            fig.add_hline(y=league_av_epa, line_width=1, line_dash="dash", line_color="#323232", layer='above',
                        annotation=dict(text='League avg', font=dict(color='white', size=10), yanchor='bottom', xanchor='left'), annotation_position='left')

            # Game formatting
            for i in range(len(games)):
                # Opp logo
                fig.add_layout_image(
                    x=game_midpoints[i],
                    y=1,
                    sizex=play_num_range*.04,
                    sizey=play_num_range*.04,
                    xanchor='center',
                    yanchor='middle',
                    xref='x', 
                    yref='paper',
                    source=opp_logos[i],
                )
                # Opp color background
                fig.add_shape(
                    type="rect",
                    xref='x',
                    x0=game_endpoints[i][0],  # x-coordinate of the left edge
                    x1=game_endpoints[i][1],  # x-coordinate of the right edge
                    yref='paper',
                    y0=0,  # y-coordinate of the bottom edge
                    y1=1,  # y-coordinate of the top edge
                    fillcolor=opp_colors[i],  # Color to fill the rectangle
                    opacity=0.6,  # Opacity of the fill color
                    line=dict(color="white", width=1),  # Line properties for the border
                    layer="below"  # Place the shape below the traces
                )
                # Number of plays
                fig.add_annotation(
                    text=f'{game_endpoints[i][1] - game_endpoints[i][0]} Plays',
                    font=dict(color='white', size=10),
                    xref='x', 
                    yref='paper',
                    x=game_midpoints[i],
                    y=.975,
                    align='center',
                    showarrow=False
                )

            # Team wordmark
            response = requests.get(team_wordmark)
            logo_img = Image.open(BytesIO(response.content))
            fig.add_layout_image(
                x=0.5,
                y=1.1,
                sizex=.15,
                sizey=.15,
                xanchor='center',
                yanchor='middle',
                xref='paper', 
                yref='paper',
                source=logo_img,
            )

            fig.update_traces(line=dict(width=3))
            fig.update_yaxes(
                linecolor='#f0f0f0', mirror=True,
                title='Rolling EPA / Play',
                tickformat='.2f',
                title_standoff=1,
                range=[-0.6, 0.6] if unit == 'posteam' else [0.6, -0.6],
            )
            fig.update_xaxes(
                linecolor='#f0f0f0', mirror=True,
                title='Play #',
                tickformat=',',
                showgrid=False,
                title_standoff=1
            )
            title = f'Team Form: {"Offense" if unit == "posteam" else "Defense"}'
            fig.update_layout(
                title=f'<b>{title}</b><br><sup>75-play rolling EPA / Play; last 8 games</sup>',
                template='nfl_template',
                showlegend=False,
                margin=dict(t=75, b=40)
            )
            # Credits
            fig.add_annotation(
                text=f'Figure: @clankeranalytic | Data: nfl_data_py | {datetime.today().strftime("%Y-%m-%d")}',
                showarrow=False,
                xref='paper',
                yref='paper',
                y=-0.09, 
                x=1,
                align='right'
            )

            # fig.show()

            if export: 
                pio.write_image(fig, f'{VISUALS_FOLDER}/week {week}/game preview/{home_team}-{away_team}/{title} - {team} - Week {week} - {home_team} vs {away_team}.png',
                                    scale=6, width=900, height=500)


def pass_rush_down_distance_tendencies(team_data: pd.DataFrame, league_data: pd.DataFrame, home_team: str, away_team: str, week: int, export: bool = False) -> go.Figure:
    ''' Pass / Rushing by Down & Distance '''

    matchup_teams = [away_team, home_team]

    ## Data (filter to Normal Game State) ## 

    # Filter 1st downs to only 1st and 10
    league_pbp_normal_gs = league_data.loc[~((league_data['down'] == 1) & (league_data['ydstogo'] != 10)),:].copy()
    league_pbp_normal_gs['Down & Distance'] = league_pbp_normal_gs['Down & Distance'].str.replace('1st & Long', '1st & 10')

    # Only normal snaps, Q1-3 and within 2 tds, NO 4th downs
    league_pbp_normal_gs = league_pbp_normal_gs.loc[(~league_pbp_normal_gs['Is Special Teams Play']) &
                                (league_pbp_normal_gs['qtr'] <= 3) & 
                                (league_pbp_normal_gs['score_differential'] <= 14) &
                                (league_pbp_normal_gs['down'] != 4),:]

    league_pbp_normal_gs['Is 1st Down'] = league_pbp_normal_gs['down'] == 1

    # Groupby down / distance
    down_distance_gpby = league_pbp_normal_gs.groupby(['posteam', 'down', 'Down & Distance']).aggregate(
        Plays=('Offensive Snap', 'sum'),
        Pass=('pass', 'sum'),
        Rush=('rush', 'sum'),
    ).rename_axis(index={'posteam': 'team'})
    down_distance_gpby['% Pass'] = down_distance_gpby['Pass'] / down_distance_gpby['Plays']
    down_distance_gpby['% Rush'] = down_distance_gpby['Rush'] / down_distance_gpby['Plays']
    down_distance_gpby['Diff'] = down_distance_gpby['% Pass'] - down_distance_gpby['% Rush']

    down_distance_gpby = down_distance_gpby.merge(team_data['team_logo_espn'], left_index=True, right_index=True)
    # print(down_distance_gpby.head().to_string())

    ## Figure ##

    # Data
    # data = down_distance_gpby.loc[(down_distance_gpby.index.get_level_values('down') != 4) &
    #                             (down_distance_gpby.index.get_level_values(2) != 'Total'), :]
    # x = data['% Pass'].tolist()
    # y = data.index.get_level_values(2).tolist()
    # logos = data['team_logo_espn'].tolist()
    # teams = data.index.get_level_values(0).tolist()

    x = down_distance_gpby['% Pass'].tolist()
    y = down_distance_gpby.index.get_level_values('Down & Distance').tolist()
    logos = down_distance_gpby['team_logo_espn'].tolist()
    teams = down_distance_gpby.index.get_level_values('team').tolist()

    y_order = ['1st & 10']
    for down in ['2nd', '3rd']:
        for distance in ['Short', 'Medium', 'Long']:
            y_order.append(f'{down} & {distance}')

    # Plot
    dot_plot = px.scatter(
        x=x,
        y=y,
    )

    # Init figure
    fig = go.Figure()

    for trace in dot_plot.data:
        fig.add_trace(trace)

    # Logos
    for i in range(len(logos)):
        if math.isnan(x[i]): 
            continue
        op = 1 if teams[i] in matchup_teams else 0.4
        size = 0.6 if teams[i] in matchup_teams else 0.4
        layer = 'above' if teams[i] in matchup_teams else 'below'
        fig.add_layout_image(
            source=logos[i],  # The loaded image
            xref="x",    # Reference x-coordinates to the x-axis
            yref="y",    # Reference y-coordinates to the y-axis
            x=x[i], # X-coordinate of the image's center
            y=y[i], # Y-coordinate of the image's center
            sizex=size,   # Width of the image in data units
            sizey=size,   # Height of the image in data units
            xanchor="center", # Anchor the image by its center horizontally
            yanchor="middle", # Anchor the image by its middle vertically
            layer=layer, # Place image above other plot elements
            opacity=op
        )

    # Format
    fig.update_traces(marker=dict(opacity=0))

    fig.update_xaxes(
        title=dict(
            text=f'<span style="font-size: 10px"><-- More Run Heavy</span>       <b>% Pass</b>       <span style="font-size: 10px">More Pass Heavy --></span>',
            font=dict(weight='normal')
        ),
        tickformat='.0%',
        range=[-0.05, 1.05],
        dtick=.1,
        linecolor='#f0f0f0', mirror=True,
    )
    fig.update_yaxes(
        categoryorder="array", 
        categoryarray=y_order,
        autorange='reversed',
        linecolor='#f0f0f0', mirror=True,
        showgrid=False,
    )
    fig.update_layout(
        template='nfl_template',
        title=f'<b>Pass / Rush Tendencies by Down & Distance</b><br><sup>"Normal" game state: qtrs 1-3, score within 14 pts</sup>',
        margin=dict(t=50, l=75, b=60),
    )

    # Credits
    fig.add_annotation(
        text=f'Figure: @clankeranalytic | Data: nfl_data_py | {datetime.today().strftime("%Y-%m-%d")}',
        showarrow=False,
        xref='paper',
        yref='paper',
        y=-0.125, 
        x=1,
        align='right'
    )

    # fig.show()

    if export: 
        pio.write_image(fig, f'{VISUALS_FOLDER}/week {week}/game preview/{home_team}-{away_team}/Pass Rush Tendencies by Down & Distance - Week {week} - {home_team} vs {away_team}.png',
                            scale=6, width=900, height=500)


def viewing_guide_offensive_tendencies(player_info: pd.DataFrame, team_data: pd.DataFrame, league_data: pd.DataFrame, home_team: str, away_team: str, week: int, export: bool = False) -> go.Figure:
    ''' Viewing Guide - Offensive Tendencies'''
    
    matchup_teams = [away_team, home_team]
    colors = [team_data.loc[team_data.index == away_team, 'team_color'].values[0], team_data.loc[team_data.index == home_team, 'team_color'].values[0]]
    wordmarks = [team_data.loc[team_data.index == away_team, 'team_wordmark'].values[0], team_data.loc[team_data.index == home_team, 'team_wordmark'].values[0]]

    ## Data ##

    ## QB Positions
    qb_pos_order = ['Under Center', 'Shotgun', 'Pistol']
    
    qb_pos = league_data.groupby(['posteam', 'QB Position']).aggregate(
        Plays=('QB Position', 'size'),
        Rush=('rush', 'sum'),
        Pass=('pass', 'sum'),
        Pure_Rush = ('rush', lambda x: x[(league_data['is_rpo'] == False)].sum()),
        RPO_Rush = ('rush', lambda x: x[(league_data['is_rpo'] == True)].sum()),
        RPO_Pass = ('pass', lambda x: x[(league_data['is_rpo'] == True)].sum()),
        PA_Pass = ('pass', lambda x: x[(league_data['is_rpo'] == False) & (league_data['is_play_action'] == True)].sum()),
        Pure_Pass = ('pass', lambda x: x[(league_data['is_rpo'] == False) & (league_data['is_play_action'] == False)].sum()),
        Yards=('yards_gained', 'sum'),
        PassYards=('passing_yards', 'sum'),
        RushYards=('rushing_yards', 'sum'),
    )
    qb_pos['% Plays'] = qb_pos['Plays'] / qb_pos.groupby(level=0)['Plays'].transform('sum')

    # Reindex
    qb_pos = qb_pos.reindex(labels=qb_pos_order, level='QB Position')

    # Names
    play_type_cols = ['Pure_Rush', 'RPO_Rush', 'RPO_Pass', 'PA_Pass', 'Pure_Pass']
    qb_pos = qb_pos.rename(columns={col: col.replace('_', ' ') for col in play_type_cols})
    play_type_cols = [col.replace('_', ' ') for col in play_type_cols]

    # print(qb_pos.loc[qb_pos.index.get_level_values(0) == 'IND',:].to_string())

    ## Play Types
    play_types = qb_pos.melt(
        ignore_index=False,
        value_vars=play_type_cols,
        var_name='Play Type',
        value_name='# Plays'
    ).set_index('Play Type', append=True)
    play_types['% Pos Plays'] = play_types['# Plays'] / play_types.groupby(level=['posteam', 'QB Position'])['# Plays'].transform('sum')

    # Sort / Reindex
    play_types = play_types.sort_index()
    play_types = play_types.reindex(labels=qb_pos_order, level='QB Position')
    play_types = play_types.reindex(labels=play_type_cols, level='Play Type')

    # print(play_types.loc[play_types.index.get_level_values(0) == 'IND',:].head().to_string())


    ## Pass Locations
    pass_loc_order = ['Short Left', 'Short Middle', 'Short Right', 'Medium Left', 'Medium Middle', 'Medium Right', 'Long Left', 'Long Middle', 'Long Right']

    # Location by team
    by_pass_loc = league_data.groupby(['posteam', 'Pass Location']).aggregate(     #'pass length', 'pass_location'
        Plays=('pass', 'sum'),
        Yards=('passing_yards', 'sum'),
        FirstDowns=('first_down', 'sum'),
        Successes=('success', 'sum'),
        EPA=('epa', 'sum')
    )
    by_pass_loc['% Plays'] = by_pass_loc['Plays'] / by_pass_loc.groupby(level=0)['Plays'].sum()
    by_pass_loc['Success Rate'] = by_pass_loc['Successes'] / by_pass_loc['Plays']
    by_pass_loc['EPA / Play'] = by_pass_loc['EPA'] / by_pass_loc['Plays']

    # Target leaders
    pass_loc_targets = league_data.groupby(['posteam', 'receiver', 'Pass Location']).aggregate(
        Player_ID=('receiver_player_id', 'first'),
        Plays=('pass', 'sum'),
        Yards=('passing_yards', 'sum'),
        FirstDowns=('first_down', 'sum'),
        Successes=('success', 'sum'),
        EPA=('epa', 'sum')
    )
    pass_loc_targets['% Plays'] = pass_loc_targets['Plays'] / pass_loc_targets.groupby(level=0)['Plays'].sum()
    pass_loc_targets['Success Rate'] = pass_loc_targets['Successes'] / pass_loc_targets['Plays']
    pass_loc_targets['EPA / Play'] = pass_loc_targets['EPA'] / pass_loc_targets['Plays']

    target_leaders = pass_loc_targets.groupby(level=[0,2])['Plays'].aggregate(['idxmax', 'max'])
    target_leaders['Target Leader'] = target_leaders['idxmax'].str[1]
    target_leaders['Targets'] = target_leaders['max']

    # Combine
    by_pass_loc = by_pass_loc.merge(target_leaders[['Target Leader', 'Targets']], left_index=True, right_index=True)

    team_player_ids = pass_loc_targets.groupby(level=[0,1])['Player_ID'].first().reset_index()
    team_player_ids = team_player_ids.merge(player_info[['gsis_id', 'headshot']], left_on='Player_ID', right_on='gsis_id', how='left').drop(columns=['gsis_id'])
    # print(team_player_ids.head().to_string())
    by_pass_loc = by_pass_loc.reset_index().merge(team_player_ids, left_on=['posteam', 'Target Leader'], right_on=['posteam', 'receiver'], how='left').set_index(['posteam', 'Pass Location'])

    by_pass_loc['text'] = 'EPA / Play:<br>' + by_pass_loc['EPA / Play'].round(2).astype(str) + '<br><br>' + 'Target Leader:<br>' + by_pass_loc['Target Leader'] + ' (' + by_pass_loc['Targets'].astype(int).astype(str) + ')'
    by_pass_loc['text2'] = 'Target Leader:<br>' + by_pass_loc['Target Leader'] + ' (' + by_pass_loc['Targets'].astype(int).astype(str) + ')'
    by_pass_loc['text3'] = (by_pass_loc['% Plays'] * 100).round(0).astype(int).astype(str) + '%<br>' + 'Target Leader:<br>' + by_pass_loc['Target Leader'] + ' (' + by_pass_loc['Targets'].astype(int).astype(str) + ')'

    by_pass_loc = by_pass_loc.reset_index().set_index('posteam')
    by_pass_loc['Side'] = by_pass_loc['Pass Location'].str.split(' ').str[1]
    by_pass_loc['Depth'] = by_pass_loc['Pass Location'].str.split(' ').str[0]

    by_pass_loc = by_pass_loc.set_index(['Depth', 'Side'], append=True)
    by_pass_loc = by_pass_loc.reindex(labels=['Short', 'Medium', 'Long'], level='Depth')
    by_pass_loc = by_pass_loc.reindex(labels=['Left', 'Middle', 'Right'], level='Side')

    # print(by_pass_loc.loc[by_pass_loc.index.get_level_values(0) == 'IND',:].to_string())

    ## Team Quarterbacks

    # Starter
    # TODO

    # Leading passer
    team_qbs = league_data.groupby(['posteam', 'passer'])['posteam'].size()
    team_qbs = team_qbs.groupby(level='posteam').aggregate(['idxmax', 'max'])
    team_qbs['Leading Passer'] = team_qbs['idxmax'].str[1]
    team_qbs['Plays'] = team_qbs['max']
    team_qbs = team_qbs.drop(columns=['idxmax', 'max']).reset_index()

    p = player_info.reset_index()[['short_name', 'latest_team', 'headshot']]
    team_qbs = team_qbs.merge(p, left_on=['posteam', 'Leading Passer'], right_on=['latest_team', 'short_name'], how='left')

    # print(team_qbs.loc[team_qbs['posteam'].isin(MATCHUP_TEAMS)].to_string())


    ## Run Locations
    run_loc_order = ['L END', 'LT', 'LG', 'C', 'RG', 'RT', 'R END']

    by_run_loc = league_data.groupby(['posteam', 'Run Location']).aggregate(        #'run loc full'
        Plays=('rush', 'sum'),
        Yards=('rushing_yards', 'sum'),
        FirstDowns=('first_down', 'sum'),
        Successes=('success', 'sum'),
        EPA=('epa', 'sum'),
        Stuffs=('rush', lambda x: x[(league_data['rushing_yards'] <= 0)].sum())
    )
    by_run_loc['% Plays'] = by_run_loc['Plays'] / by_run_loc.groupby(level=0)['Plays'].sum()
    by_run_loc['Success Rate'] = by_run_loc['Successes'] / by_run_loc['Plays']
    by_run_loc['EPA / Play'] = by_run_loc['EPA'] / by_run_loc['Plays']
    by_run_loc['Stuff Rate'] = by_run_loc['Stuffs'] / by_run_loc['Plays']

    by_run_loc = by_run_loc.reindex(labels=run_loc_order, level='Run Location')

    # print(by_run_loc.loc[by_run_loc.index.get_level_values(0) == 'IND',:].to_string())


    ## Figure ##

    ## Charts
    play_type_color_map = {}
    scale = ['white', colors[0]]
    for i in range(len(play_type_cols)):
        play_type_color_map[play_type_cols[i]] = px.colors.qualitative.T10[i]

    pies = []
    bars = []
    pass_loc_heatmaps = []
    pass_loc_texts = []
    pass_loc_xs = []
    pass_loc_ys = []
    pass_loc_headshots = []
    run_loc_heatmaps = []

    for i in range(len(matchup_teams)):
        team = matchup_teams[i]

        # QB Position
        team_qb_pos_sl = qb_pos.loc[qb_pos.index.get_level_values('posteam') == team, :]
        
        qb_loc_pie = go.Pie(
            labels=team_qb_pos_sl.index.get_level_values('QB Position'),
            values=team_qb_pos_sl['Plays'],
            hole=0.6,
            opacity=1,
            marker=dict(line=dict(width=2, color='#323232')),
            legend='legend',
            legendgroup='QB Position'
        )

        # Play Types
        team_play_types_sl = play_types.loc[play_types.index.get_level_values('posteam') == team, :]

        team_bars = []
        for play_type in play_type_cols:
            play_type_sl = team_play_types_sl.loc[team_play_types_sl.index.get_level_values('Play Type') == play_type,:]

            x = play_type_sl['% Pos Plays'].to_numpy()
            y = play_type_sl.index.get_level_values('QB Position').to_numpy()
            text=play_type_sl['% Pos Plays'].to_numpy()
            pattern = '+' if play_type in ['RPO Rush', 'RPO Pass', 'PA Pass'] else ''

            team_bar = go.Bar(
                x=x,
                y=y,
                text=text,
                texttemplate='%{text:.0%}',
                insidetextanchor="middle",
                name=play_type,
                marker=dict(pattern=dict(shape=pattern, size=3), color=play_type_color_map[play_type], line=dict(width=2, color='#323232')),
                legend='legend2',
                legendgroup='Play Type',
                orientation='h',
                opacity=1
            )
            team_bars.append(team_bar)

        # Pass loc
        team_pass_loc = by_pass_loc.loc[by_pass_loc.index.get_level_values(0) == team, :]

        pass_loc_x = team_pass_loc.index.get_level_values('Side').to_numpy()
        pass_loc_y = team_pass_loc.index.get_level_values('Depth').to_numpy()
        pass_loc_y = list(map(pass_len_mapper_func, pass_loc_y))
        pass_loc_z = team_pass_loc['% Plays'].to_numpy()
        pass_loc_text = team_pass_loc['text2'].to_numpy()
        headshots = team_pass_loc['headshot'].to_numpy()
        
        coloraxis = 'coloraxis' if i == 0 else 'coloraxis2'
        pass_loc = go.Heatmap(
            x=pass_loc_x, 
            y=pass_loc_y, 
            z=pass_loc_z,
            coloraxis=coloraxis,#'coloraxis',
            xgap=1, ygap=1
        )

        # Run Loc
        run_loc_sl = by_run_loc.loc[by_run_loc.index.get_level_values('posteam') == team,:]
        run_loc_sl = run_loc_sl.reindex(labels=run_loc_order, level=1)

        x = run_loc_sl.index.get_level_values('Run Location').tolist()
        y = run_loc_sl.index.get_level_values('posteam').unique().tolist()
        z = run_loc_sl[['% Plays']].transpose().to_numpy()
        text = z

        run_loc = go.Heatmap(
            x=x,
            y=y,
            z=z,
            coloraxis=coloraxis, #'coloraxis2',
            text=z,
            texttemplate='%{text:.0%}',
            xgap=1, ygap=1
        )

        # Lists
        pies.append(qb_loc_pie)

        bars.append(team_bars)

        pass_loc_heatmaps.append(pass_loc)
        pass_loc_texts.append(pass_loc_text)
        pass_loc_xs.append(pass_loc_x)
        pass_loc_ys.append(pass_loc_y)
        pass_loc_headshots.append(headshots)

        run_loc_heatmaps.append(run_loc)


    ## Create Figure
    N_COLS = 2
    N_ROWS = 4

    H_SPACING = 0.2 / N_COLS
    V_SPACING = 0.06

    row_heights = [2.5,3,5,1]
    col_avail_width = 1 - (H_SPACING * (N_COLS-1))
    col_width = col_avail_width / N_COLS

    fig = make_subplots(rows=N_ROWS, cols=N_COLS, 
                        specs=[[{'type': 'domain'}, {'type': 'domain'}], [{'type': 'xy'}, {'type': 'xy'}], [{'type': 'xy'}, {'type': 'xy'}], [{'type': 'xy'}, {'type': 'xy'}]], 
                        horizontal_spacing=H_SPACING, vertical_spacing=V_SPACING,
                        row_heights=row_heights,
                        print_grid=True)

    # Pies
    for i in range(len(pies)):
        fig.add_trace(
            pies[i],
            row=1, col=1+i
        )

        
    fig.update_traces(sort=False, marker=dict(colors=px.colors.qualitative.G10), row=1)

    # Bars
    for i in range(len(bars)):
        if i == 1:
            for bar in bars[i]:
                bar.update(showlegend=False)

        fig.add_traces(
            data=bars[i],
            rows=2, cols=1+i
        )

    # Pass locs
    for i in range(len(pass_loc_heatmaps)):
        x = pass_loc_xs[i]
        y = pass_loc_ys[i]
        xref = 'x3' if i == 0 else 'x4'
        yref = 'y3' if i == 0 else 'y4'

        # Heatmap
        fig.add_trace(
            pass_loc_heatmaps[i],
            row=3, col=1+i
        )

        # Text
        text = pass_loc_texts[i]
        for t in range(len(text)):
            fig.add_annotation(
                x=x[t],
                y=y[t],
                xref=xref,
                yref=yref,
                yanchor='bottom',
                xanchor='center',
                showarrow=False,
                align='left',
                text=text[t],
                font=dict(size=10),
            )

        # Headshots
        headshots = pass_loc_headshots[i]
        for h in range(len(headshots)):
            fig.add_layout_image(
                source=headshots[h],
                xref=xref, 
                yref=yref,
                x=x[h],
                y=y[h],
                sizex=0.6,
                sizey=0.6,
                xanchor='center',
                yanchor='top',
            )

    # Run Locs
    for i in range(len(run_loc_heatmaps)):
        fig.add_trace(
            run_loc_heatmaps[i],
            row=4, col=1+i
        )

    # Wordmarks
    unit_size = (1 - H_SPACING) / 2
    for i in range(len(wordmarks)):
        response = requests.get(wordmarks[i])
        logo_img = Image.open(BytesIO(response.content))
        fig.add_layout_image(
            x=(unit_size / 2) + ((unit_size+H_SPACING)*i),
            y=1.05,
            sizex=.25,
            sizey=.25,
            xanchor='center',
            yanchor='middle',
            xref='paper', 
            yref='paper',
            source=logo_img,
        )

    # Row titles
    total_height = sum(row_heights)

    avail_pct = 1 - (V_SPACING*(N_ROWS-1))
    current_y = 1

    row_titles = [None, 'Play Types', 'Pass Locations', 'Run Locations']
    for ix, val in enumerate(row_heights):
        prop = val / total_height
        pct = prop * avail_pct
        
        y = current_y - (pct / 2)
        
        if row_titles[ix]:
            fig.add_annotation(
                text=row_titles[ix],
                font=dict(weight='bold', size=14),
                textangle=-90,
                showarrow=False,
                xref='paper', yref='paper',
                xanchor='center', yanchor='middle',
                x=0.5, y=y
            )

        # Team QB
        if ix == 0:
            for t in range(2):
                response = requests.get(team_qbs.loc[team_qbs['posteam'] == matchup_teams[t], 'headshot'].values[0])
                headshot = Image.open(BytesIO(response.content))

                headshot_x = (col_width / 2) + ((col_width+H_SPACING)*t)
                fig.add_layout_image(
                    source=headshot,
                    xref='paper', 
                    yref='paper',
                    x=headshot_x,
                    y=y,
                    sizex=0.1,
                    sizey=0.1,
                    xanchor='center',
                    yanchor='middle',
                )

        current_y -= (pct + V_SPACING)

    ## Formatting
    fig.update_xaxes(
        linecolor='#f0f0f0', mirror=True
    )
    fig.update_yaxes(
        linecolor='#f0f0f0', mirror=True
    )
    fig.update_xaxes(
        row=2,
        tickformat='.0%',
        range=[-.01, 1.05],
    )
    fig.update_yaxes(
        row=2,
        autorange='reversed',
    )

    # By Rows / Cols
    fig.update_xaxes(
        col=1,
        showticklabels=True,
    )
    fig.update_yaxes(
        col=1,
        showticklabels=True,
    )
    fig.update_xaxes(
        col=2,
        showticklabels=True,
    )

    fig.update_yaxes(
        col=2,
        showticklabels=False,
    )

    # Heatmaps formatting
    for row in [3,4]:
        fig.update_xaxes(
            row=row,
            showgrid=False,
        )
        fig.update_yaxes(
            row=row,
            showgrid=False,
        )

    fig.update_yaxes(
        row=4,
        showticklabels=False
    )

    width=1000
    height=1250
    fig.update_layout(
        template='nfl_template',
        title=dict(
            text=f'<b style="font-size: 24px;">Viewing Guide: Offensive Tendencies</b><br><span style="font-size: 16px;">Week {week}: {home_team} vs. {away_team}</span>',
            yref='container', y=0.97,
            x=0.5
        ),
        barmode='stack',
        coloraxis=dict(
            colorbar=dict(
                title=dict(
                text='% Attempts',
                font=dict(weight='bold')
                ),
                tickformat='.0%',
                dtick=0.1,
                len=0.25,
                y=0.35,
                xanchor='right',
                yanchor='middle',
                x=-0.05,
                xref='paper', yref='paper',
            ),
            cmin=0,
            colorscale=['white', colors[0]]
        ),
        coloraxis2=dict(
            colorbar=dict(
                title=dict(
                text='% Attempts',
                font=dict(weight='bold')
                ),
                tickformat='.0%',
                dtick=0.1,
                len=0.25,#0.15,
                y=0.35,#0.035,
                x=1.01,
                xanchor='left',
                yanchor='middle',
                xref='paper', yref='paper',
            ),
            cmin=0,
            colorscale=['white', colors[1]]
        ),
        legend=dict(
            x=1.01, y=0.95,
            xref='paper', yref='paper',
            xanchor='left',
            yanchor='middle',
            title=dict(text='QB Position', font=dict(weight='bold')),
            traceorder='reversed',
        ),
        legend2=dict(
            x=1.01, y=0.675,
            xref='paper', yref='paper',
            xanchor='left',
            yanchor='middle',
            title=dict(text='Play Type', font=dict(weight='bold')),
            traceorder='reversed',
        ),
        width=width, height=height,
        margin_pad=5,
        margin=dict(l=125, r=125, t=150, b=100),
        autosize=True,
    )


    # Credits
    fig.add_annotation(
        text=f'Figure: @clankeranalytic | Data: nfl_data_py | {datetime.today().strftime("%Y-%m-%d")}',
        showarrow=False,
        xref='paper',
        yref='paper',
        y=-0.075, 
        x=1,
        align='right'
    )

    # fig.show()

    if export:
        pio.write_image(fig, f'{VISUALS_FOLDER}/week {week}/game preview/{home_team}-{away_team}/Viewing Guide Offensive Tendencies - Week {week} - {home_team} vs {away_team}.png', format='png', 
                    scale=6, width=width, height=height)


def drives_per_game(team_data: pd.DataFrame, league_data: pd.DataFrame, home_team: str, away_team: str, week: int, export: bool = False) -> go.Figure:
    ''' Drives per Game '''
    # TODO - trim end of game non-drives

    matchup_teams = [away_team, home_team]
    colors = [team_data.loc[team_data.index == away_team, 'team_color'].values[0], team_data.loc[team_data.index == home_team, 'team_color'].values[0]]

    ## Data ##
    league_data['master-drive'] = league_data['game_id'] + league_data['drive'].astype(str)

    drives_per_game = league_data.groupby(['posteam', 'game_id']).aggregate(
        Drives=('master-drive', 'nunique')
    )

    sl = drives_per_game.loc[drives_per_game.index.get_level_values(0).isin(matchup_teams),:]
    away_avg = sl.loc[sl.index.get_level_values(0) == away_team, 'Drives'].mean()
    home_avg = sl.loc[sl.index.get_level_values(0) == home_team, 'Drives'].mean()
    league_avg = drives_per_game['Drives'].mean()

    fig = px.histogram(
        x=sl['Drives'].to_numpy(),
        color=sl.index.get_level_values('posteam'),
        barmode='stack',
        opacity=0.8,
        text_auto=True,
        color_discrete_map={
            away_team: colors[0],
            home_team: colors[1],
        }
    )

    fig.add_annotation(
        text=f'{away_team} avg: {away_avg:,.1f}',
        font=dict(size=14, color='#323232'),
        xref='paper', yref='paper', xanchor='left',
        x=0.1, y=0.9,
        showarrow=False,
    )
    fig.add_annotation(
        text=f'{home_team} avg: {home_avg:,.1f}',
        font=dict(size=14, color='#323232'),
        xref='paper', yref='paper', xanchor='left',
        x=0.1, y=0.85,
        showarrow=False,
    )
    fig.add_annotation(
        text=f'League avg: {league_avg:,.1f}',
        font=dict(size=14, color='#323232'),
        xref='paper', yref='paper', xanchor='left',
        x=0.1, y=0.80,
        showarrow=False,
    )

    fig.update_xaxes(
        linecolor='#f0f0f0', mirror=True,
        title='Drives',
        title_standoff=10
    )
    fig.update_yaxes(
        linecolor='#f0f0f0', mirror=True,
        title='Number of Games',
        title_standoff=10
    )
    fig.update_traces(insidetextanchor='middle', marker=dict(line=dict(width=2, color='white')))
    fig.update_layout(
        template='nfl_template',
        title=dict(
            text=f'<b>Drives per Game</b><br><sup>Week {week} Preview: {home_team} vs. {away_team}</sup>'
        ),
        legend=dict(
            title=None,
            font=dict(size=14, color='#323232'),
            x=0.9, xanchor='center',
            y=0.9,
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(pad=5)
    )
    # Credits
    fig.add_annotation(
        text=f'Figure: @clankeranalytic | Data: nfl_data_py | {datetime.today().strftime("%Y-%m-%d")}',
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
        pio.write_image(fig, f'{VISUALS_FOLDER}/week {week}/game preview/{home_team}-{away_team}/Drives per Game - Week {week} Preview - {home_team} vs {away_team}.png', scale=6, width=900, height=500)

    return fig

def strengths_weaknesses(team_data: pd.DataFrame, league_data: pd.DataFrame, home_team: str, away_team: str, week: int, export: bool = False) -> list[go.Figure]:
    ''' Strengths / Weaknesses '''

    matchup_teams = [away_team, home_team]
    wordmarks = [team_data.loc[team_data.index == away_team, 'team_wordmark'].values[0], team_data.loc[team_data.index == home_team, 'team_wordmark'].values[0]]

    ## Data ##

    # Offensive / Defensive Stats
    team_offense = get_team_stats(league_data, unit='offense')
    team_offense = team_offense.merge(team_data[['team_logo_espn', 'team_wordmark']], left_index=True, right_index=True)
    # print(team_offense.sort_values(by='EPA / Play', ascending=False).head().to_string())

    team_defense = get_team_stats(league_data, unit='defense')
    team_defense = team_defense.merge(team_data[['team_logo_espn', 'team_wordmark']], left_index=True, right_index=True)
    # print(team_defense.sort_values(by='EPA / Play', ascending=False).head().to_string())

    # All teams
    teams = team_offense.index.tolist()

    # Columns of interest
    pass_cols = ['Pass Yards / Play', 'Pass Success Rate', 'Explosive Pass Rate', 'Pass 1D Rate', 'Completion %', 'Sack Rate', 'INT Rate']
    rush_cols = ['Rush Yards / Play', 'Rush Success Rate', 'Explosive Rush Rate', 'Rush 1D Rate', 'Stuff Rate']
    off_cols = ['On Schedule Rate', 'Scramble Yards / Game', 'TOs / Game', 'TFLs / Game', 'Penalties / Game', 'Penalty Yards / Game', 'Third Down Conv %']
    cols = pass_cols + rush_cols + off_cols

    epa_cols = ['EPA / Play', 'Pass EPA / Play', 'Rush EPA / Play']

    # Sort order
    asc_cols = ['Stuff Rate', 'Sack Rate', 'INT Rate', 'TOs / Game', 'TFLs / Game', 'Penalties / Game', 'Penalty Yards / Game']
    desc_cols = list(filter(lambda x: x not in asc_cols, cols))

    # Display format
    col_fmt = {'Pass Yards / Play': '.1f', 'Pass Success Rate': '.1%', 'Explosive Pass Rate': '.1%', 'Pass 1D Rate': '.1%', 'Completion %': '.1%', 'Sack Rate': '.1%', 'INT Rate': '.1%', 'Rush Yards / Play': '.1f', 'Rush Success Rate': '.1%', 'Explosive Rush Rate': '.1%', 'Rush 1D Rate': '.1%', 'Stuff Rate': '.1%', 'On Schedule Rate': '.1%', 'Scramble Yards / Game': '.1f', 'TOs / Game': '.1f', 'TFLs / Game': '.1f', 'Penalties / Game': '.1f', 'Penalty Yards / Game': '.0f', 'Third Down Conv %': '.1%'}

    
    # Offense
    offense_epa_df = pd.DataFrame(index=teams)
    offense_ranks_df = pd.DataFrame(index=teams)

    for col in epa_cols:
        asc = False
        method = 'max'
        offense_epa_df[col] = team_offense[col].rank(method=method, ascending=asc)
    
    for col in cols:
        asc = False if col in desc_cols else True
        method = 'max' if col in desc_cols else 'min'
        offense_ranks_df[col] = team_offense[col].rank(method=method, ascending=asc)

    # Defense
    defense_epa_df = pd.DataFrame(index=teams)
    defense_ranks_df = pd.DataFrame(index=teams)

    for col in epa_cols:
        asc = True
        method = 'min'
        defense_epa_df[col] = team_defense[col].rank(method=method, ascending=asc)
    
    for col in cols:
        # Rank
        asc = False if col in asc_cols else True
        method = 'max' if col in asc_cols else 'min'
        defense_ranks_df[col] = team_defense[col].rank(method=method, ascending=asc)

    # print(offense_ranks_df.head().to_string())
    # print(defense_ranks_df.head().to_string())


    ## Figures ##

    figs = []
    for i in range(2):
        team1 = matchup_teams[i]
        team2 = matchup_teams[1-i]
        wordmarks = [wordmarks[i], wordmarks[1-i]]
        tables = []

        print(f' --- {team1} Offense vs {team2} Defense --- ')
        o_ranks = offense_ranks_df.loc[team1, :].sort_values()
        o_ranks_vals = team_offense.loc[team1, o_ranks.index]
        o_ranks_df = pd.DataFrame(index=o_ranks.index, data={'Rank': o_ranks.values, 'Value': o_ranks_vals})
        o_ranks_df['Value'] = o_ranks_df['Value']

        d_ranks = defense_ranks_df.loc[team2, :].sort_values()
        d_ranks_vals = team_defense.loc[team2, d_ranks.index]
        d_ranks_df = pd.DataFrame(index=d_ranks.index, data={'Rank': d_ranks.values, 'Value': d_ranks_vals})
        d_ranks_df['Value'] = d_ranks_df['Value']

        # EPA
        ranks = offense_epa_df.loc[team1, :].tolist()
        vals = team_offense.loc[team1, epa_cols].tolist()

        color_scale_len = len(px.colors.diverging.PRGn) - 1
        rank_colors = [px.colors.diverging.PRGn_r[int((r / 32)*color_scale_len)] for r in ranks]
        text_colors = []
        for r in ranks:
            if r <= 12 or r >= 22: text_colors.append('white')
            else: text_colors.append('#323232')

        tbl = go.Table(
            columnwidth=[3,1,1],
            header=dict(
                values=['Metric', 'Rank', 'Value'],
                fill_color=['#CCCCCC'],
                font=dict(weight='bold', color='#323232')
            ),
            cells=dict(
                values=[epa_cols, ranks, vals],
                format=['', '.0f', '.3f'],
                fill_color=['white', rank_colors, 'white'],
                font=dict(color=['#323232', text_colors, '#323232'])
            ),
        )
        tables.append(tbl)

        ranks = defense_epa_df.loc[team2, :].tolist()
        vals = team_defense.loc[team2, epa_cols].tolist()

        color_scale_len = len(px.colors.diverging.PRGn) - 1
        rank_colors = [px.colors.diverging.PRGn_r[int((r / 32)*color_scale_len)] for r in ranks]
        text_colors = []
        for r in ranks:
            if r <= 12 or r >= 22: text_colors.append('white')
            else: text_colors.append('#323232')

        tbl = go.Table(
            columnwidth=[3,1,1],
            header=dict(
                values=['Metric', 'Rank', 'Value'],
                fill_color=['#CCCCCC'],
                font=dict(weight='bold', color='#323232')
            ),
            cells=dict(
                values=[epa_cols, ranks, vals],
                format=['', '.0f', '.3f'],
                fill_color=['white', rank_colors, 'white'],
                font=dict(color=['#323232', text_colors, '#323232'])
            ),
        )
        tables.append(tbl)

        # Strengths
        off_strength = o_ranks_df.head()
        cols = off_strength.columns.tolist()
        vals = [off_strength.index.tolist()] + [off_strength[col].tolist() for col in cols]
        val_fmts = ['', '.0f', [col_fmt[metric] for metric in off_strength.index.tolist()]]

        ranks = off_strength['Rank'].astype(int).tolist()
        color_scale_len = len(px.colors.diverging.PRGn) - 1
        val_colors = [px.colors.diverging.PRGn_r[int((r / 32)*color_scale_len)] for r in ranks]
        text_colors = []
        for r in ranks:
            if r <= 12 or r >= 22: text_colors.append('white')
            else: text_colors.append('#323232')

        tbl = go.Table(
            columnwidth=[3,1,1],
            header=dict(
                values=['Metric'] + cols,
                fill_color=['#CCCCCC'],
                font=dict(weight='bold', color='#323232')
            ),
            cells=dict(
                values=vals,
                format=val_fmts,
                fill_color=['white', val_colors, 'white'],
                font=dict(color=['#323232', text_colors, '#323232'])
            ),
        )
        tables.append(tbl)

        def_strength = d_ranks_df.head()
        cols = def_strength.columns.tolist()
        vals = [def_strength.index.tolist()] + [def_strength[col].tolist() for col in cols]
        val_fmts = ['', '.0f', [col_fmt[metric] for metric in def_strength.index.tolist()]]

        ranks = def_strength['Rank'].astype(int).tolist()
        color_scale_len = len(px.colors.diverging.PRGn) - 1
        val_colors = [px.colors.diverging.PRGn_r[int((r / 32)*color_scale_len)] for r in ranks]
        text_colors = []
        for r in ranks:
            if r <= 12 or r >= 22: text_colors.append('white')
            else: text_colors.append('#323232')

        tbl = go.Table(
            columnwidth=[3,1,1],
            header=dict(
                values=['Metric'] + cols,
                fill_color=['#CCCCCC'],
                font=dict(weight='bold', color='#323232')
            ),
            cells=dict(
                values=vals,
                format=val_fmts,
                fill_color=['white', val_colors, 'white'],
                font=dict(color=['#323232', text_colors, '#323232'])
            ),
        )
        tables.append(tbl)

        # Weaknesses
        off_weakness = o_ranks_df.sort_values(by='Rank', ascending=False).head()
        cols = off_weakness.columns.tolist()
        vals = [off_weakness.index.tolist()] + [off_weakness[col].tolist() for col in cols]
        val_fmts = ['', '.0f', [col_fmt[metric] for metric in off_weakness.index.tolist()]]

        ranks = off_weakness['Rank'].astype(int).tolist()
        color_scale_len = len(px.colors.diverging.PRGn) - 1
        val_colors = [px.colors.diverging.PRGn_r[int((r / 32)*color_scale_len)] for r in ranks]
        text_colors = []
        for r in ranks:
            if r <= 12 or r >= 22: text_colors.append('white')
            else: text_colors.append('#323232')

        tbl = go.Table(
            columnwidth=[3,1,1],
            header=dict(
                values=['Metric'] + cols,
                fill_color=['#CCCCCC'],
                font=dict(weight='bold', color='#323232')
            ),
            cells=dict(
                values=vals,
                format=val_fmts,
                fill_color=['white', val_colors, 'white'],
                font=dict(color=['#323232', text_colors, '#323232'])
            ),
        )
        tables.append(tbl)

        def_weakness = d_ranks_df.sort_values(by='Rank', ascending=False).head()
        cols = def_weakness.columns.tolist()
        vals = [def_weakness.index.tolist()] + [def_weakness[col].tolist() for col in cols]
        val_fmts = ['', '.0f', [col_fmt[metric] for metric in def_weakness.index.tolist()]]

        ranks = def_weakness['Rank'].astype(int).tolist()
        color_scale_len = len(px.colors.diverging.PRGn) - 1
        val_colors = [px.colors.diverging.PRGn_r[int((r / 32)*color_scale_len)] for r in ranks]
        text_colors = []
        for r in ranks:
            if r <= 12 or r >= 22: text_colors.append('white')
            else: text_colors.append('#323232')

        tbl = go.Table(
            columnwidth=[3,1,1],
            header=dict(
                values=['Metric'] + cols,
                fill_color=['#CCCCCC'],
                font=dict(weight='bold', color='#323232')
            ),
            cells=dict(
                values=vals,
                format=val_fmts,
                fill_color=['white', val_colors, 'white'],
                font=dict(color=['#323232', text_colors, '#323232'])
            ),
        )
        tables.append(tbl)


        ## Figure ##

        N_ROWS = 3
        N_COLS = 2
        H_SPACING = 0.35 / N_COLS
        V_SPACING = 0.1 / N_ROWS
        row_titles = ['Overall', 'Strengths', 'Weaknesses']
        row_heights = [3,4,4]
        fig = make_subplots(rows=N_ROWS, cols=N_COLS, 
                            horizontal_spacing=H_SPACING, vertical_spacing=V_SPACING,
                            specs=[[{"type": "table"}]*N_COLS]*N_ROWS,
                            row_heights=row_heights,
                            )

        fig.add_traces(
            data=tables,
            rows=[1,1,2,2,3,3], 
            cols=[1,2,1,2,1,2],
        )

        # Row titles
        current_y = 1
        y_avail = 1 - (V_SPACING*(N_ROWS-1))
        for i in range(len(row_titles)):
            row_prop = row_heights[i] / sum(row_heights)
            row_size = y_avail * row_prop

            y = current_y - (row_size / 2)
            fig.add_annotation(
                text=row_titles[i],
                textangle=0,
                font=dict(size=14, weight='bold'),
                showarrow=False,
                xref='paper', yref='paper',
                xanchor='center', yanchor='bottom',
                x=0.5, y=y
            )
            
            current_y = current_y - row_size - V_SPACING

        # Wordmarks
        unit_size = (1 - H_SPACING) / 2
        for i in range(len(wordmarks)):
            response = requests.get(wordmarks[i])
            logo_img = Image.open(BytesIO(response.content))
            fig.add_layout_image(
                x=(unit_size / 2) + ((unit_size+H_SPACING)*i),
                y=1.075,
                sizex=.175,
                sizey=.175,
                xanchor='center',
                yanchor='middle',
                xref='paper', 
                yref='paper',
                source=logo_img,
            )
        
        # Football for possession
        img = Image.open('./static/football.png')
        fig.add_layout_image(
            x=unit_size * 0.75,
            y=1.075,
            sizex=.05,
            sizey=.05,
            xanchor='left',
            yanchor='middle',
            xref='paper',
            yref='paper',
            source=img,
        )

        fig.update_layout(
            template='nfl_template',
            title=f'<b>{team1} Offense vs. {team2} Defense</b><br><sup>Comparing strengths and weaknesses ahead of their Week {week} matchup</sup>',
            margin=dict(b=25, l=25, r=25, t=115),
            width=700,
            height=600
        )

        # Credits
        fig.add_annotation(
            text=f'Figure: @clankeranalytic | Data: nfl_data_py | {datetime.today().strftime("%Y-%m-%d")}',
            showarrow=False,
            xref='paper',
            yref='paper',
            y=-.025, 
            x=1,
            align='right'
        )
        # fig.show()
        
        # Export
        if export: 
            pio.write_image(fig, f'{VISUALS_FOLDER}/week {week}/game preview/{home_team}-{away_team}/{team1} Offense vs. {team2} Defense - Week {week}.png',
                                    scale=6, width=700, height=600)
            
        figs.append(fig)

    return figs



''' Main '''

def run_game_preview(league_data: pd.DataFrame, team_data: pd.DataFrame, player_info: pd.DataFrame, home_team: str, away_team: str, week: int):
    print(f'Generating visuals: {home_team} vs. {away_team}')

    # Create folder
    folder = f'{VISUALS_FOLDER}/week {week}/game preview/{home_team}-{away_team}'

    if not os.path.exists(folder):
        os.mkdir(folder)
    # else:
    #     print(f'{home_team}-{away_team}: already done!')
    #     return

    # Run visuals
    print(f'team form')
    team_form(team_data=team_data, league_data=league_data, home_team=home_team, away_team=away_team, week=week, export=True)
    print(f'pass / rush tendencies')
    pass_rush_down_distance_tendencies(team_data=team_data, league_data=league_data, home_team=home_team, away_team=away_team, week=week, export=True)
    print(f'viewing guide')
    viewing_guide_offensive_tendencies(player_info=player_info, team_data=team_data, league_data=league_data, home_team=home_team, away_team=away_team, week=week, export=True)
    print(f'Strengths / weaknesses')
    strengths_weaknesses(team_data=team_data, league_data=league_data, home_team=home_team, away_team=away_team, week=week, export=True)

    print(f'Done.')


def main(season: int, week: int):
    # Import data
    team_data = get_team_info()
    
    league_data = get_pbp_data(years=[season])
    league_data = league_data.loc[(league_data['week'] < week), :]

    player_info = get_player_info()

    # Get matchups
    matchups = get_matchups(years=[season])
    matchups = matchups.loc[matchups['week'] == week, ['home_team', 'away_team']].to_dict(orient='records')

    # Run for each game
    for game in matchups:
        if game['away_team'] != 'IND': continue

        run_game_preview(league_data=league_data, team_data=team_data, player_info=player_info, home_team=game['home_team'], away_team=game['away_team'], week=week)
