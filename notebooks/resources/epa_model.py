

from pprint import pprint
import math
from datetime import datetime

import pandas as pd
import numpy as np

import nfl_data_py as nfl

from resources.get_nfl_data import get_team_info, get_matchups, get_weeks, get_pbp_data



''' Constants / Parameters  '''

## Parameters
INPUT_YEARS = [i for i in range(2025, 2026)]

FEATURE_TYPE = 'EPA / Play'
LAST_N_WEEKS = [4,8,12,16]

## Constants ## 

EPA_COLS = []
EPA_PLAY_COLS = []
for n in LAST_N_WEEKS:
    for unit in ['O', 'D', 'ST']:
        EPA_COLS.append(f'Last_{n}_EPA_{unit}')
        EPA_PLAY_COLS.append(f'Last_{n}_EPA_{unit}_Play')

FEATURE_COLS = EPA_PLAY_COLS if FEATURE_TYPE == 'EPA / Play' else EPA_COLS
FEATURES = [f'Home_Team_{col}' for col in FEATURE_COLS] + [f'Away_Team_{col}' for col in FEATURE_COLS]


''' Helpers '''

def get_week_epa_inputs(master_epa_df: pd.DataFrame, teams: list, master_week: int):
    
    # Start return df
    teams_df = pd.DataFrame(data={'team': teams}).set_index('team')

    # Sum up EPA and Plays for each team and last n games
    for team in teams:
        team_sl = master_epa_df.loc[master_epa_df.index.get_level_values(1) == team, :]

        for n in [4,8,12,16]:
            sl = team_sl.loc[(team_sl.index.get_level_values(0) < master_week),:].tail(n)
            # if team == 'IND' and n == 4 and master_week == :
            #     print(sl.head().to_string())
                
            for unit in ['O', 'D', 'ST']:
                epa = sl[f'EPA_{unit}'].sum()
                plays = sl[f'Plays_{unit}'].sum()

                teams_df.loc[team, f'Last_{n}_EPA_{unit}'] = epa
                teams_df.loc[team, f'Last_{n}_EPA_{unit}_Play'] = epa / plays

    teams_df = teams_df.reset_index()

    return teams_df


''' Main '''

def get_epa_model_inputs(prediction_season: int, prediction_week: int):
    '''
    Get inputs for epa predictions models

    Returns
    -------
    '''


    ''' Import / Process Data '''

    ## Download ##

    # Team info
    team_data = get_team_info()

    # Matchups
    master_matchups_df = get_matchups(years=INPUT_YEARS)

    # PBP
    pbp_data = get_pbp_data(years=INPUT_YEARS)

    # Weeks
    master_weeks = get_weeks(years=INPUT_YEARS)
    print(master_weeks.head().to_string())

    # Add week back to matchup
    master_matchups_df = master_matchups_df.merge(master_weeks, left_on=['season', 'week'], right_on=['season', 'week'])

    print(master_weeks.shape)
    print(master_weeks.tail().to_string())
    print(master_matchups_df.shape)
    print(master_matchups_df.loc[(master_matchups_df['season'] == prediction_season) & (master_matchups_df['week'] == prediction_week),:].to_string())


    ''' Establish some variables '''

    C_MASTER_WEEK = master_weeks.loc[(master_weeks['season'] == prediction_season) & (master_weeks['week'] == prediction_week), 'master_week'].values[0]
    INPUT_WEEKS = master_weeks.loc[(master_weeks['season'] >= 2019) & (master_weeks['master_week'] <= C_MASTER_WEEK), 'master_week'].unique().tolist()

    input_matchups = master_matchups_df.loc[master_matchups_df['master_week'].isin(INPUT_WEEKS),:]

    print(C_MASTER_WEEK)
    print(INPUT_WEEKS)
    print(input_matchups.head(2).to_string())
    print(input_matchups.tail(2).to_string())



    ''' Calculate Weekly EPA '''

    pbp_adv_slice_nonst = pbp_data.loc[(pbp_data['Offensive Snap']) & (~pbp_data['Is Special Teams Play']), :]
    pbp_adv_slice_st = pbp_data.loc[~pbp_data['Is Special Teams Play'], :]

    # Offense
    offense_epa = pbp_adv_slice_nonst.groupby(['season', 'week', 'posteam']).aggregate(
        Plays_O=('posteam', 'size'),
        EPA_O=('epa', 'sum')
    )
    offense_epa['EPA_O_Play'] = offense_epa['EPA_O'] / offense_epa['Plays_O']
    offense_epa.index = offense_epa.index.set_names('team', level='posteam')

    # Defense
    defense_epa = pbp_adv_slice_nonst.groupby(['season', 'week', 'defteam']).aggregate(
        Plays_D=('posteam', 'size'),
        EPA_D=('epa', 'sum')
    )
    defense_epa['EPA_D'] = -1 * defense_epa['EPA_D']
    defense_epa['EPA_D_Play'] = defense_epa['EPA_D'] / defense_epa['Plays_D']
    defense_epa.index = defense_epa.index.set_names('team', level='defteam')

    # ST
    special_teams_epa = pbp_adv_slice_st.groupby(['season', 'week', 'posteam']).aggregate(
        Opp=('defteam', 'first'),
        POS_Plays_ST=('posteam', 'size'),
        POS_EPA_ST=('epa', 'sum')
    )

    def get_def_plays(row):
        seas = row.name[0]
        w = row.name[1]
        opp = row['Opp']
        return special_teams_epa.loc[(seas, w, opp), 'POS_Plays_ST']

    def get_def_epa(row):
        seas = row.name[0]
        w = row.name[1]
        opp = row['Opp']
        return -1*special_teams_epa.loc[(seas, w, opp), 'POS_EPA_ST']

    special_teams_epa['DEF_Plays_ST'] = special_teams_epa.apply(lambda x: get_def_plays(x), axis=1)
    special_teams_epa['DEF_EPA_ST'] = special_teams_epa.apply(lambda x: get_def_epa(x), axis=1)

    special_teams_epa['Plays_ST'] = special_teams_epa['POS_Plays_ST'] + special_teams_epa['DEF_Plays_ST']
    special_teams_epa['EPA_ST'] = special_teams_epa['POS_EPA_ST'] + special_teams_epa['DEF_EPA_ST']
    special_teams_epa['EPA_ST_Play'] = special_teams_epa['EPA_ST'] / special_teams_epa['Plays_ST']

    special_teams_epa.index = special_teams_epa.index.set_names('team', level=2)

    # Combine
    master_epa_df = offense_epa.merge(defense_epa, left_index=True, right_index=True)
    master_epa_df = master_epa_df.merge(special_teams_epa, left_index=True, right_index=True).reset_index()

    master_epa_df = master_epa_df.merge(master_weeks, left_on=['season', 'week'], right_on=['season', 'week'], how='left')
    print(master_epa_df.loc[(master_epa_df['season'] == 2025) & (master_epa_df['week'] == prediction_week - 1),:].head().to_string())


    ''' Reshape dfs '''

    master_weeks = master_weeks.set_index(['season', 'week'])
    master_epa_df = master_epa_df.set_index(['master_week', 'team'])

    print(master_weeks.tail().to_string())
    print(master_matchups_df.loc[master_matchups_df['master_week'] == 129,:].to_string())
    print(master_epa_df.loc[master_epa_df.index.get_level_values(0) == 129,:].to_string())



    ''' Forge Historical Weekly EPA Inputs for Historical Matchups '''

    ## EPA Inputs
    epa_inputs_df = pd.DataFrame(columns=['master_week', 'team'] + EPA_COLS + EPA_PLAY_COLS)

    for week in INPUT_WEEKS:

        home_teams = input_matchups.loc[input_matchups['master_week'] == week, 'home_team'].unique().tolist()
        away_teams = input_matchups.loc[input_matchups['master_week'] == week, 'away_team'].unique().tolist()
        
        df = get_week_epa_inputs(master_epa_df=master_epa_df, teams=home_teams+away_teams, master_week=week)
        df['master_week'] = week

        epa_inputs_df = pd.concat([epa_inputs_df, df])

    epa_inputs_df = epa_inputs_df.reset_index(drop=True)

    ## Add back to input matchups df

    # Home team EPA
    rename_dict = {col: f'Home_Team_{col}' for col in EPA_COLS + EPA_PLAY_COLS}
    input_matchups = input_matchups.merge(epa_inputs_df, left_on=['master_week', 'home_team'], right_on=['master_week', 'team'], how='left').rename(columns=rename_dict).drop(columns='team')

    # Away team EPA
    rename_dict = {col: f'Away_Team_{col}' for col in EPA_COLS + EPA_PLAY_COLS}
    input_matchups = input_matchups.merge(epa_inputs_df, left_on=['master_week', 'away_team'], right_on=['master_week', 'team'], how='left').rename(columns=rename_dict).drop(columns='team')

    print(input_matchups.loc[input_matchups['master_week'] == C_MASTER_WEEK, :].to_string())


    ''' Return '''

    input_matchups_prev = input_matchups.loc[input_matchups['master_week'] < C_MASTER_WEEK, :].copy()
    input_matchups_pred = input_matchups.loc[input_matchups['master_week'] == C_MASTER_WEEK, :].copy()

    return input_matchups_prev, input_matchups_pred