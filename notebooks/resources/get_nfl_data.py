'''
Jack Miller
Oct 2025

Download and process data from nfl_data_py
'''


from pandas import DataFrame
import numpy as np

import nfl_data_py as nfl



''' Constants '''

PLAY_TYPES = ['GAME_START', 'KICK_OFF', 'PENALTY', 'PASS', 'RUSH', 'PUNT', 'FIELD_GOAL', 'SACK',\
            'END_QUARTER', 'TIMEOUT', 'UNSPECIFIED', 'XP_KICK', 'INTERCEPTION', 'PAT2', 'END_GAME', \
            'COMMENT', 'FUMBLE_RECOVERED_BY_OPPONENT', 'FREE_KICK']
PLAY_TYPES_SPECIAL = ['KICK_OFF', 'PAT2', 'PUNT', 'FIELD_GOAL', 'XP_KICK']
NON_PLAY_TYPES = ['GAME_START','END_QUARTER', 'TIMEOUT', 'END_GAME', 'COMMENT', 'FREE_KICK']

FTN_COLS = ['nflverse_game_id','nflverse_play_id', 'starting_hash', 'qb_location',
       'n_offense_backfield', 'n_defense_box', 'is_no_huddle', 'is_motion',
       'is_play_action', 'is_screen_pass', 'is_rpo', 'is_trick_play',
       'is_qb_out_of_pocket', 'is_interception_worthy', 'is_throw_away',
       'read_thrown', 'is_catchable_ball', 'is_contested_ball',
       'is_created_reception', 'is_drop', 'is_qb_sneak', 'n_blitzers',
       'n_pass_rushers', 'is_qb_fault_sack']


''' Helpers '''

def distance_range(ydstogo: int):
    if ydstogo <= 3:
        return 'Short'
    elif ydstogo <= 6:
        return 'Medium'
    else:
        return 'Long'

## Yard Thresholds
def down_distance_range(down, yds):
    down_s = ''
    match down:
        case 1:
            down_s = '1st'
            # return '1st'
        case 2:
            down_s = '2nd'
        case 3:
            down_s = '3rd'
        case 4:
            down_s = '4th'
        case default:
            return ''
        
    yds_range = ''
    if yds <= 3:
        yds_range = 'Short'
    elif yds <= 6:
        yds_range = 'Medium'
    else:
        yds_range = 'Long'

    return f'{down_s} & {yds_range}'

def run_location(run_location, run_gap):
    if run_location == 'middle':
        return 'C'
    
    if run_gap == 'end':
        if run_location == 'left':
            return 'L END'
        elif run_location == 'right':
            return 'R END'
    elif run_gap == 'tackle':
        if run_location == 'left':
            return 'LT'
        elif run_location == 'right':
            return 'RT'
    elif run_gap == 'guard':
        if run_location == 'left':
            return 'LG'
        elif run_location == 'right':
            return 'RG'

def pass_length(air_yards):
    if not air_yards:
        return
    
    if air_yards <= 10:
        return 'Short'
    elif air_yards <= 20:
        return 'Medium'
    else:
        return 'Long'

def qb_position(qb_location):
    if qb_location == 'U':
        return 'Under Center'
    elif qb_location == 'S':
        return 'Shotgun'
    elif qb_location == 'P':
        return 'Pistol'


''' Main '''


def get_team_info() -> DataFrame:
    current_teams = ['ARI', 'NO', 'BUF', 'BAL', 'JAX', 'CAR', 'CIN', 'CLE', 'DAL', 'PHI', 'GB', 'DET', 
                     'HOU', 'LA', 'KC', 'LAC', 'LV', 'NE', 'IND', 'MIA', 'MIN', 'CHI', 'WAS', 'NYG', 
                     'NYJ', 'PIT', 'SEA', 'SF', 'ATL', 'TB', 'TEN', 'DEN']
    
    ## Download ##
    team_data = nfl.import_team_desc().set_index('team_abbr').rename_axis(index={'team_abbr': 'team'})
    team_data = team_data.copy()

    ## Filter ##
    team_data = team_data.loc[team_data.index.isin(current_teams), :]

    return team_data


def get_player_info() -> DataFrame:

    # Download
    player_info = nfl.import_players()

    # Cleaning
    player_info.loc[(player_info['first_name'] == 'Sam') & (player_info['last_name'] == 'Darnold'), 'short_name'] = 'S.Darnold'

    # Set index
    player_info = player_info.set_index(['latest_team', 'short_name'])

    return player_info


def get_matchups(years: list[int], include_qb: bool = False) -> DataFrame:

    # Download
    schedule_data = nfl.import_schedules(years=years).copy()

    # Some cleaning
    schedule_data = schedule_data.replace('OAK', 'LV')
    
    # Add some columns
    schedule_data['winner'] = np.where(schedule_data['result'] > 0, 1, 0)

    # Filer to desired columns / weeks
    COLS = ['game_id', 'season', 'week', 'gameday', 'gametime', 'home_team', 'away_team', 'home_score', 'away_score', 'result', 'winner', 'total', 'home_moneyline', 'away_moneyline', 'spread_line', 'away_spread_odds', 'home_spread_odds', 'total_line', 'under_odds', 'over_odds']
    if include_qb:
        COLS = COLS + ['home_qb_id', 'away_qb_id']
    FILTERS = (schedule_data['game_type'] == 'REG')

    matchups_df = schedule_data.loc[FILTERS, COLS].sort_values(by=['season', 'week', 'gameday', 'gametime']).reset_index(drop=True)

    return matchups_df

def get_weeks(years: list[int]) -> DataFrame:
    
    # Download schedules
    schedule_data = nfl.import_schedules(years=years).copy()

    # Create master
    master_weeks = schedule_data.loc[(schedule_data['game_type'] == 'REG'), ['season', 'week']].drop_duplicates()
    master_weeks = master_weeks.sort_values(by=['season', 'week'], ascending=[True, True]).reset_index(drop=True)
    master_weeks.index = master_weeks.index + 1
    master_weeks = master_weeks.reset_index(names=['master_week'])

    return master_weeks

def get_pbp_data(years: list[int]) -> DataFrame:
    '''
    Download and process play-by-play data from nfl_data_py

    Params
    ------
    years : list[int]
        years of pbp data to download

    Returns
    -------
    pandas dataframe
    '''

    ## Download ##
    pbp_data: DataFrame = nfl.import_pbp_data(years)
    pbp_data = pbp_data.reset_index(drop=True).copy()

    # Add ftn
    ftn_years = list(filter(lambda x: x >= 2022, years))
    ftn = nfl.import_ftn_data(years=ftn_years, columns=FTN_COLS)
    ftn = ftn.copy()

    pbp_data = pbp_data.merge(ftn, left_on=['game_id', 'play_id'], 
                                 right_on=['nflverse_game_id', 'nflverse_play_id'],
                                how='left')


    ## Modifications ##

    # Replace old team names
    for col in ['home_team', 'away_team', 'posteam', 'defteam']:
        pbp_data[col] = pbp_data[col].replace('OAK', 'LV')
    
    ## Add columns ##

    # Non-play types
    conditions = ((pbp_data['play_type'].notna()) &\
                (~pbp_data['play_type'].isin(['qb_kneel', 'qb_spike'])) &\
                (pbp_data['timeout'] == 0) &\
                (~pbp_data['play_type_nfl'].isin(NON_PLAY_TYPES)))
    pbp_data['Non-Play Type'] = conditions

    # Play Counted
    pbp_data['Play Counted'] = (pbp_data['penalty_team'] != pbp_data['posteam'])

    # Drive
    pbp_data['Master Drive ID'] = pbp_data['game_id'] + pbp_data['drive'].astype(str)
    
    # Snaps
    pbp_data['Offensive Snap'] = (((pbp_data['pass'] == 1) | (pbp_data['rush'] == 1)) & (pbp_data['epa'].notna()))

    # Flag for special teams
    special_conditions = ((pbp_data['play_type_nfl'].isin(PLAY_TYPES_SPECIAL)) | (pbp_data['special_teams_play'] == 1))
    pbp_data['Is Special Teams Play'] = special_conditions
    
    # Successes
    pbp_data['% ydstogo'] = pbp_data['yards_gained'] / pbp_data['ydstogo']
    pbp_data['Successful Play'] = (
        ((pbp_data['down'] == 1) & (pbp_data['% ydstogo'] >= 0.4)) |
        ((pbp_data['down'] == 2) & (pbp_data['% ydstogo'] >= 0.6)) |
        (pbp_data['first_down'] == 1) |
        (pbp_data['touchdown'] == 1)
    )

    # Explosives
    pbp_data['Explosive Play'] = np.where(pbp_data['yards_gained'] >= 15, 1, 0)

    # On schedule play
    on_schedule_conditions = (
        ((pbp_data['down'] == 1) & (pbp_data['ydstogo'] <= 10)) |
        ((pbp_data['down'] == 2) & (pbp_data['ydstogo'] <= 6)) | 
        ((pbp_data['down'] == 3) & (pbp_data['ydstogo'] <= 4)) | 
        ((pbp_data['down'] == 4) & (pbp_data['ydstogo'] <= 2))
    )
    pbp_data['On Schedule Play'] = on_schedule_conditions

    # Down Distance
    pbp_data['Distance'] = pbp_data['ydstogo'].apply(lambda x: distance_range(x))

    # Down & Distance
    pbp_data['Down & Distance'] = pbp_data.apply(lambda x: down_distance_range(x['down'], x['ydstogo']), axis=1)

    # Play locations
    pbp_data['QB Position'] = pbp_data['qb_location'].apply(lambda x: qb_position(x))

    pbp_data['Run Location'] = pbp_data.apply(lambda x: run_location(x['run_location'], x['run_gap']), axis=1)

    pbp_data['Pass Length'] = pbp_data['air_yards'].apply(lambda x: pass_length(x))
    pbp_data['Pass Location'] = pbp_data['Pass Length'] + ' ' + pbp_data['pass_location'].str.capitalize()

    ## Filter ##

    # Regular season
    pbp_data = pbp_data.loc[pbp_data['season_type'] == 'REG', :]


    return pbp_data
