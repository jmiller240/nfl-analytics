'''
Jack Miller
Oct 2025

Download and process data from nfl_data_py
'''


from pandas import DataFrame

import nfl_data_py as nfl



''' Constants '''

PLAY_TYPES = ['GAME_START', 'KICK_OFF', 'PENALTY', 'PASS', 'RUSH', 'PUNT', 'FIELD_GOAL', 'SACK',\
            'END_QUARTER', 'TIMEOUT', 'UNSPECIFIED', 'XP_KICK', 'INTERCEPTION', 'PAT2', 'END_GAME', \
            'COMMENT', 'FUMBLE_RECOVERED_BY_OPPONENT', 'FREE_KICK']
PLAY_TYPES_SPECIAL = ['KICK_OFF', 'PAT2', 'PUNT', 'FIELD_GOAL', 'XP_KICK']
NON_PLAY_TYPES = ['GAME_START','END_QUARTER', 'TIMEOUT', 'END_GAME', 'COMMENT', 'FREE_KICK']


''' Helpers '''

## Yard Thresholds
def distance_range(down, yds):
    
    down_s = ''
    match down:
        case 1:
            # down_s = '1st'
            return '1st'
        case 2:
            down_s = '2nd'
        case 3:
            down_s = '3rd'
        case 4:
            down_s = '4th'
        case default:
            return ''
        
    yds_range = ''
    if yds <= 2:
        yds_range = 'Short'
    elif yds <= 6:
        yds_range = 'Medium'
    else:
        yds_range = 'Long'

    return f'{down_s} & {yds_range}'



''' Main '''


def get_team_info() -> DataFrame:
    current_teams = ['ARI', 'NO', 'BUF', 'BAL', 'JAX', 'CAR', 'CIN', 'CLE', 'DAL', 'PHI', 'GB', 'DET', 
                     'HOU', 'LA', 'KC', 'LAC', 'LV', 'NE', 'IND', 'MIA', 'MIN', 'CHI', 'WAS', 'NYG', 
                     'NYJ', 'PIT', 'SEA', 'SF', 'ATL', 'TB', 'TEN', 'DEN']
    
    ## Download ##
    team_data = nfl.import_team_desc().set_index('team_abbr')
    team_data = team_data.copy()

    ## Filter ##
    team_data = team_data.loc[team_data.index.isin(current_teams), :]

    return team_data


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
    pbp_data: DataFrame = nfl.import_pbp_data(years, downcast=True)
    pbp_data = pbp_data.copy()

    ## Modifications ##

    # Replace old team names
    for col in ['home_team', 'away_team', 'posteam', 'defteam']:
        pbp_data[col] = pbp_data[col].replace('OAK', 'LV')
    
    ## Add columns ##

    # Non-play types
    conditions = (~pbp_data['play_type'].isna()) &\
                (~pbp_data['play_type'].isin(['qb_kneel', 'qb_spike'])) &\
                (pbp_data['timeout'] == 0) &\
                (~pbp_data['play_type_nfl'].isin(NON_PLAY_TYPES))
    pbp_data['Non-Play Type'] = conditions

    # Snaps
    pbp_data['Offensive Snap'] = (((pbp_data['pass'] == 1) | (pbp_data['rush'] == 1)) & (~pbp_data['epa'].isna()))

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

    # Down & Distance
    pbp_data['Down & Distance'] = pbp_data.apply(lambda x: distance_range(x['down'], x['ydstogo']), axis=1)


    ## Filter ##

    # Regular season
    pbp_data = pbp_data.loc[pbp_data['season_type'] == 'REG', :]


    return pbp_data
