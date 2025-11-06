

import pandas as pd

import nfl_data_py as nfl
from .get_nfl_data import get_team_info



def get_player_stats(pbp_data: pd.DataFrame) -> pd.DataFrame:

    # player_info = nfl.import_players().set_index(['latest_team', 'short_name']).rename_axis(index={'latest_team': 'team', 'short_name': 'player'})
    player_info = nfl.import_players() #.set_index('gsis_id').rename_axis(index={'gsis_id': 'player_id'})
    team_info = get_team_info()

    ## Run / Pass ##
    run_data = pbp_data.loc[pbp_data['rush'] == 1, :]
    pass_data = pbp_data.loc[pbp_data['pass'] == 1, :]

    ## Passing ##
    by_passer = pass_data.groupby(['posteam', 'passer']).aggregate(
        player_id=('passer_player_id', 'first'),
        Plays=('pass', 'sum'),
        Attempts=('pass_attempt', 'sum'),
        Completions=('complete_pass', 'sum'),
        Yards=('passing_yards', 'sum'),
        TDs=('touchdown', 'sum'),
        INTs=('interception', 'sum'),
        Sacks=('sack', 'sum'),
        SackYards=('yards_gained', lambda x: x[pass_data['sack'] == 1].sum()),
        FirstDowns=('first_down', 'sum'),
        Successes=('success', 'sum'),
        EPA=('epa', 'sum'),
    )
    by_passer['Attempts'] = by_passer['Attempts'] - by_passer['Sacks']

    by_passer['Yds / Att'] = round(by_passer['Yards'] / by_passer['Attempts'], 2)
    by_passer['Success Rate'] = round((by_passer['Successes'] / by_passer['Plays']) * 100, 2)
    by_passer['EPA / Play'] = round((by_passer['EPA'] / by_passer['Plays']), 2)
    by_passer['1D Rate'] = round((by_passer['FirstDowns'] / by_passer['Attempts']) * 100, 2)
    by_passer['TD Rate'] = round((by_passer['TDs'] / by_passer['Attempts']) * 100, 2)


    ## Receiving ##
    by_receiver = pass_data.groupby(['posteam', 'receiver']).aggregate(
        player_id=('receiver_player_id', 'first'),
        Plays=('pass', 'sum'),
        Targets=('pass_attempt', 'sum'),
        Receptions=('complete_pass', 'sum'),
        Yards=('yards_gained', 'sum'),
        TDs=('touchdown', 'sum'),
        FirstDowns=('first_down', 'sum'),
        Successes=('success', 'sum'),
        EPA=('epa', 'sum'),
    ).sort_values(by=['posteam', 'Targets'], ascending=False)

    by_receiver['Yds / Rec'] = round(by_receiver['Yards'] / by_receiver['Receptions'], 2)
    by_receiver['Success Rate'] = round((by_receiver['Successes'] / by_receiver['Plays']) * 100, 2)
    by_receiver['EPA / Play'] = round((by_receiver['EPA'] / by_receiver['Plays']), 2)
    by_receiver['1D Rate'] = round((by_receiver['FirstDowns'] / by_receiver['Plays']) * 100, 2)
    by_receiver['TD Rate'] = round((by_receiver['TDs'] / by_receiver['Plays']) * 100, 2)


    ## Rushing ##
    by_rusher = run_data.groupby(['posteam', 'rusher']).aggregate(
        player_id=('rusher_player_id', 'first'),
        Plays=('rush', 'sum'),
        Attempts=('rush_attempt', 'sum'),
        Yards=('rushing_yards', 'sum'),
        TDs=('touchdown', 'sum'),
        FirstDowns=('first_down', 'sum'),
        Successes=('success', 'sum'),
        EPA=('epa', 'sum'),
    ).sort_values(by=['posteam', 'Attempts'], ascending=False)

    by_rusher['Yds / Att'] = round(by_rusher['Yards'] / by_rusher['Attempts'], 2)
    by_rusher['Success Rate'] = round((by_rusher['Successes'] / by_rusher['Attempts']) * 100, 2)
    by_rusher['EPA / Play'] = round((by_rusher['EPA'] / by_rusher['Plays']), 2)
    by_rusher['1D Rate'] = round((by_rusher['FirstDowns'] / by_rusher['Attempts']) * 100, 2)
    by_rusher['TD Rate'] = round((by_rusher['TDs'] / by_rusher['Attempts']) * 100, 2)

    ## Combine ##
    passer_sl = by_passer.rename_axis(index={'posteam': 'team', 'passer': 'player'})
    passer_sl.columns = ['Passing ' + col for col in passer_sl.columns]

    receiver_sl = by_receiver.rename_axis(index={'posteam': 'team', 'receiver': 'player'})
    receiver_sl.columns = ['Receiving ' + col for col in receiver_sl.columns]

    rusher_sl = by_rusher.rename_axis(index={'posteam': 'team', 'rusher': 'player'})
    rusher_sl.columns = ['Rushing ' + col for col in rusher_sl.columns]

    player_epa = passer_sl.merge(receiver_sl, left_index=True, right_index=True, how='outer')
    player_epa = player_epa.merge(rusher_sl, left_index=True, right_index=True, how='outer')

    # Player info
    player_epa['player_id'] = player_epa['Passing player_id']
    player_epa.loc[player_epa['player_id'].isna(), 'player_id'] = player_epa['Receiving player_id']
    player_epa.loc[player_epa['player_id'].isna(), 'player_id'] = player_epa['Rushing player_id']

    # Totals
    player_epa = player_epa.fillna(0)
    for col in ['Plays', 'Yards', 'TDs', 'FirstDowns', 'EPA', 'Successes']:
        player_epa[f'Total {col}'] = player_epa[f'Passing {col}'] + player_epa[f'Receiving {col}'] + player_epa[f'Rushing {col}']

        if col != 'Plays':
            player_epa[f'{col} / Play'] = player_epa[f'Total {col}'] / player_epa[f'Total Plays']
    
    player_epa = player_epa.rename(columns={
        'TDs / Play': 'TD Rate',
        'Successes / Play': 'Success Rate',
        'FirstDowns / Play': '1D Rate',
    })
    player_epa = player_epa.sort_values(by='Total EPA', ascending=False)

    # Logos / Headshots / Colors
    player_epa['team_logo_espn'] = player_epa.index.get_level_values(0).map(team_info['team_logo_espn'])
    player_epa['team_color'] = player_epa.index.get_level_values(0).map(team_info['team_color'])

    player_epa = player_epa.reset_index()
    player_epa = player_epa.merge(player_info[['gsis_id', 'position', 'headshot']], left_on='player_id', right_on='gsis_id', how='left').drop(columns=['gsis_id'])
    player_epa = player_epa.set_index(['team', 'player'])

    return player_epa