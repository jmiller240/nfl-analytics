
import pandas as pd







def calc_offensive_stats(pbp_data: pd.DataFrame) -> pd.DataFrame:

    ## Run / Pass data ##
    run_data = pbp_data.loc[(pbp_data['play_type'] == 'run') &
                                (pbp_data['Is Special Teams Play'] == False), :]

    pass_data = pbp_data.loc[(pbp_data['play_type'] == 'pass') &
                                (pbp_data['Is Special Teams Play'] == False), :]


    ## Rushing ##

    team_rushing = run_data.groupby('posteam').aggregate(
        RushAttempts=('posteam', 'size'),
        RushYards=('yards_gained', 'sum'),
        RushTDs=('touchdown', 'sum'),
        RushFirstDowns=('first_down', 'sum'),
        RushSuccesses=('Successful Play', 'sum'),
        RushEPA=('epa', 'sum')
    ).sort_values(by='RushEPA', ascending=False)

    team_rushing['Rush Yds / Att'] = team_rushing['RushYards'] / team_rushing['RushAttempts']
    team_rushing['Rush Success Rate'] = team_rushing['RushSuccesses'] / team_rushing['RushAttempts']
    team_rushing['Rush 1D Rate'] = team_rushing['RushFirstDowns'] / team_rushing['RushAttempts']
    team_rushing['Rush TD Rate'] = team_rushing['RushTDs'] / team_rushing['RushAttempts']
    team_rushing['Rush EPA / Att'] = team_rushing['RushEPA'] / team_rushing['RushAttempts']

    # print(team_rushing.head().to_string())

    ## Passing ## 

    team_passing = pass_data.groupby('posteam').aggregate(
        PassAttempts=('pass_attempt', 'sum'),
        PassYards=('yards_gained', 'sum'),
        PassTDs=('touchdown', 'sum'),
        PassFirstDowns=('first_down', 'sum'),
        PassSuccesses=('Successful Play', 'sum'),
        Sacks=('sack', 'sum'),
        PassEPA=('epa', 'sum')
    ).sort_values(by='PassEPA', ascending=False)

    team_passing['Pass Yds / Att'] = team_passing['PassYards'] / team_passing['PassAttempts']
    team_passing['Pass Success Rate'] = team_passing['PassSuccesses'] / team_passing['PassAttempts']
    team_passing['Pass 1D Rate'] = team_passing['PassFirstDowns'] / team_passing['PassAttempts']
    team_passing['Pass TD Rate'] = team_passing['PassTDs'] / team_passing['PassAttempts']
    team_passing['Pass EPA / Att'] = team_passing['PassEPA'] / team_passing['PassAttempts']

    # print(team_passing.head().to_string())

    ## Combine ##
    team_offense = team_passing.merge(team_rushing, left_index=True, right_index=True)
    team_offense['Snaps'] = team_offense['RushAttempts'] + team_offense['PassAttempts']
    team_offense['EPA / Play'] = (team_offense['RushEPA'] + team_offense['PassEPA']) / team_offense['Snaps']

    return team_offense



def get_team_stats(pbp_data: pd.DataFrame, unit: str):
    ROUND = 3

    gpby_col = 'posteam' if unit == 'offense' else 'defteam'

    ## Standard ##
    team_standard = pbp_data.loc[(~pbp_data['Is Special Teams Play']), :].groupby(gpby_col).aggregate(
        Games=('game_id', 'nunique'),
        Plays=('posteam', lambda x: x[(pbp_data['rush_attempt'] == 1) | (pbp_data['pass_attempt'] == 1)].shape[0]),
        OnSchedulePlays=('posteam', lambda x: x[(pbp_data['On Schedule Play'])].shape[0]),
        TDs=('touchdown', 'sum'),
        FirstDowns=('first_down', 'sum'),
        ThirdDownAtts=(gpby_col, lambda x: x[(pbp_data['third_down_converted'] == 1) | (pbp_data['third_down_failed'] == 1)].shape[0]),
        ThirdDownConvs=('third_down_converted', 'sum'),

        RushAttempts=('rush_attempt', 'sum'),
        RushYards=('rushing_yards', 'sum'),#, lambda x: x[pbp_data['rush'] == 1].sum()),
        RushTDs=('rush_touchdown', 'sum'),
        Rush1Ds=('first_down_rush', 'sum'),
        DesignedRushPlays=('rush', 'sum'),
        DesignedRushAttempts=('rush_attempt', lambda x: x[pbp_data['rush'] == 1].sum()),
        DesignedRushYards=('rushing_yards', lambda x: x[(pbp_data['rush'] == 1)].sum()),
        QBScrambles=('qb_scramble', lambda x: x[pbp_data['rush_attempt'] == 1].sum()),
        ScrambleYards=('rushing_yards', lambda x: x[(pbp_data['qb_scramble'] == 1) & (pbp_data['rush_attempt'] == 1)].sum()),

        DesignedPassPlays=('pass', 'sum'),
        Dropbacks=('qb_dropback', 'sum'),
        PassCompletions=('complete_pass', 'sum'),
        PassAttempts=('pass_attempt', 'sum'),
        PassYards=('passing_yards', 'sum'),# lambda x: x[pbp_data['pass'] == 1].sum()),
        PassTDs=('pass_touchdown', 'sum'),
        Pass1Ds=('first_down_pass', 'sum'),

        Sacks=('sack', 'sum'),
        SackYards=('yards_gained', lambda x: x[pbp_data['sack'] == 1].sum()),
        INTs=('interception', 'sum'),

        TFLs=('tackled_for_loss', 'sum'),
        Fumbles=('fumble_lost', 'sum'),

        Penalties=('penalty', lambda x: x[pbp_data['penalty_team'] == pbp_data[gpby_col]].sum()),
        PenaltyYards=('penalty_yards', lambda x: x[pbp_data['penalty_team'] == pbp_data[gpby_col]].sum()),
        Penalty1Ds=('first_down_penalty', 'sum'),# lambda x: x[pbp_data['penalty_team'] == pbp_data['defteam']].sum()),

        Drives=('drive_play_id_started', 'nunique'),
    )

    # Adjustments
    team_standard['PassYards'] = team_standard['PassYards'] + team_standard['SackYards']
    team_standard['PassAttempts'] = team_standard['PassAttempts'] - team_standard['Sacks']

    # Totals
    team_standard['Total Yards'] = team_standard['RushYards'] + team_standard['PassYards']
    team_standard['Turnovers'] = team_standard['INTs'] + team_standard['Fumbles']


    ## Advanced ##
    team_advanced = pbp_data.loc[(pbp_data['Offensive Snap']) & (~pbp_data['Is Special Teams Play']), :].groupby(gpby_col).aggregate(
    # team_advanced = pbp_data.groupby(gpby_col).aggregate(
        PlaysAdv=('posteam', 'size'),
        PassPlays=('pass', 'sum'),
        RushPlays=('rush', 'sum'),
        EPA=('epa', 'sum'),
        RushEPA=('epa', lambda x: x[pbp_data['rush'] == 1].sum()),
        PassEPA=('epa', lambda x: x[pbp_data['pass'] == 1].sum()),
        Successes=('success', 'sum'),
        RushSuccesses=('success', lambda x: x[pbp_data['rush'] == 1].sum()),
        PassSuccesses=('success', lambda x: x[pbp_data['pass'] == 1].sum()),
        WPA=('wpa', 'sum'),
        RushWPA=('wpa', lambda x: x[pbp_data['rush'] == 1].sum()),
        PassWPA=('wpa', lambda x: x[pbp_data['pass'] == 1].sum()),
    )

    team_advanced['EPA / Play'] = round(team_advanced['EPA'] / team_advanced['PlaysAdv'], ROUND)
    team_advanced['Rush EPA / Play'] = round(team_advanced['RushEPA'] / team_advanced['RushPlays'], ROUND)
    team_advanced['Pass EPA / Play'] = round(team_advanced['PassEPA'] / team_advanced['PassPlays'], ROUND)
    team_advanced['Success Rate'] = round(team_advanced['Successes'] / team_advanced['PlaysAdv'], ROUND)
    team_advanced['Rush Success Rate'] = round(team_advanced['RushSuccesses'] / team_advanced['RushPlays'], ROUND)
    team_advanced['Pass Success Rate'] = round(team_advanced['PassSuccesses'] / team_advanced['PassPlays'], ROUND)

    ## Master ##
    master = team_standard.merge(team_advanced, left_index=True, right_index=True)
    master = master.sort_index()

    return master