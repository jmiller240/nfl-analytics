
import os
import src.loaders as loaders


# ----- Constants / Parameters -----

CURRENT_SEASON = 2026
UPDATE_CURRENT_SEASON_DATA = True


# ----- Script -----

# Delete current season files, if necessary
if UPDATE_CURRENT_SEASON_DATA:
    files_to_remove = [
        f'data/schedules_{CURRENT_SEASON}.csv',
        f'data/ftn_charting_{CURRENT_SEASON}.csv',
        f'data/pbp_{CURRENT_SEASON}.csv',
    ]
    for file in files_to_remove:
        if os.path.exists(f'./{file}'):
            print(f'Deleting {file}')
            os.remove(file)

# Players
loaders.get_player_data()

# Schedules
seasons = [i for i in range(1999, CURRENT_SEASON + 1)]
loaders.get_schedules(years=seasons)

# FTN
seasons = [i for i in range(2022, CURRENT_SEASON + 1)]
loaders.get_ftn_data(years=seasons)

# PBP
seasons = [i for i in range(1999, CURRENT_SEASON + 1)]
loaders.get_pbp_data(years=seasons)