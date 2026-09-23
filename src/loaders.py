"""
Data loading and caching layer.

Keeping this separate from app.py matters: as the project grows (new
sports, new sources, a future FastAPI backend), the Dash UI code can stay
thin and just call these functions. If you swap Dash for React someday,
this whole file is reusable as-is behind a FastAPI endpoint.
"""

from pathlib import Path

import nflreadpy as nfl
import pandas as pd
import polars as pl
import numpy as np

from src.constants import PLAY_TYPES_SPECIAL

CACHE_DIR = Path('./data')


# -------- nflreadpy downloaders / cachers ---------

def get_team_data():
    """
    nfl.load_teams()
    """

    # Cache file
    cache_file = CACHE_DIR / f"teams.csv"

    # Read cached if exists
    if cache_file.exists():
        print(f'Reading local file')
        return pl.read_csv(cache_file)

    # Otherwise download + cache
    print(f'nfl.load_teams()')
    df = nfl.load_teams()
    df.write_csv(cache_file)

    return df

def get_player_data():
    """
    nfl.load_players()
    """

    # Cache file
    cache_file = CACHE_DIR / f"players.csv"

    # Read cached if exists
    if cache_file.exists():
        print(f'Reading local file')
        return pl.read_csv(cache_file)

    # Otherwise download + cache
    print(f'nfl.load_players()')
    df = nfl.load_players()
    df.write_csv(cache_file)

    return df

def get_schedules(years: list[int]) -> pl.DataFrame:
    """
    nfl.load_schedules()
    """

    # Load each year
    dfs_list: list[pl.DataFrame] = []
    for year in years:
        # Cache file
        cache_file = CACHE_DIR / f"schedules_{year}.csv"
    
        # Read cached if exists
        if cache_file.exists():
            print(f'Reading {cache_file.name}')
            df = pl.read_csv(cache_file)
        else:
            # Otherwise download + cache
            print(f'nfl.load_schedules({year})')
            df = nfl.load_schedules([year])
            df.write_csv(cache_file)

        dfs_list.append(df)

    comb_df = pl.concat(dfs_list, how='vertical_relaxed')

    return comb_df


def get_ftn_data(years: list[int]) -> pl.DataFrame:
    """
    nfl.load_ftn_charting()
    """

    if min(years) < 2022:
        years = list([i for i in range(2022, max(years) + 1)])

    # Load each year
    dfs_list: list[pl.DataFrame] = []
    for year in years:
        # Cache file
        cache_file = CACHE_DIR / f"ftn_charting_{year}.csv"

        # Read cached if exists
        if cache_file.exists():
            print(f'Reading {cache_file.name}')
            df = pl.read_csv(cache_file)
        else:
            # Otherwise download + cache
            print(f'nfl.load_ftn_charting({year})')
            df = nfl.load_ftn_charting([year])
            df.write_csv(cache_file)

        dfs_list.append(df)

    comb_df = pl.concat(dfs_list, how='vertical_relaxed')

    return comb_df

def get_pbp_data(years: list[int]) -> pl.DataFrame:
    """
    nfl.load_pbp()
    """

    # Load each year
    dfs_list: list[pl.DataFrame] = []
    for year in years:
        # Cache file
        cache_file = CACHE_DIR / f"pbp_{year}.csv"

        # Read cached if exists
        if cache_file.exists():
            print(f'Reading {cache_file.name}')
            df = pl.read_csv(cache_file)
        else:
            # Otherwise download + cache
            print(f'nfl.load_pbp({year})')
            df = nfl.load_pbp([year])
            df.write_csv(cache_file)

        dfs_list.append(df)

    comb_df = pl.concat(dfs_list, how='vertical_relaxed')

    return comb_df


# -------- My Utilities --------

def get_full_event_data(years: list[int]) -> pl.DataFrame:
    # Cache file
    cache_file = CACHE_DIR / f"full_event_data_{min(years)}_{max(years)}.csv"

    # ---- Load ----

    # Read cached if exists
    if cache_file.exists():
        print(f'get_pbp_data Reading local file')
        return pl.read_csv(cache_file)

    # Otherwise get
    pbp_data = get_pbp_data(years=years).to_pandas()
    ftn_data = get_ftn_data(years=years)

    # ---- Add cols ----
    # Drive
    pbp_data['Master Drive ID'] = pbp_data['game_id'] + pbp_data['drive'].astype(str)
    # pbp_data.with_columns(
    #     pl.concat_str([pl.col('game_id'), pl.col('drive').cast(pl.Int8).cast(pl.String)], separator='_').alias('Master Drive ID'),
    # )

    # Snaps
    condt = (
        ((pbp_data['pass'] == 1) | (pbp_data['rush'] == 1)) & 
        (pbp_data['epa'].notna()) & 
        (pbp_data['posteam'].notna())
    )
    pbp_data['Offensive Snap'] = np.where(condt, 1, 0)

    # Flag for special teams
    special_conditions = ((pbp_data['play_type_nfl'].isin(PLAY_TYPES_SPECIAL)) | (pbp_data['special_teams_play'] == 1))
    pbp_data['Is Special Teams Play'] = np.where(special_conditions, 1, 0)
    
    # Explosives
    pbp_data['Explosive Play'] = np.where(pbp_data['yards_gained'] >= 15, 1, 0)

    # On schedule play
    on_schedule_conditions = (
        ((pbp_data['down'] == 1) & (pbp_data['ydstogo'] <= 10)) |
        ((pbp_data['down'] == 2) & (pbp_data['ydstogo'] <= 6)) | 
        ((pbp_data['down'] == 3) & (pbp_data['ydstogo'] <= 4)) | 
        ((pbp_data['down'] == 4) & (pbp_data['ydstogo'] <= 2))
    )
    pbp_data['On Schedule Play'] = np.where(on_schedule_conditions, 1, 0)

    # Play locations
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
        
        # if air_yards <= 0:
        #     return 'Behind LOS'
        if air_yards <= 10:
            return 'Short'
        elif air_yards <= 20:
            return 'Medium'
        else:
            return 'Long'

    pbp_data['Run Location'] = pbp_data.apply(lambda x: run_location(x['run_location'], x['run_gap']), axis=1)

    pbp_data['Pass Length'] = pbp_data['air_yards'].apply(lambda x: pass_length(x))
    pbp_data['Pass Location'] = pbp_data['Pass Length'] + ' ' + pbp_data['pass_location'].str.capitalize()

    # ---- Cache ----
    pbp_data = pl.DataFrame(pbp_data)
    pbp_data.write_csv(cache_file)

    return pbp_data

