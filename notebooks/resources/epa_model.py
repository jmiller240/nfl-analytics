

from pprint import pprint
import math
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, f1_score, classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler

import nfl_data_py as nfl

from resources.get_nfl_data import get_team_info, get_matchups, get_weeks, get_pbp_data



''' Constants / Parameters  '''

## Parameters
INPUT_YEARS = [i for i in range(2018, 2026)]

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



class EPAModel:

    def __init__(self):
        pass

    
    def predict_week(self, prediction_season: int, prediction_week: int):

        # Get team info
        team_data = get_team_info().reset_index()

        # Get inputs
        input_matchups, pred_matchups = self.get_epa_inputs(prediction_season=prediction_season, prediction_week=prediction_week)

        X = input_matchups[FEATURES].to_numpy()

        ''' Picks Model '''
        print(f'Picks Model')

        # Get X and y
        y = input_matchups['winner'].to_numpy()
        print(f'X shape:', X.shape)
        print(f'Y shape:', y.shape)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Create a Logistic Regression model
        picks_model = LogisticRegression(max_iter=100, solver='liblinear') # Increased max_iter for convergence

        # Train the model
        picks_model.fit(X_train, y_train)

        # Make predictions
        y_pred = picks_model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy*100:,.2f}%")

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel().tolist()
        print(cm)
        print(f"Pred Home, Home Win: {tp}")
        print(f"Pred Home, Away Win: {fp}")
        print(f"Pred Away, Home Win: {fn}")
        print(f"Pred Away, Away Win: {tn}")
        pred_home = tp+fp
        pred_away = fn+tn
        home_winners = tp+fn
        away_winners = fp+tn

        print(f'Pred Home: {pred_home} ({pred_home / len(y_test):.2%})')
        print(f'Pred Away: {pred_away}')
        print(f'Home Winners: {home_winners} ({home_winners / len(y_test):,.2%})')
        print(f'Away Winners: {away_winners}')

        f1 = f1_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)        # recall: % of values picked out (i.e., % of winners picked); precision: % correct (i.e., accuracy of picks)
        print(f"f1: {f1:,.5f}")
        print(class_report)


        ''' Scores Model '''
        print(f'Scores Model')

        ## Home Score ##
        print(f'Home scores')

        y = input_matchups['home_score'].to_numpy()
        print(f'X shape:', X.shape)
        print(f'Y shape:', y.shape)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Create a Logistic Regression model
        home_score_model = LinearRegression() # Increased max_iter for convergence

        # Train the model
        home_score_model.fit(X_train, y_train)

        # Make predictions
        y_pred = home_score_model.predict(X_test)

        # Evaluate the model
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        # Print evaluation metrics
        print(f"R-squared: {r2:.4f}")
        print(f"Mean squared error: {mse:.4f}")
        print(f"Root mean squared error: {rmse:.4f}")


        ## Away Score ##
        print(f'Away scores')

        y = input_matchups['away_score'].to_numpy()
        print(f'X shape:', X.shape)
        print(f'Y shape:', y.shape)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Create a Logistic Regression model
        away_score_model = LinearRegression() # Increased max_iter for convergence

        # Train the model
        away_score_model.fit(X_train, y_train)

        # Make predictions
        y_pred = away_score_model.predict(X_test)

        # Evaluate the model
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        # Print evaluation metrics
        print(f"R-squared: {r2:.4f}")
        print(f"Mean squared error: {mse:.4f}")
        print(f"Root mean squared error: {rmse:.4f}")


        ''' Predict week '''
        print(f'Predicting week')

        # Get inputs
        X = pred_matchups[FEATURES].to_numpy()

        ## PICKS
        y_pred = picks_model.predict(X)
        probs = picks_model.predict_proba(X)

        ## SCORES
        home_scores = home_score_model.predict(X)
        away_scores = away_score_model.predict(X)

        ## Collect results
        predictions_df = pred_matchups[['game_id', 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score', 'home_moneyline', 'away_moneyline', 'spread_line', 'away_spread_odds', 'home_spread_odds', 'total_line', 'under_odds', 'over_odds']].copy().reset_index(drop=True)

        # Win Probs - Moneyline
        predictions_df['prob_home'] = [probs[i][1] for i in range(len(probs))]
        predictions_df['prob_away'] = [probs[i][0] for i in range(len(probs))]
        predictions_df['pred'] = np.where(y_pred == 1, predictions_df['home_team'], predictions_df['away_team'])
        predictions_df['pred_prob'] = predictions_df[['prob_home','prob_away']].max(axis=1)

        predictions_df['pred_home_ml'] = np.where(predictions_df['prob_home'] > predictions_df['prob_away'],
                                        (-100*predictions_df['prob_home'])/(1 - predictions_df['prob_home']),
                                        ((1 - predictions_df['prob_home'])/predictions_df['prob_home'])*100).astype(int)
        predictions_df['pred_away_ml'] = np.where(predictions_df['prob_away'] > predictions_df['prob_home'],
                                        (-100*predictions_df['prob_away'])/(1 - predictions_df['prob_away']),
                                        ((1 - predictions_df['prob_away'])/predictions_df['prob_away'])*100).astype(int)
                                    
        predictions_df['home_ml_value'] = predictions_df['home_moneyline'] - predictions_df['pred_home_ml']
        predictions_df['away_ml_value'] = predictions_df['away_moneyline'] - predictions_df['pred_away_ml']
        predictions_df['moneyline_value'] = np.where(predictions_df['home_ml_value'] >= predictions_df['away_ml_value'], 'home', 'away')

        # Scores / Spread / Total
        predictions_df['pred_home_score'] = home_scores
        predictions_df['pred_away_score'] = away_scores
        predictions_df['pred_winner_score'] = np.where(predictions_df['pred_home_score'] >= predictions_df['pred_away_score'], 'home', 'away')
        predictions_df['pred_spread'] = predictions_df['pred_home_score'] - predictions_df['pred_away_score']
        predictions_df['spread_difference'] = predictions_df['pred_spread'] - predictions_df['spread_line']
        predictions_df['spread_pick'] = np.where(predictions_df['spread_difference'] >= 0, 'home', 'away')
        predictions_df['pred_total'] = home_scores + away_scores
        predictions_df['total_pick'] = np.where(predictions_df['pred_total'] >= predictions_df['total_line'], 'over', 'under')

        # Format moneylines
        predictions_df['pred_home_ml_viz'] = np.where(predictions_df['pred_home_ml'] > 0, '+' + predictions_df['pred_home_ml'].astype(str), predictions_df['pred_home_ml'].astype(str))
        predictions_df['pred_away_ml_viz'] = np.where(predictions_df['pred_away_ml'] > 0, '+' + predictions_df['pred_away_ml'].astype(str), predictions_df['pred_away_ml'].astype(str))

        # Colors and logos
        predictions_df = predictions_df.merge(team_data[['team', 'team_color', 'team_logo_espn']], left_on='home_team', right_on='team', how='left').drop(columns=['team']).rename(columns={'team_color': 'home_color', 'team_logo_espn': 'home_logo'})
        predictions_df = predictions_df.merge(team_data[['team', 'team_color', 'team_logo_espn']], left_on='away_team', right_on='team', how='left').drop(columns=['team']).rename(columns={'team_color': 'away_color', 'team_logo_espn': 'away_logo'})

        return predictions_df



    def get_epa_inputs(self, prediction_season: int, prediction_week: int):
        '''
        Get inputs for epa predictions models

        Returns
        -------
        '''

        ''' Import / Process Data '''

        ## Download ##

        # Team info
        # team_data = get_team_info()

        # Matchups
        master_matchups_df = get_matchups(years=INPUT_YEARS)

        # Weeks
        master_weeks = get_weeks(years=INPUT_YEARS)

        # Add week back to matchup
        master_matchups_df = master_matchups_df.merge(master_weeks, left_on=['season', 'week'], right_on=['season', 'week'])

        # PBP
        pbp_data = get_pbp_data(years=INPUT_YEARS)


        ''' Establish some variables '''

        C_MASTER_WEEK = master_weeks.loc[(master_weeks['season'] == prediction_season) & (master_weeks['week'] == prediction_week), 'master_week'].values[0]
        INPUT_WEEKS = master_weeks.loc[(master_weeks['season'] >= 2019) & (master_weeks['master_week'] < C_MASTER_WEEK), 'master_week'].unique().tolist()
        ALL_MATCHUP_WEEKS = INPUT_WEEKS + [C_MASTER_WEEK]

        matchups = master_matchups_df.loc[master_matchups_df['master_week'].isin(ALL_MATCHUP_WEEKS),:]

        print(f'Current master week:', C_MASTER_WEEK)
        print(f'Input weeks:', INPUT_WEEKS)
        print(matchups.head(2).to_string())
        print(matchups.tail(2).to_string())


        ''' Calculate EPA for every historical team / game '''

        pbp_adv_slice_nonst = pbp_data.loc[(pbp_data['Offensive Snap']) & (~pbp_data['Is Special Teams Play']), :]
        pbp_adv_slice_st = pbp_data.loc[pbp_data['Is Special Teams Play'], :]

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

        special_teams_epa.index = special_teams_epa.index.set_names('team', level='posteam')

        # Combine
        master_epa_df = offense_epa.merge(defense_epa, left_index=True, right_index=True)
        master_epa_df = master_epa_df.merge(special_teams_epa, left_index=True, right_index=True)

        master_epa_df = master_epa_df.reset_index().merge(master_weeks, left_on=['season', 'week'], right_on=['season', 'week'], how='left')


        ''' Reshape dfs '''

        master_weeks = master_weeks.set_index(['season', 'week'])
        master_epa_df = master_epa_df.set_index(['master_week', 'team'])


        ''' Forge Home / Away EPA Inputs for every matchup '''

        ## EPA Inputs
        epa_inputs_df = pd.DataFrame(columns=['master_week', 'team'] + EPA_COLS + EPA_PLAY_COLS)

        for week in ALL_MATCHUP_WEEKS:

            home_teams = matchups.loc[matchups['master_week'] == week, 'home_team'].unique().tolist()
            away_teams = matchups.loc[matchups['master_week'] == week, 'away_team'].unique().tolist()
            
            df = self.get_week_epa_inputs(master_epa_df=master_epa_df, teams=home_teams+away_teams, master_week=week)
            df['master_week'] = week

            epa_inputs_df = pd.concat([epa_inputs_df, df])

        epa_inputs_df = epa_inputs_df.reset_index(drop=True)

        ## Add back to input matchups df

        # Home team EPA
        rename_dict = {col: f'Home_Team_{col}' for col in EPA_COLS + EPA_PLAY_COLS}
        matchups = matchups.merge(epa_inputs_df, left_on=['master_week', 'home_team'], right_on=['master_week', 'team'], how='left').rename(columns=rename_dict).drop(columns='team')

        # Away team EPA
        rename_dict = {col: f'Away_Team_{col}' for col in EPA_COLS + EPA_PLAY_COLS}
        matchups = matchups.merge(epa_inputs_df, left_on=['master_week', 'away_team'], right_on=['master_week', 'team'], how='left').rename(columns=rename_dict).drop(columns='team')

        # print(matchups.loc[(matchups['master_week'] == 137) & (matchups['away_team'] == 'IND'),:].to_string())

        ''' Return '''

        input_matchups = matchups.loc[matchups['master_week'].isin(INPUT_WEEKS), :].copy()
        pred_matchups = matchups.loc[matchups['master_week'] == C_MASTER_WEEK, :].copy()

        return input_matchups, pred_matchups


    ''' Helpers '''

    def get_week_epa_inputs(self, master_epa_df: pd.DataFrame, teams: list, master_week: int):
        
        # Start return df
        teams_df = pd.DataFrame(data={'team': teams}).set_index('team')

        # Sum up EPA and Plays for each team and last n games
        for team in teams:
            team_sl = master_epa_df.loc[master_epa_df.index.get_level_values(1) == team, :]

            for n in [4,8,12,16]:
                sl = team_sl.loc[(team_sl.index.get_level_values(0) < master_week),:].tail(n)
                # if team == 'IND' and n == 4 and master_week == 137:
                    # print(sl.head().to_string())
                    
                for unit in ['O', 'D', 'ST']:
                    epa = sl[f'EPA_{unit}'].sum()
                    plays = sl[f'Plays_{unit}'].sum()

                    teams_df.loc[team, f'Last_{n}_EPA_{unit}'] = epa
                    teams_df.loc[team, f'Last_{n}_EPA_{unit}_Play'] = epa / plays

        teams_df = teams_df.reset_index()

        return teams_df

