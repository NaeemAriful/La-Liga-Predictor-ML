import pandas as pd
import glob
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier

folder_path = 'd:/ML_Model/' 

all_files = glob.glob(os.path.join(folder_path, "*.csv"))

data_list = []
for file in all_files:
    if 'merged' in file:
        continue  # Skip merged file
    if 'home_df' in file:
        continue  # Skip home_df file
    if 'team_stats' in file:
        continue  # Skip team_stats file


    df = pd.read_csv(file, on_bad_lines='skip')
    data_list.append(df)

combined_data = pd.concat(data_list, ignore_index=True)

#Data Cleaning
keep_cols = ['Date','HomeTeam','AwayTeam','FTHG','FTAG','HS','AS','HST','AST','HC','AC','HY','AY','HR','AR']
combined_data = combined_data[keep_cols]

combined_data['Date'] = pd.to_datetime(combined_data['Date'], dayfirst=True, errors='coerce') #DMY format
combined_data = combined_data.dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])
combined_data = combined_data.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam']).copy()
combined_data = combined_data.sort_values('Date') # Crucial for Elo


combined_data['SeasonStart'] = None
combined_data['Season'] = None

for i in range(len(combined_data)):

    current_date = combined_data.loc[i, 'Date']
    year = current_date.year
    month = current_date.month

    if month >= 8:
        season_start = year
    else:
        season_start = year - 1

    combined_data.loc[i, 'SeasonStart'] = season_start

    season_end = season_start + 1
    season_end_last_two = season_end % 100

    if season_end_last_two < 10:
        season_end_last_two = '0' + str(season_end_last_two)
    else:
        season_end_last_two = str(season_end_last_two)

    season_string = str(season_start) + '-' + season_end_last_two
    combined_data.loc[i, 'Season'] = season_string


# 3. CALCULATE ELO RATINGS 

elo_ratings = {team: 1500 for team in pd.concat([combined_data['HomeTeam'], combined_data['AwayTeam']]).unique()}
k_factor = 30 

def get_elo_update(w_elo, l_elo, result):
    expected_win = 1 / (10 ** ((l_elo - w_elo) / 400) + 1)
    return k_factor * (result - expected_win)

home_elos, away_elos = [], []

for _, row in combined_data.iterrows():
    h_team, a_team = row['HomeTeam'], row['AwayTeam']
    h_elo, a_elo = elo_ratings[h_team], elo_ratings[a_team]
    
    home_elos.append(h_elo)
    away_elos.append(a_elo)
    
    # Result from Home perspective: 1=Win, 0.5=Draw, 0=Loss
    if row['FTHG'] > row['FTAG']: h_res, a_res = 1, 0
    elif row['FTHG'] < row['FTAG']: h_res, a_res = 0, 1
    else: h_res, a_res = 0.5, 0.5
    
    shift = get_elo_update(h_elo, a_elo, h_res)
    elo_ratings[h_team] += shift
    elo_ratings[a_team] -= shift

combined_data['Home_Elo'] = home_elos
combined_data['Away_Elo'] = away_elos


home_view = pd.DataFrame({
    'Season': combined_data['Season'], 'Date': combined_data['Date'],
    'Team': combined_data['HomeTeam'], 'Opponent': combined_data['AwayTeam'],
    'Home': 1, 'GoalsFor': combined_data['FTHG'], 'GoalsAgainst': combined_data['FTAG'],
    'Shots': combined_data['HS'], 'ShotsOnTarget': combined_data['HST'],
    'Corners': combined_data['HC'], 'Yellow': combined_data['HY'], 'Red': combined_data['HR'],
    'Team_Elo': combined_data['Home_Elo'], 'Opp_Elo': combined_data['Away_Elo']
})

away_view = pd.DataFrame({
    'Season': combined_data['Season'], 'Date': combined_data['Date'],
    'Team': combined_data['AwayTeam'], 'Opponent': combined_data['HomeTeam'],
    'Home': 0, 'GoalsFor': combined_data['FTAG'], 'GoalsAgainst': combined_data['FTHG'],
    'Shots': combined_data['AS'], 'ShotsOnTarget': combined_data['AST'],
    'Corners': combined_data['AC'], 'Yellow': combined_data['AY'], 'Red': combined_data['AR'],
    'Team_Elo': combined_data['Away_Elo'], 'Opp_Elo': combined_data['Home_Elo']
})

team_df = pd.concat([home_view, away_view], ignore_index=True).sort_values(['Team', 'Date']).reset_index(drop=True)
team_df['Result'] = np.where(team_df['GoalsFor'] > team_df['GoalsAgainst'], 'Win', np.where(team_df['GoalsFor'] == team_df['GoalsAgainst'], 'Draw', 'Loss'))
team_df['Elo_Gap'] = team_df['Team_Elo'] - team_df['Opp_Elo']

# -----------------------------
# 5. Impute & EWMA
# -----------------------------
num_cols = ['GoalsFor','GoalsAgainst','Shots','ShotsOnTarget','Corners','Yellow','Red']
team_df[num_cols] = team_df[num_cols].fillna(0)

for col in num_cols:
    team_df[f'EWMA_{col}_5'] = team_df.groupby('Team')[col].transform(lambda x: x.shift(1).ewm(span=5, adjust=False).mean())

ewma_cols = [f'EWMA_{c}_5' for c in num_cols]
team_df[ewma_cols] = team_df[ewma_cols].fillna(0)

# -----------------------------
# 6. Encode Teams
# -----------------------------
le = LabelEncoder()
all_teams = pd.concat([combined_data['HomeTeam'], combined_data['AwayTeam']]).unique()
le.fit(all_teams)
team_df['Team_enc'] = le.transform(team_df['Team'])
team_df['Opp_enc'] = le.transform(team_df['Opponent'])

# -----------------------------
# 7. Split & Model
# -----------------------------
train = team_df[team_df['Date'] < '2024-08-01']
test = team_df[team_df['Date'] >= '2024-08-01']

features = ['Home', 'Team_enc', 'Opp_enc', 'Elo_Gap'] + ewma_cols
X_train, y_train = train[features], train['Result']
X_test, y_test = test[features], test['Result']

model = RandomForestClassifier(n_estimators=200, min_samples_split=10, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 8. Evaluation
# -----------------------------
preds = model.predict(X_test)
print(f"Realistic Accuracy (Elo + EWMA): {accuracy_score(y_test, preds):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, preds))

# -----------------------------
# 9. Prediction Function
# -----------------------------
def predict_upcoming_match(team_a, team_b, home_adv):
    # Get latest data
    team_a_data = team_df[team_df['Team'] == team_a].iloc[-1]
    team_b_data = team_df[team_df['Team'] == team_b].iloc[-1]
    
    # Use latest Elo calculated from the last game
    # We must calculate the gap for the NEW match
    match_row = pd.DataFrame([{
        'Home': home_adv,
        'Team_enc': le.transform([team_a])[0],
        'Opp_enc': le.transform([team_b])[0],
        'Elo_Gap': team_a_data['Team_Elo'] - team_b_data['Team_Elo'],
        **team_a_data[ewma_cols]
    }])
    
    probs = model.predict_proba(match_row[features])[0]
    print(f"\nPrediction for {team_a} vs {team_b}:")
    for outcome, prob in zip(model.classes_, probs):
        print(f"{outcome}: {prob*100:.2f}%")

predict_upcoming_match("Girona", "Getafe", 1)



