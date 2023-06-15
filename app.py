# Packages
from pybaseball import team_game_logs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import (LogisticRegression, PassiveAggressiveClassifier, Perceptron,RidgeClassifier, SGDClassifier)
from sklearn.naive_bayes import (BernoulliNB, CategoricalNB, ComplementNB, GaussianNB)
from sklearn.neighbors import (KNeighborsClassifier, RadiusNeighborsClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import streamlit as st

def data_collection(team, start_year, end_year):
    data = []
    for i in range(start_year, end_year):
        season = i + 1
        game_logs = team_game_logs(season, team)
        game_logs = pd.DataFrame(game_logs)
        game_logs['year'] = season
        data.append(game_logs)
    df = pd.concat(data)
    df[['Home', 'other_team']] = df['Rslt'].str.split(",", expand=True)[1].str.split('-', expand=True).astype(int)
    df['Rslt'] = df['Rslt'].str.split(",", expand=True)[0]
    df['Date'] = df['Date'].str.split("(", expand=True)[0]
    df['Date'] = df['Date'].str.replace(', ', ' ') + ' ' + df.year.astype(str)
    df['Date'] = df['Date'].str.replace('susp', '')
    df['OppStart'] = df['OppStart'].str.split("(", expand=True)[0]
    df['Date'] = pd.to_datetime(df['Date'], format='%b %d %Y')
    return df

def predict(home, opp, oppstart, xtrain, ytrain):
    features = xtrain.copy()
    # Feature engineering: 5-day averages
    numerical_columns = ['other_team', 'H', '2B', 'HR', 'BA', 'SLG']
    columns = []
    for column in numerical_columns:
        features[column + '_5d_avg'] = features[column].rolling(window=5, min_periods=1).mean()
        columns.append(column + '_5d_avg')
    # Getting label encoder for categorical variables
    label_encoder = LabelEncoder()
    features['Home'] = label_encoder.fit_transform(features['Home'])
    features['Opp'] = label_encoder.fit_transform(features['Opp'])
    features['OppStart'] = label_encoder.fit_transform(features['OppStart'])
    ytrain = label_encoder.fit_transform(ytrain)
    # Getting Proper Model to be Applied
    accuracy_scores = []
    classifiers = all_estimators(type_filter='classifier')
    for name, ClassifierClass in classifiers:
        try:
            classifier = ClassifierClass()
            classifier.fit(features[columns], ytrain)
            y_pred = classifier.predict(features[columns])
            accuracy = accuracy_score(ytrain, y_pred)
            accuracy_scores.append({'Model': name, 'Accuracy': accuracy})
        except Exception as e:
            pass
    accuracy_df = pd.DataFrame(accuracy_scores)
    accuracy_df.sort_values('Accuracy', ascending=False, inplace=True)
    top_models = accuracy_df.head(5)
    new_preds = features.tail(1)
    input_values = new_preds[columns].values[0]
    inputs = {
        **dict(zip(columns, input_values)),
        'Home': home,
        'Opp': opp,
        'OppStart': oppstart
    }
    input_df = pd.DataFrame(inputs, index=[0])
    # Fitting the top model
    model_name = top_models.iloc[0][0]
    model = eval(model_name + '()')
    model.fit(features[columns], ytrain)
    y_pred = model.predict(input_df[columns])
    return y_pred

def main():
    st.title("Baseball Result Prediction")
    team_abbreviations = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CWS', 'CIN', 'CLE', 'COL', 'DET', 'HOU', 'KC', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'OAK', 'PHI', 'PIT', 'SD', 'SEA', 'SF', 'STL', 'TB', 'TEX', 'TOR', 'WSH']
    team = st.selectbox("Home Team", sorted(team_abbreviations))

    start_year = st.number_input("Start Year", min_value=2018, max_value=2050, value=2018, step=1)
    end_year = st.number_input("End Year", min_value=2018, max_value=2050, value=2023, step=1)

    if start_year >= end_year:
        st.error("Start Year should be less than End Year.")
        return

    # Collecting data based on user inputs
    df = data_collection(team, start_year, end_year)

    # Getting opponent options
    opponents = sorted(df['Opp'].unique(), key=lambda x: x.split()[0])
    opp = st.selectbox("Opponent", opponents)

    # Getting opponent start options
    oppstarts = sorted(df['OppStart'].unique(), key=lambda x: x.split()[0])
    oppstart = st.selectbox("Opponent Start", oppstarts)

    # User input for home
    home = st.radio("Home", (0, 1), index=1)

    # Splitting data into train and test sets
    xtrain, _, ytrain, _ = train_test_split(df[['Home', 'Opp', 'OppStart', 'other_team', 'H', '2B', 'HR', 'BA', 'SLG']], df['Rslt'], test_size=0.2)

    if st.button("Predict"):
        prediction = predict(home, opp, oppstart, xtrain, ytrain)
        st.subheader("Prediction Result")
        if prediction == 1:
            st.write("The team is predicted to win.")
        else:
            st.write("The team is predicted to lose.")

if __name__ == "__main__":
    main()
