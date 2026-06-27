import pickle
import pandas as pd
import streamlit as st

# trained in ../notebooks/03_ipl_win_prediction.ipynb (logistic regression,
# split by match so the score is not inflated by leakage)
with open("pipe.pkl", "rb") as f:
    pipe = pickle.load(f)

teams = sorted([
    "Sunrisers Hyderabad", "Mumbai Indians", "Royal Challengers Bangalore",
    "Kolkata Knight Riders", "Kings XI Punjab", "Chennai Super Kings",
    "Rajasthan Royals", "Delhi Capitals",
])

cities = sorted([
    "Hyderabad", "Bangalore", "Mumbai", "Indore", "Kolkata", "Delhi",
    "Chandigarh", "Jaipur", "Chennai", "Pune", "Ahmedabad", "Visakhapatnam",
    "Mohali", "Bengaluru", "Nagpur", "Dharamsala", "Cuttack", "Ranchi",
])

st.title("IPL Win Predictor")
st.caption("Win probability for the chasing side, from a logistic regression "
           "trained on ball-by-ball data (2008-2019).")

col1, col2 = st.columns(2)
batting_team = col1.selectbox("Batting team", teams)
bowling_team = col2.selectbox("Bowling team", teams, index=1)

city = st.selectbox("City", cities)
target = st.number_input("Target", min_value=1, max_value=300, value=180, step=1)

col3, col4, col5 = st.columns(3)
score = col3.number_input("Current score", min_value=0, max_value=300, value=90, step=1)
overs = col4.number_input("Overs completed", min_value=0.0, max_value=20.0,
                          value=10.0, step=0.1)
wickets_lost = col5.number_input("Wickets lost", min_value=0, max_value=10, value=2, step=1)

if st.button("Predict probability"):
    if batting_team == bowling_team:
        st.error("Pick two different teams.")
    elif score >= target:
        st.success(f"{batting_team} have already reached the target.")
    elif overs == 0:
        st.warning("Enter at least some overs so a run rate can be computed.")
    else:
        runs_left = target - score
        balls_left = 120 - int(overs * 6)
        if balls_left <= 0:
            st.warning("No balls left in the innings.")
        else:
            state = pd.DataFrame([{
                "batting_team": batting_team,
                "bowling_team": bowling_team,
                "city": city,
                "runs_left": runs_left,
                "balls_left": balls_left,
                "wickets": 10 - wickets_lost,
                "total_runs_x": target,
                "cur_run_rate": (score * 6) / (overs * 6),
                "req_run_rate": (runs_left * 6) / balls_left,
            }])
            win = pipe.predict_proba(state)[0, 1]
            st.header(f"{batting_team}: {round(win * 100)}%")
            st.header(f"{bowling_team}: {round((1 - win) * 100)}%")
            st.caption(f"Need {runs_left} off {balls_left} balls "
                       f"with {10 - wickets_lost} wickets in hand.")
