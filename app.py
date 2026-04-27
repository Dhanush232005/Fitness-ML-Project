import streamlit as st
import joblib
import pandas as pd

knn = joblib.load("knn.pkl")
log_model = joblib.load("log_model.pkl")

data = pd.read_csv("users.csv")

data['goal'] = data['goal'].map({
    'weight_loss': 0,
    'muscle_gain': 1,
    'fitness': 2
})

st.title("🏋️ AI Fitness Recommendation System")

age = st.number_input("Age", 10, 60)
weight = st.number_input("Weight", 30, 120)
days = st.number_input("Active Days", 0, 30)

goal_option = st.selectbox("Goal", ["weight_loss", "muscle_gain", "fitness"])

goal_map = {
    "weight_loss": 0,
    "muscle_gain": 1,
    "fitness": 2
}

goal = goal_map[goal_option]

if st.button("Get Recommendation"):

    user = pd.DataFrame([[age, weight, days, goal]],
                        columns=['age', 'weight', 'days_active', 'goal'])

    dist, ind = knn.kneighbors(user)
    workout = data.iloc[ind[0]]['workout'].values[0]

    user_pred = pd.DataFrame([[age, weight]],
                             columns=['age', 'weight'])

    result = 1 if days >= 10 else 0

    st.subheader("Results")
    st.write("Workout:", workout)
    st.write("Engagement:", "Active" if result == 1 else "Inactive")
