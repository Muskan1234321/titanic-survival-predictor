import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time

# ----------------------------
# 🎨 Page Configuration
# ----------------------------
st.set_page_config(page_title="🚢 Titanic Survival Predictor", page_icon="🛳️", layout="wide")

st.markdown("""
    <style>
    .main {background-color: #f0f8ff;}
    .title {text-align: center; color: #003366;}
    .pred-box {
        background-color: #e8f5e9; padding: 15px; border-radius: 12px;
        text-align: center; border: 2px solid #4caf50;
    }
    .warn-box {
        background-color: #ffebee; padding: 15px; border-radius: 12px;
        text-align: center; border: 2px solid #f44336;
    }
    .reason-box {
        background-color: #f9f9f9; padding: 10px; border-radius: 10px;
        border: 1px solid #ccc; margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ----------------------------
# 📊 Load Dataset
# ----------------------------
data = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
data = data[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Survived"]]
data.dropna(inplace=True)
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
data["Embarked"] = data["Embarked"].map({"C": 0, "Q": 1, "S": 2})

X = data.drop("Survived", axis=1)
y = data["Survived"]

# ----------------------------
# 🧠 Train Model
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=89)
model = DecisionTreeClassifier(max_depth=6, random_state=89)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

# ----------------------------
# 🚀 UI Layout
# ----------------------------
st.markdown("<h1 class='title'>🛳️ Titanic Survival Predictor</h1>", unsafe_allow_html=True)
st.markdown(f"<h4 style='text-align:center;'>Model Accuracy: {accuracy*100:.2f}%</h4>", unsafe_allow_html=True)

st.write("Fill in your details below to find out whether you would have lived to tell the tales 🎉 or perished in the icy waters 💀")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox("🎟️ Ticket Class", [1, 2, 3], help="1st = Luxury, 3rd = Economy")
    sex = st.selectbox("🧍 Gender", ["male", "female"])
    age = st.slider("🎂 Age", 1, 80, 25)

with col2:
    sibsp = st.number_input("👨‍👩‍👧‍👦 Siblings/Spouses Aboard", 0, 8, 0)
    parch = st.number_input("👶 Parents/Children Aboard", 0, 6, 0)
    fare = st.number_input("💰 Ticket Fare ($)", 0.0, 600.0, 32.2)

with col3:
    embarked = st.selectbox("⚓ Port of Embarkation", ["C", "Q", "S"],
                            help="C = Cherbourg, Q = Queenstown, S = Southampton")

# ----------------------------
# 🔮 Prediction Logic
# ----------------------------
if st.button("🚀 Predict My Fate"):
    with st.spinner("Analyzing your survival chances... ⏳"):
        sex_val = 1 if sex == "female" else 0
        embarked_val = {"C": 0, "Q": 1, "S": 2}[embarked]
        input_data = np.array([[pclass, sex_val, age, sibsp, parch, fare, embarked_val]])
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

    # --- Reasons for prediction ---
    reasons = []
    if sex_val == 1:
        reasons.append("Being female historically gave a strong survival advantage.")
    else:
        reasons.append("Being male historically had lower survival chances.")
    if pclass == 1:
        reasons.append("First class passengers were closest to lifeboats.")
    elif pclass == 2:
        reasons.append("Second class had moderate survival chances.")
    else:
        reasons.append("Third class was farthest from lifeboats — harder to escape.")
    if age < 15:
        reasons.append("Children often received priority boarding lifeboats.")
    elif age > 50:
        reasons.append("Older passengers had reduced mobility in panic situations.")
    if fare > 100:
        reasons.append("Higher fare often correlated with wealth & quicker rescue.")
    if embarked == "C":
        reasons.append("Passengers from Cherbourg had slightly higher survival rates.")

    # --- Output animations ---
    if pred == 1:
        st.balloons()
        st.markdown(
            f'<div class="pred-box"><h2>🟢 You Survived!</h2>'
            f'<p>Congratulations! You had a <b>{prob*100:.1f}%</b> chance of survival.</p></div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="warn-box"><h2>🔴 You Did Not Survive</h2>'
            f'<p>Your chance of survival was only <b>{prob*100:.1f}%</b>.</p></div>',
            unsafe_allow_html=True
        )

        with st.spinner("💀 The ship is sinking... hold on tight!"):
            stages = [
                "🚢 The Titanic hits the iceberg... ❄️",
                "💨 Chaos spreads across the deck...",
                "🌊 Water rushes into the lower compartments...",
                "🧊 You cling to a piece of debris...",
                "😔 You slowly sink beneath the icy water...",
                "⚰️ Game Over — You did not survive."
            ]
            for s in stages:
                time.sleep(1.3)
                st.write(s)

        st.markdown(
            """
            <div style='text-align:center'>
                <img src='https://media.tenor.com/nmC9o2X81yQAAAAC/titanic-sinking.gif' width='320'>
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- Progress bar + reasoning ---
    st.progress(int(prob * 100))
    st.markdown(
        '<div class="reason-box"><h4>💡 Why this prediction?</h4><ul>' +
        ''.join([f"<li>{r}</li>" for r in reasons]) +
        '</ul></div>',
        unsafe_allow_html=True
    )

    # --- Feature Importance Chart ---
    st.markdown("### 🔍 What mattered most to the model:")
    importances = model.feature_importances_
    features = X.columns
    fig, ax = plt.subplots()
    ax.barh(features, importances, color="#1e88e5")
    ax.set_xlabel("Importance Score")
    ax.set_title("Decision Tree Feature Importance")
    st.pyplot(fig)
("---")
st.markdown(" *Created by Muskan Ijaz* 💻")
