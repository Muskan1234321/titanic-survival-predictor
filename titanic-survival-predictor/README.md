# Titanic Survival Predictor

This project predicts whether a passenger would have survived the Titanic disaster using machine learning.  
It’s built with **Python**, **Scikit-learn**, and **Streamlit** for an interactive web interface.

---

## Project Overview
The model is trained using the famous [Titanic dataset](https://www.kaggle.com/c/titanic) containing passenger information such as:
- Passenger Class (Pclass)
- Sex
- Age
- Siblings/Spouses Aboard
- Parents/Children Aboard
- Fare
- Port of Embarkation (C, Q, S)

The trained model (`titanic_model.pkl`) predicts survival chances and provides reasoning for the prediction.

---

## Technologies Used
- Python 
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Streamlit  
- Pickle (for model saving/loading)

---

##  How to Run

1) Clone the Repository
```bash
git clone https://github.com/<Muskan1234321>/titanic_model.git
cd titanic_model
```

2) Install Dependencies

Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```
3) Run the Streamlit App
```bash
streamlit run app.py
```

### Model Training

The model was trained on Decision Tree due to its interpretability and reasonable accuracy.

### Features

Clean, minimal UI built with Streamlit

Explains reasoning behind predictions

Displays feature importance chart

Uses emojis and animations for engagement 🎉

### Author

Muskan Ijaz
AI & Cybersecurity Explorer
