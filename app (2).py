import os
import pickle
import pandas as pd
import streamlit as st

# Set the base directory to where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "titanic.pkl")

# Load the trained model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

st.title("Titanic Survival Predictor")
st.write("Enter passenger details to predict survival.")

# Input widgets for features
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.radio("Sex", ["male", "female"])
age = st.slider("Age", 0, 100, 30)
sibspouse = st.number_input("Number of Siblings/Spouses Aboard (sibspouse)", min_value=0, max_value=10, value=0)
parentchild = st.number_input("Number of Parents/Children Aboard (parentchild)", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare", min_value=0.0, value=30.0)

if st.button("Predict Survival"):
    # Create a DataFrame from user inputs
    input_df = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": float(age),
        "sibspouse": int(sibspouse),
        "parentchild": int(parentchild),
        "Fare": float(fare)
    }])

    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1] # Probability of surviving (class 1)

    # Display result
    if prediction == 1:
        st.success(f"Prediction: Survived! 🎉 (Probability: {prediction_proba:.2f})")
    else:
        st.error(f"Prediction: Did not survive. 😔 (Probability: {prediction_proba:.2f})")

st.write("---")
st.write("To run this app, execute the following commands in Colab terminal:")
st.code("!streamlit run /content/app.py & npx localtunnel --port 8501")
