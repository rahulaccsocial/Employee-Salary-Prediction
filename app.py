import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load('salary_model.pkl')
encoders = joblib.load('encoders.pkl')
features = joblib.load('features.pkl')

st.title("🧑‍💼 Employee Salary Prediction")

st.markdown("---")
st.subheader("📋 Predict for a Single Employee")

user_input = {}
for feature in features:
    if feature in encoders:
        options = encoders[feature].classes_.tolist()
        user_input[feature] = st.selectbox(f"Select {feature}", options)
    else:
        user_input[feature] = st.number_input(f"Enter {feature}", step=1.0)

if st.button("Predict Salary"):
    input_df = pd.DataFrame([user_input])

    # Apply encoders
    for col in input_df.columns:
        if col in encoders:
            le = encoders[col]
            input_df[col] = le.transform(input_df[col])

    input_df = input_df[features]
    prediction = model.predict(input_df)[0]
    result = "Salary > 50K" if prediction == 1 else "Salary <= 50K"
    st.success(f"🧾 Prediction: **{result}**")

st.markdown("---")
st.subheader("📂 Upload a CSV file for batch prediction")

uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type=['csv'])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Encode categorical columns
        for col in df.columns:
            if col in encoders:
                le = encoders[col]
                df[col] = le.transform(df[col])

        df = df[features]
        predictions = model.predict(df)

        df['Prediction'] = ['>50K' if pred == 1 else '<=50K' for pred in predictions]
        st.success("✅ Prediction completed!")
        st.dataframe(df)

        # Download CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions", data=csv, file_name="salary_predictions.csv", mime='text/csv')

    except Exception as e:
        st.error(f"Error processing file: {e}")
