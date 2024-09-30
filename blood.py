import joblib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from tabulate import tabulate

# Display local image
st.image("https://github.com/nsubagya/Blood_transfusion_predict/blob/main/2uypz7k4.png", caption="Blood Donation Process", use_column_width=True)


# Load the model from the file
model = joblib.load('decision_tree_model.pkl')

# Streamlit app setup
st.title("Blood Donation Prediction App")

# Choice between file or manual entry
data_or_file = st.radio("Do you want to upload a .csv file or enter data manually?", ('CSV', 'MANUALLY'))

if data_or_file == 'MANUALLY':
    choice = st.radio("Do you want to use dates or RFMT?", ('DATES', 'RFMT'))

    if choice == 'DATES':
        first_donated_date_str = st.date_input("Enter date first donated:")
        last_donated_date_str = st.date_input("Enter last donated date:")
        Frequency = st.number_input("Enter how many times donated:", min_value=1)

        donation_vol = []
        for i in range(1, Frequency + 1):
            volume = st.number_input(f"Enter the volume of donation {i} (in ml):", min_value=0.0)
            donation_vol.append(volume)

        Donated_ml = sum(donation_vol)

        today = datetime.now()
        Recency = (today.year - last_donated_date_str.year) * 12 + (today.month - last_donated_date_str.month)
        Monetary = Donated_ml
        Time = (today.year - first_donated_date_str.year) * 12 + (today.month - first_donated_date_str.month)

    elif choice == 'RFMT':
        Recency = st.number_input("Enter Recency (months):", min_value=0.0)
        Frequency = st.number_input("Enter Frequency (times):", min_value=0.0)
        Monetary = st.number_input("Enter Monetary (c.c. blood):", min_value=0.0)
        Time = st.number_input("Enter Time (months since first donation):", min_value=0.0)

    else:
        st.error("Invalid choice. Please restart and enter 'DATES' or 'RFMT'.")

    if st.button("Predict"):
        X_test = np.array([[Recency, Frequency, Monetary, Time]])
        predictions = model.predict(X_test)
        if predictions == [0]:
            st.write("Will donate? No")
        else:
            st.write("Will donate? Yes")

elif data_or_file == 'CSV':
    file = st.file_uploader("Upload a CSV file", type=["csv"])

    if file:
        df = pd.read_csv(file)
        df.columns = ["Recency", "Frequency", "Monetary", "Time", "Donate"]
        df_drop = df[["Recency", "Frequency", "Monetary", "Time"]]

        if st.button("Predict"):
            # Create X_test array
            X_test = df_drop
            predictions = model.predict(X_test)
            df["predictions"] = predictions
            percentage_counts = df["predictions"].value_counts(normalize=True).round(2) * 100

            st.write("Prediction Results:")
            st.dataframe(df.head(10))
            st.write("Prediction Distribution:")
            st.write(percentage_counts)
