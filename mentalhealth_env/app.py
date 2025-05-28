import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st

# ------------------------- #
#      Constants & Paths    #
# ------------------------- #
MODEL_PATH = "mentalhealth_env/depression_model.pt"

# ------------------------- #
#         Model Class       #
# ------------------------- #
class DepressionModel(nn.Module):
    def __init__(self, input_dim):
        super(DepressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ------------------------- #
#      Load Data & Prep     #
# ------------------------- #
@st.cache_resource
def load_resources():
    train_df = pd.read_csv("mentalhealth_env/data/train.csv")
    test_df = pd.read_csv("mentalhealth_env/data/test.csv")

    # Preprocessing
    df = train_df.drop(columns=["id", "Name", "City", "Depression"], errors="ignore")
    num_cols = df.select_dtypes(include=[np.number]).columns
    obj_cols = df.select_dtypes(include=['object']).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[obj_cols] = df[obj_cols].fillna(df[obj_cols].mode().iloc[0])

    # Label encoding
    encoders = {}
    for col in obj_cols:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])

    # Scaling
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    return train_df, encoders, scaler, df_scaled.shape[1]

# ------------------------- #
#       Streamlit App       #
# ------------------------- #
def run_app():
    st.title("🧠 Depression Prediction App")

    if not os.path.exists(MODEL_PATH):
        st.error("Trained model not found. Please train the model first and place it in the project directory.")
        return

    train_df, encoders, scaler, input_dim = load_resources()

    # Get user inputs
    user_input = []
    for col in train_df.columns:
        if col in ["id", "Name", "City", "Depression"]:
            continue
        if col in encoders:
            value = st.selectbox(f"{col}", train_df[col].dropna().unique())
            try:
                value = encoders[col].transform([value])[0]
            except:
                value = encoders[col].transform([train_df[col].mode()[0]])[0]
        else:
            value = st.number_input(f"{col}", value=float(train_df[col].median()))
        user_input.append(value)

    if st.button("🧮 Predict"):
        try:
            input_array = np.array(user_input).reshape(1, -1)
            input_tensor = torch.tensor(scaler.transform(input_array), dtype=torch.float32)

            model = DepressionModel(input_dim)
            model.load_state_dict(torch.load(MODEL_PATH))
            model.eval()

            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            _, prediction = torch.max(output, 1)

            st.success("Prediction: " + ("Yes" if prediction.item() == 1 else "No"))
            st.info(f"Confidence Score: {probs[0][prediction].item():.2f}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# ------------------------- #
#           Main            #
# ------------------------- #
if __name__ == "__main__":
    run_app()
