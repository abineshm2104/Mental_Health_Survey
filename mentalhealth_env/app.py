import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import streamlit as st

# ------------------------- #
#       Configuration       #
# ------------------------- #
MODEL_PATH = "depression_model.pt"

# ------------------------- #
#        Data Loader        #
# ------------------------- #
@st.cache_resource
def load_data():
    train_df = pd.read_csv("mentalhealth_env/data/train.csv")
    test_df = pd.read_csv("mentalhealth_env/data/test.csv")
    sample_submission = pd.read_csv("mentalhealth_env/data/sample_submission.csv")
    return train_df, test_df, sample_submission

# ------------------------- #
#    Preprocessing Setup    #
# ------------------------- #
def preprocess_data(df, encoders=None, is_train=True):
    df = df.drop(columns=["id", "Name", "City"], errors='ignore')
    num_cols = df.select_dtypes(include=[np.number]).columns
    obj_cols = df.select_dtypes(include=['object']).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[obj_cols] = df[obj_cols].fillna(df[obj_cols].mode().iloc[0])

    if encoders is None:
        encoders = {}
        for col in obj_cols:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col])
    else:
        for col in obj_cols:
            if col in encoders:
                df[col] = encoders[col].transform(df[col])

    if is_train:
        X = df.drop(columns=["Depression"], errors='ignore')
        y = df["Depression"].values
        return X, y, encoders
    else:
        return df

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
#       Train Function      #
# ------------------------- #
def train_model(X_train, y_train, X_val, y_val, input_dim):
    model = DepressionModel(input_dim=input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

    for epoch in range(5):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

        acc = 100 * correct / total
        st.write(f"Epoch {epoch+1}: Validation Accuracy = {acc:.2f}%")

    # Save model
    torch.save(model.state_dict(), MODEL_PATH)
    return model

# ------------------------- #
#     Streamlit App UI      #
# ------------------------- #
def run_app():
    st.title("🧠 Depression Prediction App")

    train_df, test_df, sample_submission = load_data()

    # Sidebar for actions
    st.sidebar.header("Actions")

    # Train model button
    if st.sidebar.button("🔁 Train Model"):
        with st.spinner("Preprocessing and training..."):
            X, y, encoders = preprocess_data(train_df)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            y_train_tensor = torch.tensor(y_train, dtype=torch.long)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)

            model = train_model(X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, input_dim=X.shape[1])
            st.success("✅ Model trained and saved successfully!")

            # Save encoders and scaler (for reuse in prediction)
            st.session_state["encoders"] = encoders
            st.session_state["scaler"] = scaler

    # Prediction
    st.subheader("📋 Enter Patient Details to Predict Depression")

    encoders = st.session_state.get("encoders")
    scaler = st.session_state.get("scaler")

    if encoders and scaler and os.path.exists(MODEL_PATH):
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

                model = DepressionModel(input_dim=input_tensor.shape[1])
                model.load_state_dict(torch.load(MODEL_PATH))
                model.eval()

                output = model(input_tensor)
                probs = torch.softmax(output, dim=1)
                _, pred = torch.max(output, 1)

                st.write("Prediction:", "Yes" if pred.item() == 1 else "No")
                st.write(f"Confidence: {probs[0][pred].item():.2f}")

            except Exception as e:
                st.error(f"Error during prediction: {e}")
    else:
        st.info("Please train the model first using the sidebar.")

# Run app
if __name__ == "__main__":
    run_app()
