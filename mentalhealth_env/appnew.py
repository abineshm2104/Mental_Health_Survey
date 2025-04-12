import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import streamlit as st

# Load datasets
train_df = pd.read_csv("mentalhealth_env/data/train.csv")
test_df = pd.read_csv("mentalhealth_env/data/test.csv")
sample_submission = pd.read_csv("mentalhealth_env/data/sample_submission.csv")


# Preprocessing
def preprocess_data(df, is_train=True):
    df = df.drop(columns=["id", "Name", "City"], errors='ignore')  # Drop unnecessary columns
    
    # Handle missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    encoders = {}
    for col in categorical_cols:
        encoders[col] = LabelEncoder()
        df[col] = encoders[col].fit_transform(df[col])
    
    if is_train:
        X = df.drop(columns=["Depression"], errors='ignore')
        y = df["Depression"].values
        return X, y, encoders
    else:
        return df

# Apply preprocessing
X, y, encoders = preprocess_data(train_df)
test_X = preprocess_data(test_df, is_train=False)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)
test_X = scaler.transform(test_X)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
test_X_tensor = torch.tensor(test_X, dtype=torch.float32)

# DataLoader
train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32, shuffle=False)

# Define PyTorch model
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

# Model setup
model = DepressionModel(input_dim=X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def train_model(model, train_loader, val_loader, epochs=20):
    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}: Validation Accuracy = {accuracy:.2f}%")

# Train the model
train_model(model, train_loader, val_loader)

# Generate predictions for test set
model.eval()
predictions = []
with torch.no_grad():
    outputs = model(test_X_tensor)
    _, predicted = torch.max(outputs, 1)
    predictions = predicted.numpy()

# Save submission file
sample_submission["Depression"] = predictions
sample_submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

# Streamlit App
def run_app():
    st.title("Depression Prediction")

    user_input = []
    
    for col in train_df.columns:
        if col not in ["id", "Name", "City", "Depression"]:
            if col in encoders:  # If it's a categorical column
                value = st.selectbox(f"Select {col}", train_df[col].unique())
                value = encoders[col].transform([value])[0]  # Encode category
            else:  # Numerical columns
                value = st.number_input(f"Enter {col}", value=float(train_df[col].median()))
            user_input.append(value)
    
    if st.button("Predict"):
        try:
            input_array = np.array(user_input, dtype=float).reshape(1, -1)
            input_tensor = torch.tensor(scaler.transform(input_array), dtype=torch.float32)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(output, 1)
            
            st.write("Predicted Depression Status:", "Yes" if predicted.item() == 1 else "No")
            st.write(f"Confidence Score: {probs[0][predicted].item():.2f}")
        
        except Exception as e:
            st.error(f"Error processing input: {e}")

if __name__ == "__main__":
    run_app()
