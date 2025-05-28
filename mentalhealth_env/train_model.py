import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ----------------------- #
#     Neural Network      #
# ----------------------- #
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

# ----------------------- #
#     Data Preprocessing  #
# ----------------------- #
def preprocess_data(df, encoders=None, fit_encoders=True):
    df = df.drop(columns=["id", "Name", "City"], errors="ignore")

    # Fill missing values
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

    # Encode categorical columns
    if fit_encoders:
        encoders = {}
        for col in cat_cols:
            encoders[col] = LabelEncoder()
            df[col] = encoders[col].fit_transform(df[col])
    else:
        for col in cat_cols:
            df[col] = encoders[col].transform(df[col])

    return df, encoders

# ----------------------- #
#         Training        #
# ----------------------- #
def train():
    # Load data
    df = pd.read_csv("/workspaces/Mental_Health_Survey/mentalhealth_env/data/train.csv")

    # Preprocess
    df, encoders = preprocess_data(df)
    X = df.drop(columns=["Depression"])
    y = df["Depression"].values

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    # Dataloaders
    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=32, shuffle=False)

    # Model, Loss, Optimizer
    model = DepressionModel(input_dim=X.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train
    print("Training started...")
    for epoch in range(10):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Validation Accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                _, preds = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (preds == y_batch).sum().item()

        acc = 100 * correct / total
        print(f"Epoch {epoch+1} - Validation Accuracy: {acc:.2f}%")

    # Save model
    torch.save(model.state_dict(), "depression_model.pt")
    print("✅ Model saved to 'depression_model.pt'")

if __name__ == "__main__":
    train()
