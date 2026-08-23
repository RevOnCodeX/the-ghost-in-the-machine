import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TIER_B_DIR = SCRIPT_DIR.parent
DATA_EMBEDDINGS = TIER_B_DIR / "data/embeddings.npy"
DATA_LABELS = TIER_B_DIR / "data/labels.npy"
MODEL_OUT = TIER_B_DIR / "models/semantic_nn.pt"

class SemanticNN(nn.Module):
    def __init__(self):
        super(SemanticNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)

def train_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32, lr=0.001):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = SemanticNN()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t).item()
            predicted = (val_outputs >= 0.5).float()
            val_acc = (predicted == y_val_t).sum().item() / len(y_val_t)
            
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
            
    return model, history

def main():
    print("Loading embeddings...")
    X = np.load(DATA_EMBEDDINGS)
    y = np.load(DATA_LABELS)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    print("Training Semantic NN...")
    model, history = train_model(X_train, y_train, X_val, y_val, epochs=50)
    
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")
    
    # Save training history for visualization later
    import json
    with open(TIER_B_DIR / "results/training_history.json", "w") as f:
        json.dump(history, f)

if __name__ == "__main__":
    main()
