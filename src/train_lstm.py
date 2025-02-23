import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os

# Load extracted features for LSTM
class FeatureDataset(Dataset):
    def __init__(self, feature_folder):
        self.feature_files = sorted(os.listdir(feature_folder))
        self.labels = [0 if "real" in feature_folder.lower() else 1 for _ in self.feature_files]
        self.feature_folder = feature_folder

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx):
        feature_path = os.path.join(self.feature_folder, self.feature_files[idx])
        feature = torch.load(feature_path)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label
dataset = FeatureDataset("dataset/extracted_features/")
train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

# Define LSTM Model
class CustomLSTM(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, num_layers=2, output_dim=2):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return self.softmax(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")

# Save model
torch.save(model.state_dict(), "models/lstm_model_custom.pth")
