import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from timm import create_model

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_available():
    device = "mps"
# Define Swin Transformer Model (Without Pretrained Weights)
class CustomSwin(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomSwin, self).__init__()
        self.swin = create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=num_classes)

    def forward(self, x):
        return self.swin(x)

# Load dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root="dataset/train_faces", transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize model
model = CustomSwin(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss / len(train_loader):.4f}")

# Save trained model
torch.save(model.state_dict(), "models/swin_model_custom.pth")
