import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pydicom
import numpy as np

# =========================
#   1. Paths to datasets (LOCAL)
# =========================
ct_dir = r"D:\important\XMEDFUSION\Data\train"
xray_dir = r"D:\important\XMEDFUSION\IU-DATASET"

# =========================
#   2. Transform pipeline
# =========================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ensure grayscale
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# =========================
#   Helper: Load PNG/JPG/DICOM
# =========================
def load_image(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext in [".png", ".jpg", ".jpeg"]:
        img = Image.open(path).convert("L")
    elif ext == ".dcm":
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array.astype("float32")

        # normalize pixel values (0–255) for PIL
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        img = (img * 255).astype("uint8")

        img = Image.fromarray(img).convert("L")
    else:
        raise ValueError(f"Unsupported format: {ext}")
    return img

# =========================
#   3. Custom Dataset
# =========================
class ModalityDataset(Dataset):
    def __init__(self, ct_dir, xray_dir, transform=None, limit=500):
        self.samples = []
        self.transform = transform

        # Collect CT images (png/jpg/dcm, recursive)
        ct_images = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.dcm"):
            ct_images += glob.glob(os.path.join(ct_dir, "**", ext), recursive=True)
        for img in ct_images[:limit]:
            self.samples.append((img, 0))  # 0 = CT

        # Collect XRay images (png/jpg only here, add dcm if needed)
        xray_images = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.dcm"):
            xray_images += glob.glob(os.path.join(xray_dir, ext))
        for img in xray_images[:limit]:
            self.samples.append((img, 1))  # 1 = XRay

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = load_image(path)
        if self.transform:
            img = self.transform(img)
        return img, label, path

# =========================
#   4. CNN Classifier
# =========================
class ModalityClassifier(nn.Module):
    def __init__(self):
        super(ModalityClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 0 = CT, 1 = XRay
        )

    def forward(self, x):
        return self.net(x)

# =========================
# =========================
#   5. Train or Load Model
# =========================
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ModalityDataset(ct_dir, xray_dir, transform=transform, limit=1000)  # subset for speed
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

classifier = ModalityClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.001)

# Path where model will be saved
MODEL_PATH = "modality_classifier.pth"

if os.path.exists(MODEL_PATH):
    print(f"Loading saved model from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    classifier.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
else:
    print("No saved model found. Training new classifier...")
    num_epochs = 5  # increase for real training
    for epoch in range(num_epochs):
        classifier.train()
        running_loss = 0.0

        # tqdm progress bar per batch
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        for imgs, labels, _ in loop:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = classifier(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {running_loss/len(dataloader):.4f}")

    # Save trained model
    torch.save({
        "model_state": classifier.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


# =========================
#   6. Router Functions
# =========================
def process_ct_agent(path):
    print(f"✅ Routed to CT Agent: {path}")

def process_xray_agent(path):
    print(f"✅ Routed to XRay Agent: {path}")

def classify_and_route(img_path):
    classifier.eval()
    img = load_image(img_path)
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = classifier(img)
        pred = torch.argmax(output, 1).item()

    if pred == 0:
        process_ct_agent(img_path)
    else:
        process_xray_agent(img_path)
# =========================
#   7. Test the Router
# =========================
test_samples = [
    r"D:\important\XMEDFUSION\Data\test\large.cell.carcinoma\000111 (2).png",
    r"D:\important\XMEDFUSION\IU-DATASET\CXR4_IM-2050-2001.png"
]

for path in test_samples:
    classify_and_route(path)
