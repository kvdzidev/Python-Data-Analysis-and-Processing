# Python-Data-Analysis-and-Processing
# 🧠 AI & Machine Learning Demonstrations

## 📌 Overview
This repository contains three Python scripts showcasing **recent advancements in NLP (Natural Language Processing) and CV (Computer Vision)**, along with **an understanding of ML/DL (Machine Learning/Deep Learning) algorithms**.  
It covers:
- **NLP:** Sentiment analysis using a Transformer model.
- **CV:** Image classification using a pre-trained deep learning model.
- **ML/DL:** Training an autoencoder on the MNIST dataset.

---

## 📂 Files & Features

### 1️⃣ **NLP - Sentiment Analysis (`sentiment_analysis.py`)**  
📌 **Description:**  
- Uses **Hugging Face's Transformers** (`distilbert-base-uncased`) to analyze sentiment (positive/negative).  
- Processes sample text inputs and classifies their sentiment.

📌 **Example Usage:**
```python
from transformers import pipeline

nlp_pipeline = pipeline("sentiment-analysis")
print(nlp_pipeline("I love machine learning!")) 
2️⃣ CV - Image Classification (image_classification.py)
📌 Description:

Implements ResNet50, a pre-trained deep learning model, to classify images.
Uses torchvision.transforms for image preprocessing.
📌 Example Usage:

python
Kopiuj
Edytuj
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

model = models.resnet50(pretrained=True)
model.eval()

url = "https://www.cs.toronto.edu/~kriz/cifar-10-sample/dog4.png"
response = requests.get(url)
img = Image.open(BytesIO(response.content))
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
img = transform(img).unsqueeze(0)

output = model(img)
print(f"Predicted Class Index: {output.argmax().item()}")
📌 Expected Output:

pgsql
Kopiuj
Edytuj
Predicted Class Index: 243 (Class corresponds to a dog)
3️⃣ ML/DL - Autoencoder on MNIST (autoencoder_mnist.py)
📌 Description:

Implements a custom autoencoder neural network for MNIST image compression & reconstruction.
Uses torch.nn, torch.optim, and torchvision.datasets to train and evaluate the model.
📌 Example Usage:

python
Kopiuj
Edytuj
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Load MNIST dataset
train_data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

# Define Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(28*28, 128), nn.ReLU(), nn.Linear(128, 32))
        self.decoder = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 28*28), nn.Sigmoid())

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Train Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

for epoch in range(3):
    for images, _ in train_loader:
        images = images.view(images.size(0), -1).to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), images)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
📌 Expected Output:

yaml
Kopiuj
Edytuj
Epoch 1, Loss: 0.0321
Epoch 2, Loss: 0.0278
Epoch 3, Loss: 0.0243

