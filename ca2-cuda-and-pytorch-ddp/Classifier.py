# https://gemini.google.com/share/3890cb051a2b
# https://gemini.google.com/share/739edee4facf
# my prompt: how to load the dataset based on the given data-path
# https://gemini.google.com/share/2806ca9647a9
# my prompt: edit some of the dimentions based on STL-10 dataset (96x96 images) edit the maxxpooling layers and the first fully connected layer accordingly
# https://gemini.google.com/share/45ce6cfeec94
# my prompt: print at each epoch

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time

device = "cuda:0" if torch.cuda.is_available() else "cpu"

data_path = "/storage/dmls/stl10_data"
train_batch_size = 32
test_batch_size = 32
learning_rate = 0.001
num_epochs = 20

# Define transforms (STL-10 images are 96x96)
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load STL-10 Train Set
train_set = torchvision.datasets.STL10(
    root=data_path,
    split='train',       
    download=False,      
    transform=transform
)

# Load STL-10 Test Set
test_set = torchvision.datasets.STL10(
    root=data_path,
    split='test',        
    download=False,
    transform=transform
)

# Create Loaders
train_loader = torch.utils.data.DataLoader(
    train_set, 
    batch_size=train_batch_size, 
    shuffle=True,        
    num_workers=2 # Increased to 2 for better speed
)

test_loader = torch.utils.data.DataLoader(
    test_set, 
    batch_size=test_batch_size, 
    shuffle=False,
    num_workers=2
)

print(f"Train samples: {len(train_set)}") 
print(f"Test samples: {len(test_set)}")   

class my_CNN(nn.Module):
    def __init__(self):
        super(my_CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 96 -> 48 -> 24 -> 12. Final feature map is 12x12.
        self.fc1 = nn.Linear(in_features=384 * 12 * 12, out_features=1800)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(in_features=1800, out_features=360)
        self.fc3 = nn.Linear(in_features=360, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
    

if __name__ == "__main__":
    model = my_CNN()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    start_time = time.time()
    
    # --- TRAINING LOOP START ---
    for epoch in range(num_epochs):
        model.train() # Set model to training mode (enables Dropout/BatchNorm)
        running_loss = 0.0
        
        # 1. Train on Training Set
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Calculate average loss for this epoch
        avg_train_loss = running_loss / len(train_loader)

        # 2. Evaluate on Test Set (Once per epoch)
        model.eval() # Set model to evaluation mode (disables Dropout)
        correct = 0
        total = 0
        
        with torch.no_grad(): # Disable gradient calculation for testing (saves memory/time)
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                # Get predictions
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        
        # Print stats for this epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_train_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # --- TRAINING LOOP END ---

    end_time = time.time()
    cuda_mem = torch.cuda.max_memory_allocated(device=device) 
    
    print("--------------------------------------------------")
    print(f"Total Training time: {end_time - start_time:.2f} seconds")
    print(f"Max Cuda Memory Usage: {cuda_mem / (1024 ** 2):.2f} MB")
