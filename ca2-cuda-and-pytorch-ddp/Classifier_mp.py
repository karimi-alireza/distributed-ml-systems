# https://gemini.google.com/share/ac438c5ae73e
# my prompt: make this code adapted to what I have written in the Classifier.py file.
# https://gemini.google.com/share/1cbeb661d50e
# my prompt: Add --batch_size and --backend arguments through command line (like what mvajhi did in his github)
# my prompt: print every epoch
# https://gemini.google.com/share/d4f89e80c68f
# my prompt: Do the allreduce to the code to get the global accuracy
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import datetime
import os
import socket
import argparse
from contextlib import closing

# --- Configuration (Defaults) ---
DATA_PATH = "/storage/dmls/stl10_data"
LEARNING_RATE = 0.001
NUM_EPOCHS = 20

# --- Helper Functions ---
def setup(rank, world_size, master_port, backend, timeout):
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = master_port
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size, timeout=timeout)

def find_free_port():
    """ Finds a free port on localhost """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

# --- Model (STL-10 Architecture) ---
class my_CNN(nn.Module):
    def __init__(self):
        super(my_CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 192, 3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(192, 384, 3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(384 * 12 * 12, 1800)
        self.drop = nn.Dropout(0.25)
        self.fc2 = nn.Linear(1800, 360)
        self.fc3 = nn.Linear(360, 10)

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

# --- Data Loading ---
def load_data(rank, world_size, batch_size):
    transform = transforms.Compose([transforms.ToTensor()])

    # STL-10 Train
    train_set = torchvision.datasets.STL10(root=DATA_PATH, split='train', download=False, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, sampler=train_sampler, batch_size=batch_size,
        shuffle=False, persistent_workers=True, num_workers=2, pin_memory=True
    )

    # STL-10 Test
    test_set = torchvision.datasets.STL10(root=DATA_PATH, split='test', download=False, transform=transform)
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_set, num_replicas=world_size, rank=rank
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, sampler=test_sampler, batch_size=batch_size,
        shuffle=False, persistent_workers=True, num_workers=2, pin_memory=True
    )
    
    return train_loader, test_loader

# --- Training Loop ---
def train(rank, world_size, master_port, backend, timeout, args):
    setup(rank, world_size, master_port, backend, timeout)
    torch.cuda.set_device(rank)
    
    train_loader, test_loader = load_data(rank, world_size, args.batch_size)
    
    model = my_CNN().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=LEARNING_RATE)
    
    start_time = time.time()
    
    if rank == 0:
        print(f"Starting training on {world_size} GPUs | Batch Size (per GPU): {args.batch_size} | Backend: {backend}")

    for epoch in range(NUM_EPOCHS):
        # 1. Training Phase
        train_loader.sampler.set_epoch(epoch)
        ddp_model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(rank), labels.to(rank)
            
            optimizer.zero_grad()
            outputs = ddp_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)

        # 2. Evaluation Phase (Global Aggregation)
        ddp_model.eval()
        local_correct = 0
        local_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(rank), labels.to(rank)
                outputs = ddp_model(images)
                predictions = torch.max(outputs, 1)[1]
                local_correct += (predictions == labels).sum().item()
                local_total += labels.size(0)
        
        # --- NEW: Aggregate results from all GPUs ---
        # Convert to tensor to allow Distributed communication
        stats_tensor = torch.tensor([local_correct, local_total], dtype=torch.float32, device=rank)
        
        # Sum the stats across all workers (rank 0 + rank 1 + ...)
        dist.all_reduce(stats_tensor, op=dist.ReduceOp.SUM)
        
        # Extract the summed values
        global_correct = stats_tensor[0].item()
        global_total = stats_tensor[1].item()
        
        # Calculate global accuracy
        global_accuracy = 100 * global_correct / global_total
        
        if rank == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] | Loss: {avg_loss:.4f} | Global Accuracy: {global_accuracy:.2f}%")

    end_time = time.time()
    cuda_mem = torch.cuda.max_memory_allocated(device=rank)
    
    if rank == 0:
        print("--------------------------------------------------")
        print(f"Total Training Time: {end_time - start_time:.2f}s")
        print(f"Max Memory Allocated: {cuda_mem / (1024 ** 2):.2f} MB")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch STL-10 Training with DDP')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training (default: 32)')
    parser.add_argument('--backend', type=str, default='nccl', choices=['nccl', 'gloo'], help='Distributed backend (default: nccl)')
    args = parser.parse_args()

    world_size = torch.cuda.device_count()
    master_port = find_free_port()
    timeout = datetime.timedelta(seconds=10)
    
    mp.spawn(train, 
             nprocs=world_size, 
             args=(world_size, master_port, args.backend, timeout, args), 
             join=True)
