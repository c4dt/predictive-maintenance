import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights


class BinaryMVtec(Dataset):
    def __init__(self, root="./data", train=True, transform=None, normal_class="good"):
        self.transform = transform

        if train:
            data_dir = os.path.join(root, "train")
        else:
            data_dir = os.path.join(root, "test")

        self.dataset = ImageFolder(data_dir)

        self.normal_class = self.dataset.class_to_idx[normal_class]
        self.images, self.labels = self.create_binary_dataset()

    def create_binary_dataset(self):
        normal_indices = []
        abnormal_indices = []
        for i, (image, label) in enumerate(self.dataset):
            if label == self.normal_class:
                normal_indices.append(i)
            else:
                abnormal_indices.append(i)

        binary_indices = normal_indices + abnormal_indices
        binary_images = [self.dataset[i][0] for i in binary_indices]
        binary_labels = [int(i in normal_indices) for i in binary_indices]
        return binary_images, binary_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    def get_labels(self):
        return self.labels


# This is to reduce the CIFAR10 images to a binary set, else there are
# 10 labels, and this can't be used for a binary classificator.
class BinaryCIFAR10(Dataset):
    def __init__(self, root="./data", train=True, transform=None):
        self.transform = transform

        cifar10_dataset = datasets.CIFAR10(root=root, train=train, download=True)
        images, labels = np.array(cifar10_dataset.data), np.array(
            cifar10_dataset.targets
        )

        self.images, self.labels = self.create_binary_dataset(images, labels)

    def create_binary_dataset(self, images, labels):
        binary_indices = np.where(labels < 2)[0]
        binary_images = images[binary_indices]
        binary_labels = labels[binary_indices].flatten()
        return binary_images, binary_labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


def main(device, mvtec=False):
    # Load your dataset
    # Replace this with your actual dataset loading code
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if mvtec:
        train_dataset = BinaryMVtec(train=True, transform=transform)
        test_dataset = BinaryMVtec(train=False, transform=transform)
    else:
        train_dataset = BinaryCIFAR10(train=True, transform=transform)
        test_dataset = BinaryCIFAR10(train=False, transform=transform)

    train_loader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Load pre-trained ResNet model
    resnet_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Remove the last layer to use the features as input to the MLP
    resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])

    # Define the MLP classifier
    class MLPClassifier(nn.Module):
        def __init__(self, input_size, output_classes):
            super(MLPClassifier, self).__init__()
            self.fc1 = nn.Linear(input_size, 256)
            self.fc2 = nn.Linear(256, 128)
            self.fc3 = nn.Linear(128, output_classes)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Create MLP model
    mlp_model = MLPClassifier(input_size=512, output_classes=2)

    # Combine ResNet and MLP models
    class CombinedModel(nn.Module):
        def __init__(self, resnet, mlp):
            super(CombinedModel, self).__init__()
            self.resnet = resnet
            self.mlp = mlp

        def forward(self, x):
            x = self.resnet(x)
            x = self.mlp(x)
            return x

    combined_model = CombinedModel(resnet_model, mlp_model).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(combined_model.parameters(), lr=0.001)

    # Train the model
    combined_model.train()
    num_epochs = 30

    print(f"Start training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = combined_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")

    # Evaluate the model
    combined_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = combined_model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    print(f"Test accuracy: {test_accuracy}")


if __name__ == "__main__":
    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"

    print("Found device: " + device_str)
    device = torch.device(device_str)

    main(device, mvtec=False)
