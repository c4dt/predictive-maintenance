import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.datasets import ImageFolder


class BinaryMVtec(Dataset):
    def __init__(self, root='./data/small', train=True, transform=None, normal_class='good'):
        self.transform = transform

        if train:
            data_dir = os.path.join(root, 'train')
        else:
            data_dir = os.path.join(root, 'test')

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


def main(device):
    # Load your dataset
    # Replace this with your actual dataset loading code
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = BinaryMVtec(train=True, transform=transform)
    test_dataset = BinaryMVtec(train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    # Load pre-trained ResNet model
    resnet_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Remove the last layer to use the features as input to the MLP
    resnet_model = nn.Sequential(*list(resnet_model.children())[:-1])

    # Define the Linear Regression model
    class LinearRegressionModel(nn.Module):
        def __init__(self, input_size, output_classes):
            super(LinearRegressionModel, self).__init__()
            self.linear = nn.Linear(input_size, output_classes)

        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = self.linear(x)
            return x

    # Create Linear Regression model
    linear_model = LinearRegressionModel(input_size=512, output_classes=2)

    # Combine ResNet and Linear Regression models
    class CombinedModel(nn.Module):
        def __init__(self, resnet, linear):
            super(CombinedModel, self).__init__()
            self.resnet = resnet
            self.linear = linear

        def forward(self, x):
            x = self.resnet(x)
            x = self.linear(x)
            return x

    combined_model = CombinedModel(resnet_model, linear_model).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(combined_model.parameters(), lr=0.000001)

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


if __name__ == '__main__':
    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"

    print("Found device: " + device_str)
    device = torch.device(device_str)

    main(device)
