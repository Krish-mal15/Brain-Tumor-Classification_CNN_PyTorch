import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cpu')

train_path = 'tumor-data/Training'
test_path = 'tumor-data/Testing'

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

epochs = 8
batch_size = 4
learning_rate = 0.001

transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ]
)

train_dataset = ImageFolder(root=train_path, transform=transform)
test_dataset = ImageFolder(root=test_path, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


dataiter = iter(train_loader)
images, labels = next(dataiter)

# imshow(torchvision.utils.make_grid(images))


class TumorModel(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()

        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # self.conv_layer3 = nn.Sequential(
        #     nn.Conv2d(in_channels=hidden_units,
        #               out_channels=hidden_units,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=hidden_units,
        #               out_channels=hidden_units,
        #               kernel_size=3,
        #               stride=1,
        #               padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2)
        # )

        self.conv_output_size = hidden_units * (224 // 4) * (224 // 4)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.conv_output_size,
                      out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        # x = self.conv_layer3(x)

        x = self.classifier(x)

        return x


model = TumorModel(input_shape=3,
                    hidden_units=4,
                    output_shape=len(class_names)).to(device)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adamax(params=model.parameters(), lr=learning_rate)

# n_total_steps = len(train_loader)
# for epoch in range(epochs):
#
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)
#
#         # Forward pass
#         outputs = model(images)
#         loss_val = loss(outputs, labels)
#
#
#         optimizer.zero_grad()
#         loss_val.backward()
#         optimizer.step()
#
#         # if (i+1) % 2000 == 0:
#         print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss_val.item():.4f}')
#
# # print("Training Complete")
# torch.save(model.state_dict(), 'Tumor_Model.pth')
#
# with torch.no_grad():
#     n_correct = 0
#     n_samples = 0
#     n_class_correct = [0 for i in range(4)]
#     n_class_samples = [0 for i in range(4)]
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         # max returns (value ,index)
#         _, predicted = torch.max(outputs, 1)
#         n_samples += labels.size(0)
#         n_correct += (predicted == labels).sum().item()
#
#         for i in range(batch_size):
#             label = labels[i]
#             pred = predicted[i]
#             if (label == pred):
#                 n_class_correct[label] += 1
#             n_class_samples[label] += 1
#
#     acc = 100.0 * n_correct / n_samples
#     print(f'Accuracy of the network: {acc} %')
#
#     for i in range(4):
#         acc = 100.0 * n_class_correct[i] / n_class_samples[i]
#         print(f'Accuracy of {class_names[i]}: {acc} %')

