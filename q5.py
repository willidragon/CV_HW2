#q5.py

import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torchsummary import summary
import torch.nn as nn
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt


class ResNet50WithFC(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet50WithFC, self).__init__()
        resnet = models.resnet50()
        # Modify the last fully connected layer to have num_classes output units
        resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
        self.resnet = resnet

    def forward(self, x):
        # Forward pass through the network
        return self.resnet(x)

def show_images_q5():
    file_path = "inference_dataset"
    # Define a transform to resize the image and convert it to a tensor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Load the dataset from the file path
    dataset = datasets.ImageFolder(root=file_path, transform=transform)
    
    # Create a DataLoader to handle batching of images
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Store one image per class
    images_per_class = {}
    for _, (image, label) in enumerate(data_loader):
        if label.item() not in images_per_class:
            images_per_class[label.item()] = image.squeeze().permute(1, 2, 0).numpy()
        if len(images_per_class) == len(dataset.classes):
            break  # Stop after getting one image from each class

    # Plot one image per class
    fig, axs = plt.subplots(1, len(images_per_class), figsize=(15, 5))
    for idx, (class_idx, image) in enumerate(images_per_class.items()):
        ax = axs[idx] if len(images_per_class) > 1 else axs
        ax.imshow(image)
        ax.set_title(dataset.classes[class_idx])
        ax.axis('off')
    plt.show()

def show_model_structure_q5(model_path):
    # Create an instance of your custom model
    model = ResNet50WithFC()

    # Load the weights from the state dictionary
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)

    # Print the model structure to the terminal
    summary(model, input_size=(3, 224, 224))

# Replace with your actual model path
# show_model_structure('model/cat_dog.pth')
def load_model(model_path):
    model = ResNet50WithFC(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

def predict(model, image):
    # Perform inference
    with torch.no_grad():
        outputs = model(image)
    # Apply threshold to get prediction
    _, predicted = torch.max(outputs, 1)
    return 'Cat' if predicted.item() == 0 else 'Dog'


def show_comparison():

    image_path = "model/q5_compare/model_compare.png"
     # Open the image file
    image = Image.open(image_path)

    # Create a new figure and axis
    fig, ax = plt.subplots()

    # Display the image on the axis
    ax.imshow(image)

    # Remove the axis labels and ticks
    ax.axis('off')

    # Show the plot
    plt.show()

def show_inference_catdog(model_path, image_path):
    
    # Define the image transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224
        transforms.ToTensor()           # Convert the image to a PyTorch tensor
    ])
    
    # Load and transform the image
    image = Image.open(image_path)              # Load the image
    image_transformed = transform(image).unsqueeze(0)  # Transform and add batch dimension
    
    # Load the model
    model = load_model(model_path)
    
    # Perform inference and get prediction
    prediction = predict(model, image_transformed)

    return prediction






