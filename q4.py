# q4.py
import torch
from torchvision import models
from torchsummary import summary
from PIL import Image
import matplotlib.pyplot as plt
import io
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QBuffer, QIODevice

class VGG19BN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(VGG19BN, self).__init__()
        # Instantiate the VGG19 model with batch normalization
        self.model = models.vgg19_bn(pretrained=False)
        # Modify the classifier to match the number of classes (MNIST dataset has 10 classes)
        self.model.classifier[6] = torch.nn.Linear(in_features=4096, out_features=num_classes)

    def forward(self, x):
        return self.model(x)

def show_model_structure(model_path):
    # Initialize the model
    custom_model = VGG19BN(num_classes=10)
    # Load the model weights
    custom_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    # Set the model to evaluation mode
    custom_model.eval()

    # Print the summary of the model
    # Pass only the inner model to summary
    summary(custom_model.model, input_size=(3, 32, 32), device='cpu')

# Usage example (ensure the model path is correct)
# show_model_structure('model/vgg19_bn_mnist.pth')


def showAccuracyAndLoss(image_path):
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


def predict(graffiti_board_pixmap, model_path, num_classes=10):
    # Convert QPixmap to PIL Image
    buffer = QBuffer()
    buffer.open(QIODevice.ReadWrite)
    graffiti_board_pixmap.save(buffer, "PNG")
    pil_img = Image.open(io.BytesIO(buffer.data()))

    # Preprocess the image to match VGG19 input requirements
    preprocess = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=3),  # Convert to three-channel grayscale
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize for three channels
    ])
    img_tensor = preprocess(pil_img)
    img_tensor = img_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model

    # Load the model
    custom_model = VGG19BN(num_classes)
    custom_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    custom_model.eval()

    # Perform the prediction
    with torch.no_grad():
        outputs = custom_model(img_tensor)

    # Get the probability distribution
    probabilities = torch.nn.functional.softmax(outputs, dim=1).numpy().flatten()

    # Plot the histogram and set the labels for the x-axis
    plt.figure(figsize=(8, 6))
    plt.bar(np.arange(num_classes), probabilities, alpha=0.7)
    plt.xticks(np.arange(num_classes))
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')

    for i, prob in enumerate(probabilities):
        plt.text(i, prob, f'{prob:.2f}', ha='center', va='bottom')

    plt.show()

    # Return the predicted class and its probability for further use
    predicted_class = np.argmax(probabilities)
    max_probability = np.max(probabilities)
    return predicted_class, max_probability