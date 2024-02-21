# q3.py
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import mean_squared_error

def calculate_mse(original, reconstructed):
    if original.shape != reconstructed.shape:
        raise ValueError("Original and reconstructed images must have the same shape.")

    original_flat = original.flatten()
    reconstructed_flat = reconstructed.flatten()

    mse = np.mean((original_flat - reconstructed_flat) ** 2)
    return mse

def pca_dimension_reduction(image_path, error_threshold=3.0):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use the original grayscale image (0-255) for reconstruction error calculation
    gray_image_array = np.array(gray_image)
    
    # Still normalize for PCA application
    gray_image_norm = gray_image_array / 255.0
    
    n_samples, n_features = gray_image_norm.shape if gray_image_norm.shape[0] < gray_image_norm.shape[1] else gray_image_norm.shape[::-1]
    gray_image_reshaped = gray_image_norm.reshape((n_samples, n_features))
    
    pca = PCA()
    pca.fit(gray_image_reshaped)

    reconstructed_image = None
    n_components = 0
    error = float('inf')

    h, w = gray_image_array.shape
    min_dim = min(h, w)

    # Initialize variables for minimum components and MSE
    min_components = 1
    reconstruction_error = float('inf')

    for n in range(1, min_dim + 1):
        pca = PCA(n_components=n)
        reduced_data = pca.fit_transform(gray_image_norm.reshape(h, -1))
        reconstructed_data = pca.inverse_transform(reduced_data)
        # Use cv2.normalize to scale reconstructed data back to 0-255 range
        reconstructed_image = cv2.normalize(reconstructed_data, None, 0, 255, cv2.NORM_MINMAX)
        reconstructed_image = reconstructed_image.reshape(h, w).astype(np.uint8)

        # Calculate MSE between original and reconstructed images
        # mse = calculate_mse(gray_image_array, reconstructed_image)
        mse = mean_squared_error(gray_image, reconstructed_image)
        # Check if MSE is less than or equal to the set error_threshold
        if mse <= error_threshold:
            min_components = n
            reconstruction_error = mse
            break
    
    # Plot the grayscale image and the reconstruction image with min_components
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(gray_image_array, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f'Reconstructed Image (n={min_components})')
    plt.axis('off')

    plt.show()

    print(f"Minimum n value: {min_components}, MSE: {reconstruction_error}")
    
    return min_components



# Call the function with the image path
# pca_dimension_reduction("Dataset_CvDl_Hw2/Q3/logo.jpg")

