import imageio.v2 as imageio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

def load_and_display(image_path, title="Image"):
    """Load and display image"""
    img = imageio.imread(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    plt.show()
    return img

def apply_filters(image):
    """Apply various filters to image"""
    # Convert to grayscale if image is RGB
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2).astype(np.uint8)
    else:
        gray = image

    # Gaussian blur
    gaussian_blur = ndimage.gaussian_filter(gray, sigma=2)
    
    # Sobel edge detection
    sobel_x = ndimage.sobel(gray, axis=0)
    sobel_y = ndimage.sobel(gray, axis=1)
    edge_sobel = np.hypot(sobel_x, sobel_y)
    edge_sobel = edge_sobel / edge_sobel.max() * 255
    
    # Median filter
    median = ndimage.median_filter(gray, size=3)
    
    return {
        'gaussian': gaussian_blur,
        'edge': edge_sobel.astype(np.uint8),
        'median': median
    }

def display_results(original, filtered_images):
    """Display original and filtered images"""
    plt.figure(figsize=(15, 10))
    
    # Original
    plt.subplot(2, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Gaussian Blur
    plt.subplot(2, 2, 2)
    plt.imshow(filtered_images['gaussian'], cmap='gray')
    plt.title('Gaussian Blur')
    plt.axis('off')
    
    # Edge Detection
    plt.subplot(2, 2, 3)
    plt.imshow(filtered_images['edge'], cmap='gray')
    plt.title('Sobel Edge Detection')
    plt.axis('off')
    
    # Median Filter
    plt.subplot(2, 2, 4)
    plt.imshow(filtered_images['median'], cmap='gray')
    plt.title('Median Filter')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Load image
    image_path = "C:\\Users\\muham\\Downloads\\coba.jpg"  
    try:
        original_image = load_and_display(image_path, "Original Image")
        
        # Apply filters
        filtered = apply_filters(original_image)
        
        # Display results
        display_results(original_image, filtered)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()