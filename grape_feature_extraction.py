import cv2
import numpy as np
from skimage import filters, measure
import pandas as pd

# Load the image
image_path = r"D:\4 TH YEAR\FINAL YEAR PRO\GrapesDataset\GrapesNet\Dataset 1\Processed\Image(1).jpg"
image = cv2.imread(image_path)

# Check if the image was loaded correctly
if image is None:
    print(f"Error: Could not load image at {image_path}")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 1: CLAHE contrast map
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
contrast_map = clahe.apply(gray)

# Step 2: Homogeneity map using mean filter
homogeneity_map = filters.rank.mean(gray, np.ones((10, 10)))

# Step 3: Contour detection
edges = cv2.Canny(image, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Initialize a list to store feature vectors for each grape
node_features = []

# Maximum values for normalization (set these based on your dataset)
max_contrast = 255  # Maximum possible contrast score
max_homogeneity = 255  # Maximum possible homogeneity score
max_area = 1000  # Adjust according to expected maximum area
max_perimeter = 200  # Adjust according to expected maximum perimeter
threshold_good = 2.5  # Define your threshold for "Good"
threshold_average = 1.5  # Define your threshold for "Average"

for contour in contours:
    # Create a mask for the current contour
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)  # Fill contour for mask

    # Feature 1: Contrast score
    contrast_score = np.mean(contrast_map[mask == 255])
    
    # Feature 2: Homogeneity score
    homogeneity_score = np.mean(homogeneity_map[mask == 255])
    
    # Shape features using region properties
    label_img = measure.label(mask, connectivity=1)
    region = measure.regionprops(label_img)[0] if len(measure.regionprops(label_img)) > 0 else None
    
    if region:
        area = region.area
        perimeter = region.perimeter
        eccentricity = region.eccentricity
    else:
        area = perimeter = eccentricity = 0  # Default to zero if no region found

    # Print feature values for diagnosis
    print(f"Contrast: {contrast_score}, Homogeneity: {homogeneity_score}, Area: {area}, Perimeter: {perimeter}, Eccentricity: {eccentricity}")

    # Calculate score based on features
    score = (contrast_score / max_contrast) + \
            (homogeneity_score / max_homogeneity) + \
            (area / max_area) + \
            (perimeter / max_perimeter) + \
            (1 - eccentricity)

    # Categorize based on score
    if score >= threshold_good:
        label = "Good"
    elif score >= threshold_average:
        label = "Average"
    else:
        label = "Bad"

    # Combine features and label into a dictionary
    feature_vector = {
        "contrast_score": contrast_score,
        "homogeneity_score": homogeneity_score,
        "area": area,
        "perimeter": perimeter,
        "eccentricity": eccentricity,
        "score": score,
        "label": label
    }
    
    node_features.append(feature_vector)

# Create a DataFrame from the list of feature vectors
df = pd.DataFrame(node_features)

# Save the DataFrame to an Excel file
output_path = r"D:\4 TH YEAR\FINAL YEAR PRO\GrapesDataset\grape_features.xlsx"
df.to_excel(output_path, index=False)

print(f"Grape features saved to {output_path}")
