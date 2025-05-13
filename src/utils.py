import os
import numpy as np
import random
import torch
from ome_types import to_dict

def maskToTruth(mask):
    # Normalize the mask if it has pixel values greater than 1
    if np.max(mask)>1:
        mask = mask / 255.0
    # Return 1 if any pixel is non-zero, otherwise 0
    # using np.any instead of np.max as it will stop as soon as it finds a value of 1
    return int(np.any(mask > 0))



def getTruthLabels(masks):
    # For each patch:
    #   if the mask contains any pixels with the signature
    #       return 1
    #   else 
    #       return 0
    if not isinstance(masks, list):
        raise ValueError("Input must be a list of masks.")
    if not all(isinstance(mask, np.ndarray) for mask in masks):
        raise ValueError("Each mask must be a numpy array.")
    labels = [maskToTruth(mask) for mask in masks]
    return labels

def set_seed(seed=777):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def roi_to_dict(roi, downsample_factor=1):
    """
    Converts a single ROI into a dictionary with ROI ID as the key and the downsampled
    rectangle data (x, y, width, height) as the value.

    Args:
        roi: A single ROI object.
        downsample_factor: Factor by which to downsample the coordinates and dimensions.

    Returns:
        A dictionary with the ROI ID as the key and rectangle data as the value.
    """
    import numpy as np

    roi_data = {}
    roi_name = ""
    roi_id = roi.id  # Extract the ROI ID
    #print("ROI: ",roi_id)
    union_shapes = to_dict(roi.union)  # Convert the union shapes to a dictionary
    if 'labels' in union_shapes:
        for lbl in union_shapes['labels']:
            #print(lbl['text'])
            roi_name = lbl['text']
        
    if 'rectangles' in union_shapes:
        
        for rect in union_shapes['rectangles']:
            # Downsample rectangle dimensions
            roi_data[roi_id] = {
                'x': int(rect['x'] / downsample_factor),
                'y': int(rect['y'] / downsample_factor),
                'width': int(rect['width'] / downsample_factor),
                'height': int(rect['height'] / downsample_factor),
                'name': roi_name
            }

    elif 'polygons' in union_shapes:
        for polygon in union_shapes['polygons']:
            # Extract polygon points and calculate bounding box
            points = polygon['points'].split()  # Points as a space-separated string
            vertices = [tuple(map(float, point.split(','))) for point in points]  # Convert to (x, y) tuples

            # Calculate bounding box
            x_coords, y_coords = zip(*vertices)  # Separate x and y coordinates
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            width = x_max - x_min
            height = y_max - y_min

            # Downsample bounding box dimensions
            roi_data[roi_id] = {
                'x': int(x_min / downsample_factor),
                'y': int(y_min / downsample_factor),
                'width': int(width / downsample_factor),
                'height': int(height / downsample_factor),
                'name': roi_name
            }

    elif 'ellipses' in union_shapes:
        for ellipse in union_shapes['ellipses']:
            # Downsample ellipse dimensions
            roi_data[roi_id] = {
                'x': int(ellipse['x'] / downsample_factor),
                'y': int(ellipse['y'] / downsample_factor),
                'width': int(ellipse['radius_x'] / downsample_factor),
                'height': int(ellipse['radius_y'] / downsample_factor),
                'name': roi_name
            }

    return roi_data

def rois_to_dict(rois, downsample_factor=1):
    """
    Converts a list of ROIs into a single dictionary with all ROI IDs and their corresponding
    downsampled rectangle data (x, y, width, height).

    Args:
        rois: List of ROI objects (e.g., ome_metadata.rois).
        downsample_factor: Factor by which to downsample the coordinates and dimensions.

    Returns:
        A dictionary where each key is an ROI ID, and the value is a dictionary of rectangle data.
    """
    all_roi_data = {}
    for roi in rois:
        roi_data = roi_to_dict(roi, downsample_factor)
        all_roi_data.update(roi_data)
    return all_roi_data


def save_features_with_names(features, image_paths, output_path, model_name, format="csv"):
    """
    Save extracted features along with image names.

    Parameters:
        features (numpy.ndarray): The extracted feature array.
        image_paths (list): List of image file paths.
        output_path (str): Directory to save features.
        model_name (str): Name of the foundation model.
        format (str): File format ('csv' recommended for indexing).
    """
    os.makedirs(output_path, exist_ok=True)
    curr_datetime = time.strftime("%Y%m%d-%H%M%S")
    feature_file = os.path.join(output_path, f"{model_name}_features-{curr_datetime}.{format}")

    # Extract just the file names (without paths)
    image_names = [os.path.basename(path) for path in image_paths]

    # Combine the features and image names into a DataFrame
    df = pd.DataFrame(features)
    df.insert(0, "image_name", image_names)

    if format == "csv":
        df.to_csv(feature_file, index=False)
    elif format == "pkl":
        df.to_pickle(feature_file)
    elif format == "npy":
        np.save(feature_file, df.to_numpy())
    else:
        raise ValueError("Unsupported format. Choose 'csv', 'pkl', or 'npy'.")

    print(f"Features saved to: {feature_file}")

def load_features_from_csv(feature_file):
    """
    Load features from a CSV file.

    Parameters:
        feature_file (str): Path to the CSV file containing features.

    Returns:
        tuple: (image_names, feature_matrix)
    """
    df = pd.read_csv(feature_file)

    # Extract image names and feature values separately
    image_names = df["image_name"].tolist()
    feature_matrix = df.drop(columns=["image_name"]).values  # Convert features to NumPy array

    return image_names, feature_matrix
    