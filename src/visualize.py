import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage import exposure

def visualise_feature_vector_heatmap(feature_vector):
    """
    Visualizes the feature vector using a heatmap.

    Parameters:
    - feature_vector: 1D torch.Tensor of shape [n_features].
    """
    #feature_vector = feature_vector.squeeze().detach().numpy()  # Remove the batch dimension
    feature_vector = feature_vector.reshape(1, -1)  # Reshape for heatmap

    plt.figure(figsize=(12, 2))
    sns.heatmap(feature_vector, cmap='viridis', annot=False, cbar=True, xticklabels=False)
    plt.title("Feature Vector Heatmap")
    plt.show()

def viz_side_by_side(original_image, feature_map, index=0, cmap='viridis'):
    try:
        # Extract and process the selected original image from the batch
        original_image = original_image

        # Extract and process the selected feature map from the batch
        feature_img = feature_map[index].transpose(0, 2).sum(-1).detach().numpy()

        # Plotting
        plt.figure(figsize=(12, 6))  # Adjust figure size as needed

        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title(f"Original Image {index}")

        # Plot feature map
        plt.subplot(1, 2, 2)
        plt.imshow(feature_img, cmap=cmap)
        plt.title(f"Feature Map {index}")

        plt.show()

    except IndexError:
        print(f"Index {index} is out of bounds for the batch size {original_image.shape[0]}.")
    except Exception as e:
        print(f"An error occurred: {e}")



def visualise_feature_matrix_heatmap(feature_matrix):
    """
    Visualizes a feature matrix using a heatmap.

    Parameters:
    - feature_matrix: NumPy array of shape (n_samples, n_features), where each row is a sample and each column is a feature.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot the heatmap
    sns.heatmap(feature_matrix, cmap='viridis', annot=False, cbar=True)
    
    plt.title("Feature Matrix Heatmap")
    plt.xlabel("Features")
    plt.ylabel("Samples")
    
    plt.show()




def plot_rois_on_image(image, roi_dict, save=False, LEAPID=""):
    """
    Overlays ROI rectangles on an image and labels them with their IDs.

    Args:
        image: The background image (e.g., a 2D NumPy array or an image array).
        roi_dict: Dictionary where keys are ROI IDs and values are dictionaries
                  containing 'x', 'y', 'width', and 'height' for the rectangle.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))

    # Display the image
    ax.imshow(image, cmap='gray', interpolation='nearest')
    
    # Loop through each ROI in the dictionary

    for roi_id, rect in roi_dict.items():
        # Extract rectangle data
        x, y, width, height = rect['x'], rect['y'], rect['width'], rect['height']
        print(f"roi {roi_id} : {x}, {y}, {width}, {height}")
        # Create a rectangle patch
        rectangle = patches.Rectangle(
            (x, y),  # Bottom-left corner
            width,   # Width
            height,  # Height
            linewidth=2,
            edgecolor='blue',
            facecolor='none'
        )
        # Add the rectangle to the plot
        ax.add_patch(rectangle)

        # Add the ROI ID as a label
        ax.text(
            x + width / 2,  # Center of the rectangle (x)
            y + height / 2,  # Center of the rectangle (y)
            #roi_id,          # ROI ID label
            rect['name'],          # ROI ID label
            color='red',
            fontsize=10,
            ha='center',     # Horizontal alignment
            va='center'      # Vertical alignment
        )

    # Customize the plot
    ax.set_title('ROIs Overlaid on Image')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_aspect('equal')  # Keep aspect ratio equal
    #plt.gca().invert_yaxis()  # Invert y-axis to align with image coordinate system
    if save:
        plt.savefig(f"ROIs_over_{LEAPID}.png")
    plt.show()
    plt.close()


    
def overlay_coords_on_img(img, subset_metadata, img_title=""):
    xcoords = subset_metadata.loc[:,"ROICoordinateY_downsampled"] #/ 4).astype(int)

    # Assign the corrected Y coordinate
    ycoords = subset_metadata.loc[:,"ROICoordinateX_downsampled"] #/ 4).astype(int)

    # Overlay corrected coordinates

    plt.scatter(
        xcoords,
        ycoords,
        label="Corrected FOVs",
        alpha=0.8,
        c="red",
        #edgecolors="black",
        s=20
    )
    # Add labels for each point using ROI
    for i, txt in enumerate(subset_metadata['Roi']):
        plt.text(
            xcoords.iloc[i]+20,
            ycoords.iloc[i],
            str(int(txt)),
            fontsize=6,
            ha='left', 
            va='center',  # Center the text vertically
            color="black"
        )

    # Set plot parameters
    plt.title(f"Corrected FOVs for {img_title}")
    plt.legend()

    #plt.axis("equal")

    plt.imshow(img) 
    plt.show()

def display_spatial_threshold(gene, he_img, metadata, expr_matrix, threshold ):

    # Extract expression values for the gene
    expression_values = expr_matrix.loc[gene]

    # Assign colors: above threshold -> 1, below or equal -> 0
    colors = np.where(expression_values > threshold, 1, 0)

    plt.imshow(he_img)
    plt.scatter(
        metadata["ROICoordinateY_downsampled"], 
        metadata["ROICoordinateX_downsampled"], 
        c=colors, cmap="viridis", alpha=0.8
    )
    plt.colorbar(label="Above (1) vs Below (0) Threshold")
    plt.title(f"Spatial Expression of {gene}(Threshold: {threshold})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

def display_spatial_expression(gene, he_img, subset_metadata, expr_matrix , xlim=None, ylim=None):
    plt.imshow(he_img)
    plt.scatter(
        subset_metadata["ROICoordinateY_downsampled"], 
        subset_metadata["ROICoordinateX_downsampled"], 
        c=expr_matrix.loc[gene], cmap="viridis", alpha=0.8
    )
    plt.colorbar(label="Expression Level")
    plt.title(f"Spatial Expression of {gene}")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    if xlim: plt.xlim(xlim)
    if ylim: plt.ylim(ylim)
    plt.show()

def single_violin(expression_df,gene):
    # Create a violin plot of expression values
    plt.figure(figsize=(8, 6))
    sns.violinplot(y=expression_df["Expression"], color="skyblue", inner="box")
    # Calculate statistics
    mean_expression = expression_df["Expression"].mean()
    median_expression = expression_df["Expression"].median()


    # Add mean and median lines
    plt.axhline(mean_expression, color="red",  linewidth=1)
    plt.axhline(median_expression, color="green",  linewidth=1)
    plt.text(0, mean_expression, f"Mean: {mean_expression:.2f}", color="red", fontsize=8)
    plt.text(0, median_expression, f"Median: {median_expression:.2f}", color="green", fontsize=8)

    # Customize the plot
    plt.title(f"Distribution of Expression Values for {gene}")
    plt.ylabel("Expression Value")
    plt.xlabel("All Samples (ROIs)")
    plt.show()


def ometiff_to_rgb(img):
    # Load your IF image (e.g., 4-channel image with shape (4, height, width))

    # Initialize an empty RGB image
    rgb_image = np.zeros((img.shape[1], img.shape[2], 3), dtype=np.float32)

    # Map each channel to RGB components (example: Red, Green, Blue, and Yellow)
    rgb_image[:, :, 0] = exposure.rescale_intensity(img[0], in_range='image', out_range=(0, 1))  # Red
    rgb_image[:, :, 1] = exposure.rescale_intensity(img[1], in_range='image', out_range=(0, 1))  # Green
    rgb_image[:, :, 2] = exposure.rescale_intensity(img[2], in_range='image', out_range=(0, 1))  # Blue

    # If you have a fourth channel and want to add it as yellow (mixing red and green)
    rgb_image[:, :, 0] += exposure.rescale_intensity(img[3], in_range='image', out_range=(0, 0.5))
    rgb_image[:, :, 1] += exposure.rescale_intensity(img[3], in_range='image', out_range=(0, 0.5))

    # Clip values to ensure they stay within the 0-1 range
    rgb_image = np.clip(rgb_image, 0, 1)
    gamma = 0.45  # Choose a value less than 1 to make the image brighter
    brightened_channels = np.stack([exposure.adjust_gamma(channel, gamma) for channel in rgb_image.transpose(2, 0, 1)], axis=2)

    return brightened_channels

