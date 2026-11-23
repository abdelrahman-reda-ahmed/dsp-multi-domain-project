#!/usr/bin/env python3
"""
Student 5 - 2D Fourier Transform & Image Filtering
=====================================================

Required tasks:
1. Load an image and convert it to grayscale.
2. Compute the 2D Fourier Transform using fft2.
3. Center the zero-frequency component using fftshift.
4. Create ideal circular Low-Pass and High-Pass masks.
5. Apply the masks in the frequency domain.
6. Reconstruct the filtered images using the inverse FFT.
7. Display the original image, spectrum, and filtered results.

Extra features:
- Automatic image discovery when running this file independently.
- Synthetic image fallback if no image is available.
- Visualize Low-Pass and High-Pass masks.
- Experiment with different cutoff radii.
- Calculate frequency-retention percentages.
- Save images and figures in outputs/student5/.
"""

# 1. imports
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 2. image loading
def load_grayscale_image(image_path=None):
    """
    Load an image and convert it to grayscale.
    Parameters
    ----------
    image_path : str or Path
        Path to the input image.
    Returns
    -------
    gray : numpy.ndarray
        Two-dimensional grayscale image.
    """
    if image_path is None:
        raise ValueError("An image path must be provided.")
    # Read the image from disk.
    # OpenCV reads color images in BGR format.
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    # Check whether the image was loaded successfully.
    if image is None:
        raise FileNotFoundError(
            f"Unable to load image: {image_path}"
        )
    # Convert the color image from BGR to grayscale.
    # Grayscale represents image intensity using one channel.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


# 3. 2D FOURIER TRANSFORM
def compute_fft_shifted(image):
    """
    Compute the 2D FFT and center the zero-frequency component.
    Parameters
    ----------
    image : numpy.ndarray
        Grayscale image.
    Returns
    -------
    shifted : numpy.ndarray
        Centered complex Fourier spectrum.
    """
    # Transform the image from the spatial domain into the frequency domain.
    # The output is complex-valued and contains frequency magnitude and phase information.
    fft_result = np.fft.fft2(image)
    # Move the zero-frequency component from the corners to the center of the Fourier spectrum.
    # This makes frequency filtering and visualization easier.
    shifted = np.fft.fftshift(fft_result)
    return shifted


# 4. CIRCULAR LOW-PASS / HIGH-PASS MASKS
def create_circular_mask(shape, radius, kind="lowpass"):
    """
    Create an ideal circular frequency-domain mask.
    Parameters
    ----------
    shape : tuple
        Image shape (height, width).
    radius : float
        Cutoff radius measured in frequency-array pixels.
    kind : str
        'lowpass' or 'highpass'.
    Returns
    -------
    mask : numpy.ndarray
        Binary mask containing zeros and ones.
    """

    # Get image dimensions.
    height, width = shape[:2]
    # Find the center of the shifted frequency spectrum.
    center_y = height // 2
    center_x = width // 2
    # Generate coordinate grids for all frequency positions.
    Y, X = np.ogrid[:height, :width]
    # Calculate the Euclidean distance from each frequency position to the center.
    # The center represents the zero frequency (DC).
    distance = np.sqrt(
        (Y - center_y) ** 2 +
        (X - center_x) ** 2
    )
    # Create the Low-Pass mask.
    # Frequencies inside the cutoff circle = 1 (kept).
    # Frequencies outside the cutoff circle = 0 (removed).
    lowpass_mask = (distance <= radius).astype(np.float32)
    # Return the Low-Pass mask if requested.
    if kind.lower() == "lowpass":
        return lowpass_mask
    # Create the High-Pass mask if requested.
    # High-Pass is the complement of Low-Pass: Frequencies inside the circle = 0 (removed) - Frequencies outside the circle = 1 (kept).
    elif kind.lower() == "highpass":
        highpass_mask = 1.0 - lowpass_mask
        return highpass_mask
    # Reject unknown filter names.
    else:
        raise ValueError(
            "kind must be either 'lowpass' or 'highpass'."
        )

# 5. APPLY MASK AND RECONSTRUCT THE IMAGE
def apply_frequency_mask(shifted, mask):
    """
    Apply a frequency mask and reconstruct the filtered image.
    Parameters
    ----------
    shifted : numpy.ndarray
        Centered complex Fourier Transform.
    mask : numpy.ndarray
        Binary frequency mask.
    Returns
    -------
    filtered_image : numpy.ndarray
        Reconstructed real-valued image.
    """
    # Keep or remove frequencies by multiplying the spectrum element-by-element with the mask.
    # Mask value 1 -> frequency retained.
    # Mask value 0 -> frequency removed.
    filtered_spectrum = shifted * mask
    # Return the zero-frequency component to its original corner position before applying inverse FFT.
    unshifted_spectrum = np.fft.ifftshift(filtered_spectrum)
    # Convert the filtered spectrum back into the spatial domain.
    reconstructed = np.fft.ifft2(unshifted_spectrum)
    # The input image is real-valued.
    # The imaginary part should be approximately zero; tiny imaginary values may occur due to numerical errors.
    filtered_image = np.real(reconstructed)
    return filtered_image

# 6. DISPLAY ORIGINAL, SPECTRUM, LPF AND HPF
def plot_results(original, shifted, lowpass, highpass, show=True):
    """
    Display four results:
    1. Original image.
    2. Fourier magnitude spectrum.
    3. Low-Pass filtered image.
    4. High-Pass filtered image.
    """
    # Calculate Fourier magnitude.
    magnitude = np.abs(shifted)
    # Log scaling makes weaker frequencies visible.
    # Without it, strong frequency values may dominate the display.
    magnitude_spectrum = np.log1p(magnitude)
    # Create a 2x2 grid of plots.
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    # -------------------- Original Image --------------------
    axes[0, 0].imshow(original, cmap="gray")
    axes[0, 0].set_title("Original Grayscale Image")
    # -------------------- Fourier Spectrum --------------------
    axes[0, 1].imshow(magnitude_spectrum, cmap="gray")
    axes[0, 1].set_title("2D Fourier Magnitude Spectrum")
    # -------------------- Low-Pass Result --------------------
    axes[1, 0].imshow(lowpass, cmap="gray")
    axes[1, 0].set_title("Low-Pass Filtered Image")
    # -------------------- High-Pass Result --------------------
    axes[1, 1].imshow(highpass, cmap="gray")
    axes[1, 1].set_title("High-Pass Filtered Image")
    # Hide axes to make the images easier to compare.
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("2D Fourier Transform and Ideal Image Filtering", fontsize=16)
    fig.tight_layout()
    if show:
        plt.show()
    return fig
    
# 7. DISPLAY LOW-PASS AND HIGH-PASS MASKS
def plot_masks(low_mask, high_mask, radius, show=True):
    """
    Display the frequency masks.
    White (1) = retained frequencies.
    Black (0) = removed frequencies.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # Low-Pass mask.
    axes[0].imshow( low_mask, cmap="gray", vmin=0, vmax=1 )
    axes[0].set_title( f"Low-Pass Mask | Radius = {radius}" )
    # High-Pass mask.
    axes[1].imshow( high_mask, cmap="gray", vmin=0, vmax=1 )
    axes[1].set_title( f"High-Pass Mask | Radius = {radius}" )
    for ax in axes:
        ax.axis("off")
    fig.suptitle("Ideal Circular Frequency Masks")
    fig.tight_layout()
    if show:
        plt.show()
    return fig

# 8. CUTOFF RADIUS EXPERIMENT
def plot_radius_experiment(image, shifted, radii=(15, 40, 80), show=True):
    """
    Compare Low-Pass and High-Pass filters using multiple cutoff radii.
    Smaller radius: LPF removes more details and produces more blur.
    Larger radius: LPF keeps more details.
    For HPF, increasing the radius removes more low and medium frequencies.
    """
    # Create a subplot for each tested radius.
    # First row = Low-Pass results.
    # Second row = High-Pass results.
    fig, axes = plt.subplots( 2, len(radii), figsize=(4 * len(radii), 8) )
    for column, radius in enumerate(radii):
        # Create Low-Pass mask for the current radius.
        low_mask = create_circular_mask( image.shape, radius, kind="lowpass" )
        # Create High-Pass mask for the current radius.
        high_mask = create_circular_mask( image.shape, radius, kind="highpass" )
        # Apply Low-Pass filter.
        low_result = apply_frequency_mask( shifted, low_mask )
        # Apply High-Pass filter.
        high_result = apply_frequency_mask( shifted, high_mask )
        # Display Low-Pass result.
        axes[0, column].imshow( low_result, cmap="gray" )
        axes[0, column].set_title( f"LPF | Radius = {radius}" )
        # Display High-Pass result.
        axes[1, column].imshow( high_result, cmap="gray" )
        axes[1, column].set_title( f"HPF | Radius = {radius}" )
    # Hide axes from all subplots.
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle( "Effect of Cutoff Radius on Frequency Filtering", fontsize=15 )
    fig.tight_layout()
    if show:
        plt.show()
    return fig

# 9. SYNTHETIC DEMO IMAGE
def create_demo_image():
    """
    Create a synthetic grayscale image.
    This is used only when the input image is unavailable, allowing the project to run without external data.
    """
    # Create a 256 x 256 black image.
    image = np.zeros( (256, 256), dtype=np.uint8 )
    # Draw a gray rectangle.
    cv2.rectangle( image, (40, 40), (210, 210), 180, -1 )
    # Draw a white circle.
    cv2.circle( image, (128, 128), 60, 255, -1 )
    # Draw a diagonal line.
    cv2.line( image, (20, 230), (230, 20), 255, 3 )
    return image

# 10. SAVE OUTPUT IMAGES AND FIGURES
def save_results( original, lowpass, highpass, low_mask, high_mask, main_figure, mask_figure, radius_figure, output_dir ):
    """
    Save the output images and figures to disk.
    High-Pass values can be negative or positive.
    We normalize a copy for saving and visualization.
    """
    # Create the output directory if it does not exist.
    output_dir = Path(output_dir)
    output_dir.mkdir( parents=True, exist_ok=True )
    # Save original grayscale image.
    cv2.imwrite( str(output_dir / "original_grayscale.png"), original )
    # Save Low-Pass filtered image.
    # Clip to the valid image intensity range.
    lowpass_uint8 = np.clip( lowpass, 0, 255 ).astype(np.uint8)
    cv2.imwrite( str(output_dir / "lowpass_result.png"), lowpass_uint8 )
    # Normalize the High-Pass result for visualization.
    # High-Pass filtering may produce negative values,
    # so directly clipping would lose information.
    highpass_normalized = cv2.normalize( highpass, None, 0, 255, cv2.NORM_MINMAX )
    highpass_uint8 = highpass_normalized.astype(np.uint8)
    cv2.imwrite( str(output_dir / "highpass_result.png"), highpass_uint8 )
    # Save masks as black-and-white images.
    cv2.imwrite( str(output_dir / "lowpass_mask.png"), (low_mask * 255).astype(np.uint8) )
    cv2.imwrite( str(output_dir / "highpass_mask.png"), (high_mask * 255).astype(np.uint8) )
    # Save the main comparison figure.
    main_figure.savefig( output_dir / "main_comparison.png", dpi=300, bbox_inches="tight" )
    # Save mask visualization.
    mask_figure.savefig( output_dir / "frequency_masks.png", dpi=300, bbox_inches="tight" )
    # Save cutoff radius experiment.
    radius_figure.savefig( output_dir / "radius_experiment.png", dpi=300, bbox_inches="tight" )
    print(f"\nAll outputs saved in: {output_dir}")


# 11. FIND AN IMAGE WHEN RUNNING THIS FILE DIRECTLY
def find_project_image(project_root):
    """
    Find the first supported image inside data/images/.
    This function is useful when student5.py is executed directly without an image_path argument.
    """
    image_folder = project_root / "data" / "images"
    supported_extensions = ( "*.png", "*.jpg", "*.jpeg", "*.bmp" )
    candidates = []
    # Search for each supported image extension.
    for extension in supported_extensions:
        candidates.extend(image_folder.glob(extension))
    # Sort to ensure predictable selection if multiple images exist.
    candidates = sorted(candidates)
    if candidates:
        return candidates[0]
    return None

# 12. MAIN PIPELINE
def main(image_path=None, show=True):
    """
    Execute the complete Student 5 pipeline.
    Compatible with the team's main.py: main(image_path=image_path)
    Parameters
    ----------
    image_path : str or Path or None
        Input image path provided by the master runner.
    show : bool
        Whether to display plots interactively.
    """
    print("\n" + "=" * 65)
    print("STUDENT 5: 2D FOURIER & IMAGE FILTERING")
    print("=" * 65)
    # STEP 1: FIND PROJECT ROOT
    # parents[2] points to the project root.
    project_root = Path(__file__).resolve().parents[2]
    # STEP 2: FIND IMAGE IF NO PATH WAS PROVIDED
    # When main.py calls this function, it normally provides image_path automatically.
    # When running student5.py directly, image_path is None, so we search the project's data/images folder ourselves.
    if image_path is None:
        image_path = find_project_image(project_root)
    # STEP 3: LOAD IMAGE OR USE DEMO IMAGE
    if image_path is not None and Path(image_path).is_file():
        try:
            original = load_grayscale_image(image_path)
            print(f"Image loaded successfully: {image_path}")
        except (FileNotFoundError, ValueError) as error:
            print(f"Image loading error: {error}")
            print("Using synthetic demo image instead.")
            original = create_demo_image()
    else:
        print("No image found in data/images/.")
        print("Using synthetic demo image.")
        original = create_demo_image()
    # STEP 4: DISPLAY IMAGE INFORMATION
    print("\nImage information:")
    print(f"Shape      : {original.shape}")
    print(f"Data type  : {original.dtype}")
    print(f"Height     : {original.shape[0]} pixels")
    print(f"Width      : {original.shape[1]} pixels")
    print(f"Min value : {original.min()}")
    print(f"Max value : {original.max()}")
    # STEP 5: COMPUTE 2D FFT
    shifted = compute_fft_shifted(original)
    print("\n2D FFT calculated.")
    print("Zero-frequency component centered using fftshift.")
    # STEP 6: DEFINE CUTOFF RADIUS
    # This is the main radius used for the LPF and HPF results.
    # A radius of 40 is a reasonable starting experiment for a 512 x 512 image, but the effect depends on the image dimensions.
    cutoff_radius = 40
    print(f"Cutoff radius = {cutoff_radius}")
    # STEP 7: CREATE FILTER MASKS
    low_mask = create_circular_mask( original.shape, cutoff_radius, kind="lowpass" )
    high_mask = create_circular_mask( original.shape, cutoff_radius, kind="highpass" )
    print("Low-Pass and High-Pass masks created.")
    # STEP 8: CALCULATE FREQUENCY RETENTION
    total_frequencies = low_mask.size
    # Count the frequency locations retained by each filter.
    retained_low = np.sum(low_mask)
    retained_high = np.sum(high_mask)
    # Convert counts to percentages.
    low_percentage = ( retained_low / total_frequencies ) * 100
    high_percentage = ( retained_high / total_frequencies ) * 100
    print("\nFrequency retention:")
    print(f"Low-Pass retained : {low_percentage:.2f}%")
    print(f"High-Pass retained : {high_percentage:.2f}%")
    print( f"Total retained     : " f"{low_percentage + high_percentage:.2f}%" )
    # STEP 9: APPLY LOW-PASS FILTER
    lowpass = apply_frequency_mask( shifted, low_mask )
    print("Low-Pass filtering completed.")
    # STEP 10: APPLY HIGH-PASS FILTER
    highpass = apply_frequency_mask( shifted, high_mask )
    print("High-Pass filtering completed.")
    # STEP 11: PLOT MAIN RESULTS
    main_figure = plot_results( original, shifted, lowpass, highpass, show=show )
    # STEP 12: PLOT FREQUENCY MASKS
    mask_figure = plot_masks( low_mask, high_mask, cutoff_radius, show=show )
    # STEP 13: EXPERIMENT WITH DIFFERENT RADII
    test_radii = (15, 40, 80)
    radius_figure = plot_radius_experiment( original, shifted, radii=test_radii, show=show )
    print(f"\nCutoff radius experiment: {test_radii}")
    # STEP 14: SAVE OUTPUTS
    output_dir = project_root / "outputs" / "student5"
    save_results( original, lowpass, highpass, low_mask, high_mask, main_figure, mask_figure, radius_figure, output_dir )
    return {
        "original": original,
        "fft_shifted": shifted,
        "lowpass": lowpass,
        "highpass": highpass,
        "lowpass_mask": low_mask,
        "highpass_mask": high_mask,
        "cutoff_radius": cutoff_radius,
        "output_dir": output_dir,
    }
if __name__ == "__main__":
    main()
