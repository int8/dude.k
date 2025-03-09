import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from tqdm import tqdm


def linear_interpolate_row(row):

    not_nan_indices = np.where(~np.isnan(row))[0]
    if len(not_nan_indices) == 0:
        return row
    nan_indices = np.where(np.isnan(row))[0]
    valid_indices = not_nan_indices
    valid_values = row[valid_indices]

    interp_func = interp1d(
        valid_indices,
        valid_values,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )

    row[nan_indices] = interp_func(nan_indices)

    return row


def forward_fill_zeros(arr):
    arr_filled = arr.copy()
    # Iterate over each column
    for col in range(arr.shape[1]):
        data = arr[:, col]
        # Create a mask of non-zero values
        mask = data != 0
        # Get indices of non-zero values
        idx = np.where(mask, np.arange(len(data)), 0)
        # Forward-fill indices
        idx_ffill = np.maximum.accumulate(idx)
        # Handle the case where the first element is zero
        # We only want to index valid positions
        valid = (idx_ffill > 0) | mask[0]
        # Replace zeros with the last non-zero value
        data_filled = data[idx_ffill]
        # In positions where no previous non-zero value exists, keep zeros
        data_filled[~valid] = 0
        arr_filled[:, col] = data_filled
    return arr_filled


def soft_non_maximum_suppression(scores, class_window=1, threshold=0.01):
    """
    Applies soft non-maximum suppression to a numpy matrix of scores.

    Parameters:
    - scores: numpy array of shape (num_frames, num_classes)
              Contains the prediction scores for each frame and class.
    - class_window: int or list of ints
              The suppression window size. If an integer is provided, the same window size is used for all classes.
              If a list is provided, it specifies window sizes for each class.
    - threshold: float
              Minimum score threshold. Detections with scores below this threshold are ignored.

    Returns:
    - suppressed_scores: numpy array of shape (num_frames, num_classes)
              The scores after applying soft non-maximum suppression.
    """

    num_frames, num_classes = scores.shape
    suppressed_scores = np.zeros_like(scores)

    # If class_window is a single integer, convert it to a list for consistency
    if isinstance(class_window, int):
        class_window = [class_window] * num_classes

    for c in tqdm(range(num_classes), desc="Applying soft NMS"):
        window = class_window[c]
        s = scores[:, c].copy()
        frames = np.arange(num_frames)
        processed = np.zeros(num_frames, dtype=bool)
        output_s = np.zeros(num_frames)

        while True:
            # Mask out already processed frames by setting their scores to -inf
            s_masked = s.copy()
            s_masked[processed] = -np.inf

            # Find the frame with the highest score that hasn't been processed
            e1_idx = np.argmax(s_masked)
            e1_score = s_masked[e1_idx]

            # If the highest score is below the threshold, stop processing
            if e1_score < threshold or e1_score == -np.inf:
                break

            # Add the highest scoring event to the output
            output_s[e1_idx] = e1_score
            processed[e1_idx] = True  # Mark this frame as processed

            # Calculate distances from the current frame to all other frames
            distances = np.abs(frames - e1_idx)

            # Identify frames within the suppression window that haven't been processed
            within_window = (distances <= window) & (~processed)

            # Calculate suppression factors for these frames
            suppression_factor = (distances[within_window] ** 2) / (window**2)

            # Suppress the scores of the neighboring frames
            s[within_window] *= suppression_factor

        # Update the suppressed scores for this class
        suppressed_scores[:, c] = output_s

    return suppressed_scores


def preserve_peaks_in_predictions(
    predictions: np.ndarray,
    gauss_sigma: float = 3.0,
    height: float = 0.01,
    distance: int = 15,
) -> np.ndarray:
    """
    Applies Gaussian smoothing and detects peaks in multi-class predictions,
    returning a predictions matrix with only the peaks preserved.

    Parameters:
    - predictions (np.ndarray): 2D array of shape (clip_length, nr_of_classes)
                                containing the predictions.
    - gauss_sigma (float): Standard deviation for the Gaussian kernel.
    - height (float): Required height of peaks.
    - distance (int): Required minimal horizontal distance (in frames) between neighboring peaks.

    Returns:
    - peaks_only_predictions (np.ndarray): 2D array of the same shape as `predictions`,
                                           with only the peaks preserved.
    """

    # Ensure predictions is a 2D array
    if predictions.ndim == 1:
        predictions = predictions.reshape(-1, 1)

    clip_length, nr_of_classes = predictions.shape

    # Initialize the output predictions matrix with zeros
    peaks_only_predictions = np.zeros_like(predictions)

    # Iterate over each class
    for cls_idx in range(nr_of_classes):
        # Extract the predictions for the current class
        class_predictions = predictions[:, cls_idx]

        # Apply Gaussian filter to smooth the predictions
        smoothed_predictions = gaussian_filter1d(class_predictions, sigma=gauss_sigma)

        # Detect peaks in the smoothed predictions
        peaks, _ = find_peaks(smoothed_predictions, height=height, distance=distance)

        # Create an array to hold peaks with zeros elsewhere
        peaks_only_sequence = np.zeros_like(class_predictions)

        # Preserve the original predictions at the peak positions
        peaks_only_sequence[peaks] = class_predictions[peaks]

        # Assign the peaks-only sequence back to the corresponding column
        peaks_only_predictions[:, cls_idx] = peaks_only_sequence

    return peaks_only_predictions
