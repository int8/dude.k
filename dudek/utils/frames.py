import os
from copy import copy
from typing import Set

import cv2
import numpy as np

from kornia.geometry.transform import get_rotation_matrix2d, warp_affine
import kornia
import torch


def __resize(frame: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    if frame.shape[0] != target_height or frame.shape[1] != target_width:
        frame = cv2.resize(frame, (target_width, target_height))
    return frame


def save_frame(
    frame_index: int,
    frame: np.ndarray,
    path_out: str,
    target_height: int,
    target_width: int,
):
    frame = __resize(frame, target_height, target_width)
    saved = cv2.imwrite(os.path.join(path_out, f"{frame_index}.jpg"), frame)

    if not saved:
        raise Exception(f"Failed to save frame {frame_index}")


def __get_frame_numbers_around_center(
    annotation_frame_nr: int, radius_in_sec: int, fps: float = 25, stride: int = 2
) -> Set[int]:

    radius_in_frames = int(radius_in_sec * fps)

    left = annotation_frame_nr - radius_in_frames
    right = annotation_frame_nr + radius_in_frames

    return set(
        [frame_nr for frame_nr in range(left, right + 1) if frame_nr % stride == 0]
    )


def get_frame_numbers_around_centers(
    centers: Set[int], fps: float, stride: int, radius_in_sec: int
) -> Set[int]:
    frame_numbers_around_centers = copy(centers)
    for center in centers:
        frame_numbers_around_center = __get_frame_numbers_around_center(
            center,
            radius_in_sec=radius_in_sec,
            fps=fps,
            stride=stride,
        )
        frame_numbers_around_centers |= frame_numbers_around_center
    return frame_numbers_around_centers


def augment_with_camera_movement(frames_tensor, max_rotation=1.2, max_translation=15.0):
    """
    Applies small camera movement augmentations to the input frames tensor.

    Args:
        frames_tensor (torch.Tensor): Input frames tensor of shape (N, C, H, W),
                                      where N is the number of frames.
        max_rotation (float): Maximum rotation angle in degrees for augmentation.
        max_translation (float): Maximum translation in pixels for augmentation.

    Returns:
        torch.Tensor: Augmented frames tensor of the same shape as input.
    """
    N, C, H, W = frames_tensor.shape

    # Normalize pixel values if necessary (assuming input is in [0, 255])
    if frames_tensor.max() > 1.0:
        frames_tensor = frames_tensor / 255.0

    # Generate frame indices
    frame_numbers = torch.arange(N, dtype=torch.float32, device=frames_tensor.device)

    # Define the movements (e.g., sinusoidal)
    rotations = max_rotation * torch.sin(
        2 * torch.pi * frame_numbers / N
    )  # Shape: (N,)
    translations_x = max_translation * torch.sin(
        2 * torch.pi * frame_numbers / N
    )  # Shape: (N,)
    translations_y = max_translation * torch.cos(
        2 * torch.pi * frame_numbers / N
    )  # Shape: (N,)

    # Center coordinates for rotation (assuming rotation around the center of the image)
    center = (
        torch.tensor([W / 2, H / 2], device=frames_tensor.device)
        .unsqueeze(0)
        .repeat(N, 1)
    )  # Shape: (N, 2)

    # Define scales as a tensor of shape (N, 2)
    scales = torch.ones(N, 2, device=frames_tensor.device)  # Shape: (N, 2)

    # Get rotation matrices (vectorized over all frames)
    rotation_matrices = get_rotation_matrix2d(
        center=center, angle=rotations, scale=scales
    )  # Shape: (N, 2, 3)

    # Create translations tensor and adjust the rotation matrices
    translations = torch.stack((translations_x, translations_y), dim=1)  # Shape: (N, 2)
    rotation_matrices[:, :, 2] += translations

    # Apply the affine transformations to the frames using warp_affine
    augmented_frames = warp_affine(
        frames_tensor,
        rotation_matrices,
        dsize=(H, W),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )

    augmented_frames = augmented_frames.clamp(0.0, 1.0)

    uint8_augmented_frames = (augmented_frames * 255).type(torch.uint8)
    return uint8_augmented_frames


def apply_camera_pitch(video_tensor, angle_in_degrees):
    """
    Applies a camera pitch transformation to a video tensor.

    Parameters:
        video_tensor (torch.Tensor): Input video tensor of shape [B, C, H, W] and dtype torch.uint8.
        angle_in_degrees (float): The pitch angle in degrees. Positive values tilt upwards, negative downwards.

    Returns:
        torch.Tensor: The transformed video tensor with the same shape and dtype as the input.
    """
    # Check that video_tensor is a uint8 tensor
    if video_tensor.dtype != torch.uint8:
        raise ValueError("video_tensor must be of dtype torch.uint8")

    # Get the device of the input tensor
    device = video_tensor.device

    # Convert the video tensor to float and normalize to [0, 1]
    video_tensor_float = video_tensor.float() / 255.0

    # Get the dimensions of the video frames
    batch_size, channels, height, width = video_tensor_float.shape

    # Define the intrinsic camera matrix K
    focal_length = 1.0  # You can adjust this value if needed
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    K = torch.tensor(
        [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]],
        dtype=torch.float32,
        device=device,
    )

    K_inv = torch.inverse(K)

    # Expand K and K_inv to match the batch size
    K = K.unsqueeze(0).expand(batch_size, -1, -1)
    K_inv = K_inv.unsqueeze(0).expand(batch_size, -1, -1)

    # Convert angle to radians
    theta = torch.tensor(angle_in_degrees * torch.pi / 180.0, device=device)

    # Compute the rotation matrix around the X-axis for pitch
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    R = torch.tensor(
        [
            [1, 0, 0],
            [0, cos_theta.item(), -sin_theta.item()],
            [0, sin_theta.item(), cos_theta.item()],
        ],
        dtype=torch.float32,
        device=device,
    )

    # Expand R to match the batch size
    R = R.unsqueeze(0).expand(batch_size, -1, -1)

    # Compute the homography matrix H = K * R * K^-1
    H = K @ R @ K_inv

    # Apply the perspective warp using the homography matrix
    video_tensor_warped = kornia.geometry.transform.warp_perspective(
        video_tensor_float, H, dsize=(height, width), align_corners=True
    )

    # Convert the warped video tensor back to uint8
    video_tensor_warped_uint8 = (
        (video_tensor_warped * 255.0).clamp(0, 255).to(torch.uint8)
    )

    return video_tensor_warped_uint8


def crop_video(
    frames_tensor: torch.Tensor,
    crop_size_h: int,
    crop_size_w: int,
) -> torch.Tensor:
    """Applies a random crop to the input video frames and resizes back to the original size.

    Args:
        frames_tensor (torch.Tensor): Input frames tensor of shape (N, C, H, W).
            Can be of type torch.uint8 (values in [0, 255]) or torch.float32/torch.float64 (values in [0, 1]).
        crop_size_h (int): The height of the crop.
        crop_size_w (int): The width of the crop.


    Returns:
        torch.Tensor: Augmented frames tensor of the same shape and dtype as input.
    """
    N, C, H, W = frames_tensor.shape

    if crop_size_h >= H or crop_size_w >= W:
        raise ValueError("Crop size must be smaller than the original size.")

    device = frames_tensor.device
    orig_dtype = frames_tensor.dtype  # Save original data type

    # Handle different input dtypes
    if frames_tensor.dtype == torch.uint8:
        # Convert to float and normalize to [0, 1]
        frames_tensor = frames_tensor.float() / 255.0
    elif frames_tensor.dtype == torch.float32 or frames_tensor.dtype == torch.float64:
        # Assume values are already in [0, 1]
        pass
    else:
        raise ValueError(f"Unsupported tensor dtype: {frames_tensor.dtype}")

    # Randomly select the top-left corner of the crop
    x_start = torch.randint(0, W - crop_size_w + 1, (1,), device=device).item()
    y_start = torch.randint(0, H - crop_size_h + 1, (1,), device=device).item()
    x_end = x_start + crop_size_w - 1
    y_end = y_start + crop_size_h - 1

    # Create source boxes (bounding boxes for cropping) of shape (N, 4, 2)
    boxes = torch.tensor(
        [
            [
                [x_start, y_start],  # top-left
                [x_end, y_start],  # top-right
                [x_end, y_end],  # bottom-right
                [x_start, y_end],
            ]  # bottom-left
        ],
        dtype=torch.float32,
        device=device,
    )  # Shape: (1, 4, 2)
    boxes = boxes.expand(N, -1, -1)  # Expand to (N, 4, 2)

    # Define destination boxes (we crop and resize to the full image dimensions)
    dst_boxes = torch.tensor(
        [
            [
                [0, 0],  # top-left
                [W - 1, 0],  # top-right
                [W - 1, H - 1],  # bottom-right
                [0, H - 1],
            ]  # bottom-left
        ],
        dtype=torch.float32,
        device=device,
    )  # Shape: (1, 4, 2)
    dst_boxes = dst_boxes.expand(N, -1, -1)  # Expand to (N, 4, 2)

    # Apply crop and resize to each frame
    cropped_frames = kornia.geometry.transform.crop_by_boxes(
        frames_tensor,
        src_box=boxes,
        dst_box=dst_boxes,
        mode="bilinear",
        align_corners=False,
    )

    # Clamp values to [0, 1] after interpolation
    cropped_frames = cropped_frames.clamp(0.0, 1.0)

    # Convert back to original dtype and range if necessary
    if orig_dtype == torch.uint8:
        # Rescale to [0, 255] and convert to uint8
        cropped_frames = (cropped_frames * 255.0).round().type(torch.uint8)

    return cropped_frames
