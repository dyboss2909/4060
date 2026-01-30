import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def match_dimensions(foreground, background, mode="resize_fg"):
    """Make sure both images have the same height/width (no hardcoding)."""
    fg_height, fg_width = foreground.shape[:2]
    bg_height, bg_width = background.shape[:2]

    if (fg_height, fg_width) == (bg_height, bg_width):
        return foreground, background

    if mode == "resize_fg":
        # Resize the foreground to the background size (interpolation concept)
        if fg_height > bg_height or fg_width > bg_width:
            interpolation = cv2.INTER_AREA  # better for shrinking
        else:
            interpolation = cv2.INTER_LINEAR  # better for enlarging
        resized_foreground = cv2.resize(
            foreground, (bg_width, bg_height), interpolation=interpolation
        )
        return resized_foreground, background

    if mode == "crop_bg":
        # Crop the background to the foreground size (if possible)
        if bg_height < fg_height or bg_width < fg_width:
            if fg_height > bg_height or fg_width > bg_width:
                interpolation = cv2.INTER_AREA
            else:
                interpolation = cv2.INTER_LINEAR
            resized_foreground = cv2.resize(
                foreground, (bg_width, bg_height), interpolation=interpolation
            )
            return resized_foreground, background
        cropped_background = background[:fg_height, :fg_width]
        return foreground, cropped_background

    raise ValueError("mode must be 'resize_fg' or 'crop_bg'")


# Images are in the same directory as this script
base_dir = os.path.dirname(__file__)
foreground_path = os.path.join(base_dir, "panther.png")
background_path = os.path.join(base_dir, "savana.jpg")

# 1) Load images (NumPy arrays of pixel intensities 0–255)
# Foreground keeps alpha (BGRA); background is regular BGR
foreground_bgra = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)
background_bgr = cv2.imread(background_path, cv2.IMREAD_COLOR)

if foreground_bgra is None or background_bgr is None:
    raise FileNotFoundError("Could not load images. Check file names and paths.")

# 2) Match image sizes before blending (must be identical HxW)
foreground_bgra, background_bgr = match_dimensions(
    foreground_bgra, background_bgr, mode="resize_fg"
)

# 3) Convert color space (OpenCV uses BGR by default)
foreground_rgba = cv2.cvtColor(foreground_bgra, cv2.COLOR_BGRA2RGBA)
background_rgb = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2RGB)

# 4) Split channels; alpha is a mask (not binary). Normalize to [0, 1].
foreground_rgb = foreground_rgba[:, :, :3].astype(np.float32)
alpha = foreground_rgba[:, :, 3].astype(np.float32) / 255.0

# 5) Alpha blending per pixel and per channel (elementwise arithmetic)
alpha_3 = alpha[:, :, None]  # shape: (H, W, 1) for broadcasting
blended = alpha_3 * foreground_rgb + (1.0 - alpha_3) * background_rgb.astype(np.float32)

# 6) Clip to [0, 255] and convert to uint8 for display
blended = np.clip(blended, 0, 255).astype(np.uint8)

# 7) Display results
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].imshow(foreground_rgb.astype(np.uint8))
axes[0, 0].set_title("Foreground (RGB)")
axes[0, 1].imshow(alpha, cmap="gray")
axes[0, 1].set_title("Alpha Mask")
axes[1, 0].imshow(background_rgb)
axes[1, 0].set_title("Background (RGB)")
axes[1, 1].imshow(blended)
axes[1, 1].set_title("Blended Output")

for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
