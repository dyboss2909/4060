import cv2
import numpy as np
import matplotlib.pyplot as plt


def match_dimensions(foreground, background, mode="resize_fg"):
    """Ensure foreground/background have same HxW, via resize or crop."""
    fg_h, fg_w = foreground.shape[:2]
    bg_h, bg_w = background.shape[:2]

    if (fg_h, fg_w) == (bg_h, bg_w):
        return foreground, background

    if mode == "resize_fg":
        interp = cv2.INTER_AREA if fg_h > bg_h or fg_w > bg_w else cv2.INTER_LINEAR
        fg_resized = cv2.resize(foreground, (bg_w, bg_h), interpolation=interp)
        return fg_resized, background

    if mode == "crop_bg":
        if bg_h < fg_h or bg_w < fg_w:
            interp = cv2.INTER_AREA if fg_h > bg_h or fg_w > bg_w else cv2.INTER_LINEAR
            fg_resized = cv2.resize(foreground, (bg_w, bg_h), interpolation=interp)
            return fg_resized, background
        bg_cropped = background[:fg_h, :fg_w]
        return foreground, bg_cropped

    raise ValueError("mode must be 'resize_fg' or 'crop_bg'")


# File names in the same directory as this script
foreground_path = "foreground.png"
background_path = "background.jpg"

# Load foreground with alpha (BGRA) and background (BGR)
fg_bgra = cv2.imread(foreground_path, cv2.IMREAD_UNCHANGED)
bg_bgr = cv2.imread(background_path, cv2.IMREAD_COLOR)

if fg_bgra is None or bg_bgr is None:
    raise FileNotFoundError("Could not load images. Check file names and paths.")

# Match dimensions before color conversion
fg_bgra, bg_bgr = match_dimensions(fg_bgra, bg_bgr, mode="resize_fg")

# Convert to RGB/RGBA for visualization and blending
fg_rgba = cv2.cvtColor(fg_bgra, cv2.COLOR_BGRA2RGBA)
bg_rgb = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2RGB)

# Extract RGB and alpha channels, normalize alpha to [0, 1]
fg_rgb = fg_rgba[:, :, :3].astype(np.float32)
alpha = fg_rgba[:, :, 3].astype(np.float32) / 255.0

# Alpha blending per pixel and per channel
alpha_3 = alpha[:, :, None]
blended = alpha_3 * fg_rgb + (1.0 - alpha_3) * bg_rgb.astype(np.float32)

# Clip and convert to uint8
blended = np.clip(blended, 0, 255).astype(np.uint8)

# Display results
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].imshow(fg_rgb.astype(np.uint8))
axes[0, 0].set_title("Foreground (RGB)")
axes[0, 1].imshow(alpha, cmap="gray")
axes[0, 1].set_title("Alpha Mask")
axes[1, 0].imshow(bg_rgb)
axes[1, 0].set_title("Background (RGB)")
axes[1, 1].imshow(blended)
axes[1, 1].set_title("Blended Output")

for ax in axes.ravel():
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
