import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


img = mpimg.imread("boy.jpg")

# Convert to grayscale
gray_img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

fig, ax = plt.subplots(figsize=(8, 4))
ax.imshow(gray_img, cmap="gray")
ax.set_title("Grayscale")
ax.set_xticks([])
ax.set_yticks([])

ax.spines["top"].set_color("none")
ax.spines["right"].set_color("none")
ax.spines["left"].set_color("none")
ax.spines["bottom"].set_color("none")


plt.show()
