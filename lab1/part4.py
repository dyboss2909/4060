import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np



img = mpimg.imread("boy.jpg")

img_255 = img

# Posterize each channel to 4 possible values
posterized = (img_255 // 64) * 64

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].imshow(img)
axes[0].set_title("Original")
axes[0].set_xticks([])
axes[0].set_yticks([])

axes[1].imshow(posterized)
axes[1].set_title("Posterized")
axes[1].set_xticks([])
axes[1].set_yticks([])


plt.show()
