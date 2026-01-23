import matplotlib.pyplot as plt
import numpy as np


def f(x):
    return 2 * x**2 + 44 * np.cos(x)


# Task a
x = np.linspace(-5, 10, 20)
y = f(x)

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(x, y, color="black", label="20 samples")

# Task b
x_dense = np.linspace(-5, 100, 200)
y_dense = f(x_dense)
ax.plot(x_dense, y_dense, color="red", linewidth=2, label="200-point curve")

ax.set_title("f(x) samples and interpolated curve")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.legend()

# Remove spines
ax.spines["top"].set_color("none")
ax.spines["right"].set_color("none")
ax.spines["left"].set_color("none")
ax.spines["bottom"].set_color("none")

plt.show()
