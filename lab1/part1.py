import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(-5, 10, 400)
f = 2 * x**2 + 44 * np.cos(x)
g = 3 * x**2

fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.set_title("f(x) and g(x) on twin axes")
ax1.set_xlabel("x")
ax1.plot(x, f, color="red", linestyle="--", linewidth=2)
ax1.set_ylabel("f(x)")

#ax2 = ax1.twinx()
ax2.plot(x, g, color="blue", linewidth=2)
ax2.set_ylabel("g(x)")


plt.show()
