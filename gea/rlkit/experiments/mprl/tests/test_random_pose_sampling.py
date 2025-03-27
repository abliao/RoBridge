import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)


def randrange(n, vmin, vmax):
    """
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    """
    return (vmax - vmin) * np.random.rand(n) + vmin


fig = plt.figure()
ax = fig.add_subplot(projection="3d")

n = 10000


samples = np.random.normal(0, 1, (n, 3))
samples[:, 2] = np.abs(samples[:, 2])
samples /= np.linalg.norm(samples, axis=1).reshape(-1, 1)
xs, ys, zs = samples[:, 0], samples[:, 1], samples[:, 2]
ax.scatter(xs, ys, zs)

ax.set_xlabel("X Label")
ax.set_ylabel("Y Label")
ax.set_zlabel("Z Label")

plt.savefig("test.png")
