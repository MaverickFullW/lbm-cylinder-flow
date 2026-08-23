import numpy as np

D = 20
R = D / 2
nx = 30 * D
ny = 10 * D
xc = 5 * D
yc = ny / 2
y, x = np.ogrid[:ny, :nx]
cylinder = (x - xc)**2 + (y - yc)**2 <= R**2

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.imshow(cylinder, origin="lower")
    plt.title("Cylinder mask")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
