import numpy as np
import matplotlib.pyplot as plt



def visualize_ball_3d(points, 
                      ax=None, 
                      radius=1, 
                      center=(0,0,0),
                      title="3D Ball with Points Inside",
                      x_axis="X",
                      y_axis="Y",
                      z_axis="Z"):

    n_points, dim = points.shape
    assert dim == 3, f"Only 3D points accepted. Given shape: {points.shape}"

    if not ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=0.5, label='Points')

    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]
    x = radius * np.cos(u) * np.sin(v) + center[0]
    y = radius * np.sin(u) * np.sin(v) + center[1]
    z = radius * np.cos(v) + center[2]
    ax.plot_wireframe(x, y, z, color='r', alpha=0.2, linewidth=0.5)

    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])
    ax.set_zlim([-radius, radius])
    ax.set_title(title)
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    ax.set_zlabel(z_axis)


def plot_avkersize_conv(av_kersize, av_kersizes, ker_size, max_k):

    # Plot results
    x_axis = np.arange(1, max_k+1)
    y_axis = np.array(av_kersizes)
    plt.plot(x_axis, y_axis)
    plt.axhline(ker_size, color='r')
    plt.ylim(0, ker_size + 0.3 * ker_size)
    plt.grid()
    plt.xlabel("Number of samples")
    plt.ylabel("Approximate average kernel size")
    plt.title("Number of Samples vs Av. Kernel Size")
    plt.show()

    #max_diameter_total = max(max_diameters)
    print(f"Total Max Kernel Size: {av_kersize}")
    print(f"Analytical Kernel Size: {ker_size}")
    rel_error = (av_kersize - ker_size) / ker_size
    print(f"Relative Error: {rel_error}")

def plot_wckersize_conv(wc_kersizef, kersize_approxis, ker_size, max_k):

    # Plot results
    x_axis = np.arange(2, max_k+1)
    y_axis = np.array(kersize_approxis)
    plt.plot(x_axis, y_axis)
    plt.axhline(ker_size, color='r')
    plt.ylim(0, ker_size + 0.1 * ker_size)
    plt.grid()
    plt.xlabel("Number of samples")
    plt.ylabel("Approximate  wc kernel size")
    plt.title("Number of Samples vs wc Kernel Size")
    plt.show()

    #max_diameter_total = max(max_diameters)
    print(f"Total Max Kernel Size: { wc_kersizef}")
    print(f"Analytical Kernel Size: {ker_size}")
    rel_error = (wc_kersizef - ker_size) / ker_size
    print(f"Relative Error: {rel_error}")