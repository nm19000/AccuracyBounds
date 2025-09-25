import numpy as np
import matplotlib.pyplot as plt

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


def plot_times_comput(n_iter_list, times_comput):
    plt.plot(n_iter_list, times_comput, color = 'red')
    plt.title('Times of computation')
    plt.xlabel('Number of samples')
    plt.ylabel('t in seconds')
    plt.grid()
    plt.show()
    print(f'The last kernel size computation took {times_comput[-1]:6f} seconds')

def plot_avkersize_conv(av_kersize, av_kersizes, ker_size, max_k):

    # Plot results
    x_axis = np.arange(2, max_k+1)
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