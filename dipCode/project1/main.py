# pylint: disable=redefined-outer-name, unreachable, missing-function-docstring
"""
Image Processing Project 1: Images, Intensities, and Histograms
"""
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt


def build_histogram_bins(gray_img, bin_num=16):
    """Takes in a greyscale image (array) and returns a 2D array of histogram bins and bin counts, using numpy."""
    sorted_pixels = np.sort(gray_img.flatten())

    bin_values = np.linspace(0, 1, bin_num+1)
    bin_counts = np.zeros_like(bin_values)

    idx = 0
    for pixel in sorted_pixels:
        while  idx < bin_num - 1 and pixel >= bin_values[idx + 1]:
            idx += 1
        bin_counts[idx] += 1  # Increment the count for the appropriate bin

    # Ignore last value, max is counted in the bin before
    return bin_values, bin_counts[:-1]

def double_side_thresholding(gray_img, high, low):
    """Performs double-sided (high and low) thresholding on images to define regions, and outputs the thresholded
    image. Sets pixels above high threshold to 1, and pixels below threshold to 0, leaves all pixels in between the same
    """

    high_pixels = (gray_img > (high / 255))
    # mid_pixels = ((gray_img > (low / 255)) & (gray_img <= (high / 255)))
    low_pixels = (gray_img < (low / 255))
    out_img = np.copy(gray_img)

    out_img[high_pixels] = 1
    out_img[low_pixels] = 0

    return out_img

def connected_comps(gray_img, min_size):
    """Performs connected components and removes components with an area less than the indicated size."""
    # Utilize the label function to get connected regions/components
    labels, num = ski.measure.label(gray_img, background=0, return_num=True, connectivity=2)

    # Remove components with an area too small
    for component in ski.measure.regionprops(labels):
        if component.area < min_size:
            labels[labels == component.label] = 0

    # Need to re-label
    labels, num = ski.measure.label(labels, background=0, return_num=True, connectivity=2)
    return labels, num

def histogram_equalization(img):
    """Performs histogram equalization"""

def img_figure(image, title, axis):
    """Helper to add image to an axis"""
    # Specify the color map is gray, otherwise it seems matplotlib interprets the data as RGB
    axis.imshow(image, cmap='gray')

    axis.set_title(title)

def plot_histogram(bin_values, bin_counts, title, axis):
    """Plots the histogram as a bar chart using matplotlib."""
    axis.bar(bin_values[:-1], bin_counts, width=np.diff(bin_values), edgecolor='black', align='edge')
    axis.set_xlabel('Pixel Value')
    axis.set_ylabel('Count')
    axis.set_title(title)

if __name__ == "__main__":
    print("Program running...")
########################################################################################################################
    print("Part 1: Build a histogram...")
    gray_img1 = ski.io.imread('./images/houndog1.png', as_gray=True)
    gray_img2 = ski.io.imread('./images/airplane.jpg', as_gray=True)
    gray_img3 = ski.io.imread('./images/dark_light_climb.jpg', as_gray=True)
    gray_img4 = ski.io.imread('./images/ice_climb.jpg', as_gray=True)

    histogram1 = build_histogram_bins(gray_img1)
    histogram2 = build_histogram_bins(gray_img2)
    histogram3 = build_histogram_bins(gray_img3)
    histogram4 = build_histogram_bins(gray_img4)

    fig_img, axs_img = plt.subplots(2, 2, figsize=(10, 8))
    img_figure(gray_img1, 'Hound Dog', axs_img[0, 0])
    img_figure(gray_img2, 'Airplane', axs_img[0, 1])
    img_figure(gray_img3, 'Night time with flashlight', axs_img[1, 0])
    img_figure(gray_img4, 'Ice climb', axs_img[1, 1])
    plt.tight_layout()
    fig_img.savefig("example_img_grid")
    plt.show(block=False)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    plot_histogram(histogram1[0], histogram1[1], "Hound Dog image histogram", axs[0, 0])
    plot_histogram(histogram2[0], histogram2[1], "Airplane image histogram", axs[0, 1])
    plot_histogram(histogram3[0], histogram3[1], "Night time with flashlight image histogram", axs[1, 0])
    plot_histogram(histogram4[0], histogram4[1], "Ice climb image histogram", axs[1, 1])

    plt.tight_layout()
    fig.savefig("histogram_grid")
    plt.show(block=False)

########################################################################################################################
    print("Part 2: Regions and components...")
    img_th1 = double_side_thresholding(gray_img1, 200, 50)
    img_th2 = double_side_thresholding(gray_img2, 200, 50)
    img_th3 = double_side_thresholding(gray_img3, 200, 50)
    img_th4 = double_side_thresholding(gray_img4, 200, 50)

    histogram_th1 = build_histogram_bins(img_th1)
    histogram_th2 = build_histogram_bins(img_th2)
    histogram_th3 = build_histogram_bins(img_th3)
    histogram_th4 = build_histogram_bins(img_th4)

    fig_img_th, axs_img_th = plt.subplots(2, 2, figsize=(10, 8))
    img_figure(img_th1, 'Hound Dog thresholded', axs_img_th[0, 0])
    img_figure(img_th2, 'Airplane thresholded', axs_img_th[0, 1])
    img_figure(img_th3, 'Night time with flashlight thresholded', axs_img_th[1, 0])
    img_figure(img_th4, 'Ice climb thresholded', axs_img_th[1, 1])
    plt.tight_layout()
    fig_img.savefig("thresholded_img_grid")
    plt.show(block=False)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    plot_histogram(histogram_th1[0], histogram_th1[1], "Hound Dog image threshold histogram", axs[0, 0])
    plot_histogram(histogram_th2[0], histogram_th2[1], "Airplane image threshold histogram", axs[0, 1])
    plot_histogram(histogram_th3[0], histogram_th3[1], "Night time with flashlight image threshold histogram", axs[1, 0])
    plot_histogram(histogram_th4[0], histogram_th4[1], "Ice climb image threshold histogram", axs[1, 1])

    plt.tight_layout()
    fig.savefig("histogram_thresholded_grid")
    plt.show(block=False)

    labels1, num1 = connected_comps(img_th1, 100)
    labels2, num2 = connected_comps(img_th2, 150)
    labels3, num3 = connected_comps(img_th3, 20)
    labels4, num4 = connected_comps(img_th4, 150)

    colored_labels1 = ski.color.label2rgb(labels1, bg_label=0)
    colored_labels2 = ski.color.label2rgb(labels2, bg_label=0)
    colored_labels3 = ski.color.label2rgb(labels3, bg_label=0)
    colored_labels4 = ski.color.label2rgb(labels4, bg_label=0)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    img_figure(colored_labels1, 'Hound Dog Connected Components', axs[0, 0])
    img_figure(colored_labels2, 'Airplane Connected Components', axs[0, 1])
    img_figure(colored_labels3, 'Night time with flashlight Connected Components', axs[1, 0])
    img_figure(colored_labels4, 'Ice climb Connected Components', axs[1, 1])

    plt.tight_layout()
    fig.savefig("Connected_components_grid")
    plt.show(block=True)

########################################################################################################################
    print("Part 3: Histogram Equalization")
