# pylint: disable=redefined-outer-name, unreachable, missing-function-docstring
"""
Image Processing Project 0: Starter Code
"""

import os
import numpy as np
import skimage as ski
import matplotlib.pyplot as plt

# Question 1.a:
def numpy_vec_op():
    # TODO: generate a numpy vector containing 0-4 using numpy function arange()
    ##  ***************** Your code starts here ***************** ##

    arr = np.arange(start=0, stop=5)

    ##  ***************** Your code ends here ***************** ##

    x = np.array([4, 5, 6, 7, 8])

    # TODO: perform a dot product with 'x', store it in 'result'

    ##  ***************** Your code starts here ***************** ##

    result = np.dot(arr, x)

    ##  ***************** Your code ends here ***************** ##

    if result != 70.0:
        print("\033[91mIncorrect!\033[00m")
    else:
        print("\033[92mCorrect!\033[00m")


# Question 1.b:
def numpy_mat_op():
    # TODO: generate a numpy matrix [[0, 1], [2, 3]] using numpy function arange() and reshape()
    ##  ***************** Your code starts here ***************** ##

    arr = np.arange(0, 4)
    B = arr.reshape(2, 2)

    ##  ***************** Your code ends here ***************** ##

    A = np.array([[5, 6], [7, 8]])

    # TODO: perform a dot product with 'A'
    ##  ***************** Your code starts here ***************** ##

    result = np.dot(B, A)

    ##  ***************** Your code ends here ***************** ##

    if np.sum(result - np.array([[7, 8], [31, 36]])) != 0:
        print("\033[91mIncorrect!\033[00m")
    else:
        print("\033[92mCorrect!\033[00m")


# Question 1.c:
def numpy_3d_op():
    # TODO: create a 10x10x3 matrix (think of this as an RGB image)
    ##  ***************** Your code starts here ***************** ##

    fake_rgb = np.ones((10, 10, 3))

    ##  ***************** Your code ends here ***************** ##

    weights = [0.3, 0.6, 0.1]

    # TODO: Compute the dot-product of the 10x10x3 matrix and 'weights' to get a 10x10 matrix
    ##  ***************** Your code starts here ***************** ##

    result = np.dot(fake_rgb, weights)

    ##  ***************** Your code ends here ***************** ##
    # Notice that you just took your 3-color "image" and made it 1 color!

    if result.shape != (10, 10):
        print("\033[91mIncorrect!\033[00m")
    else:
        print("\033[92mCorrect!\033[00m")


def image_io(folder_path: str) -> None:
    """ Reads in images, converts them to grayscale, displays them and saves the image grid
    """
    grayscale_weights = [0.3, 0.6, 0.1]
    images = []
    print("Reading in files")
    for file in os.listdir(folder_path):
        # Read the images
        file_path = os.path.join(folder_path, file)
        # Note this required installing the imagecodecs to read a Tiff file. Unsure why it wasn't required by skimage.
        image = ski.io.imread(file_path)

        # Convert to grayscale and add to list
        if image.ndim == 3:
            # Only convert to grayscale if the image actually has 3 dimensions
            print(f"Convert file {file} to grayscale using weights: {grayscale_weights}")
            gray_img = np.dot(image[..., :3], grayscale_weights)
            images.append(gray_img)
        else:
            print(f"File: {file} already in grayscale")
            images.append(image)


    # Display in a grid using plt.subplots
    fig, axs = plt.subplots(2, 2)
    # Specify the color map is gray, otherwise it seems matplotlib interprets the data as RGB
    axs[0, 0].imshow(images[0], cmap='gray')
    axs[0, 1].imshow(images[1], cmap='gray')
    axs[1, 0].imshow(images[2], cmap='gray')
    axs[1, 1].imshow(images[3], cmap='gray')
    # Save images into files first
    print ("Saving image grid as a new file")
    fig.savefig("image_grid")

    # Then show the plot
    print ("Displaying images")
    plt.show()


if __name__ == "__main__":
    print("Question 1.a: ", end="")
    numpy_vec_op()
    print("Question 1.b: ", end="")
    numpy_mat_op()
    print("Question 1.c: ", end="")
    numpy_3d_op()
    print("Question 2:")
    image_io("./images")
