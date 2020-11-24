import random
import cv2 as cv
import numpy as np


def display(image, name):
    """Display given image.
                    Parameters
                    ----------
                    out : OpenCV type
                       The image to be shown
                    name : str
                       Title of the window
                    """
    cv.imshow(name, image)
    cv.waitKey()
    cv.destroyAllWindows()


def addGaussianNoise(image, setting):
    output = np.zeros(image.shape, np.uint8)  # Creation of the blank output image
    th = 1 - setting
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()  # Randomize places of the "salt and pepper"
            if rdn < setting:
                output[i][j] = 0
            elif rdn > th:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def medianFiltering(image, coef):
    """Removes salt and pepper noise
                     Parameters
                     ----------
                     image : OpenCV type
                        The image to be shown
                     coef : int
                        Coefficient of the blur use in order to remove noises
                     """
    median = cv.medianBlur(image, coef)
    compare = np.concatenate((image, median), axis=1)

    return compare


def sharpening(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    im = cv.filter2D(image, -1, kernel)
    display(im, 'Sharpened Image')


img_high = cv.imread('Photo_HF.jpeg', 1)
img_low = cv.imread('Photo_LOW.jpg', 1)

display(img_high, 'Original Image')

noisy_high_1 = addGaussianNoise(img_high, 0.01)
noisy_high_2 = addGaussianNoise(img_high, 0.05)

display(noisy_high_1, 'After')
display(noisy_high_2, 'After')

filtered_SNP = medianFiltering(noisy_high_2, 5)
display(filtered_SNP, 'Repaired')

sharpening(filtered_SNP)