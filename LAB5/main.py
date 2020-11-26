import random
import cv2 as cv
import numpy as np


class T1:
    def __init__(self, image, name):
        self.image = image
        self.name = name

    def __display(self, image, title):
        """Display given image.
                    Parameters
                    ----------
                    out : OpenCV type
                       The image to be shown
                    name : str
                       Title of the window
                    """
        cv.imshow(self.name + ' ' + title, image)
        cv.waitKey()
        cv.destroyAllWindows()

    def addSNP(self, setting):
        self.__display(self.image, "Original Image")
        output = np.zeros(self.image.shape, np.uint8)  # Creation of the blank output image
        th = 1 - setting
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                rdn = random.random()  # Randomize places of the "salt and pepper"
                if rdn < setting:
                    output[i][j] = 0
                elif rdn > th:
                    output[i][j] = 255
                else:
                    output[i][j] = self.image[i][j]
        return output

    def medianFiltering(self, img, coef):
        """Removes salt and pepper noise
                         Parameters
                         ----------
                         image : OpenCV type
                            The image to be shown
                         coef : int
                            Coefficient of the blur use in order to remove noises
                         """
        median = cv.medianBlur(img, coef)
        compare = np.concatenate((img, median), axis=1)

        self.__display(compare, 'SNP and median filtering')
        return median

    def sharpening(self, image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        im = cv.filter2D(image, -1, kernel)
        self.__display(im, 'Sharpened Image')


class T2:
    def __init__(self, image, name):
        self.image = image
        self.name = name

    def __display(self, image, title):
        """Display given image.
                    Parameters
                    ----------
                    out : OpenCV type
                       The image to be shown
                    name : str
                       Title of the window
                    """
        cv.imshow(self.name + ' ' + title, image)
        cv.waitKey()
        cv.destroyAllWindows()

    def addGaussian(self):
        # Generate Gaussian noise
        gauss = np.random.normal(0, 1, self.image.size)
        gauss = gauss.reshape(self.image.shape[0], self.image.shape[1], self.image.shape[2]).astype('uint8')
        # Add the Gaussian noise to the image
        img_gauss = cv.add(self.image, gauss)
        self.__display(img_gauss, 'Image with Gaussian Noise')
        
        return img_gauss

    def GaussianFiltering(self, image):
        im = cv.GaussianBlur(image, (3, 3), 0, borderType=cv.BORDER_CONSTANT)
        self.__display(im, 'Filtered Image Gaussian')

        return im

    def sharpening(self, image):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        im = cv.filter2D(image, -1, kernel)
        self.__display(im, 'Sharpened Image')


def main():
    img_high = cv.imread('Photo_HF.jpeg', 1)
    img_low = cv.imread('Photo_LOW.jpg', 1)

    # TASK1 for the high frequency image
    task1_hf = T1(img_high, 'High Freq.')
    task1_lf = T1(img_low, 'Low Freq.')

    noisy_high1 = task1_hf.addSNP(0.01)
    noisy_high1 = task1_hf.medianFiltering(noisy_high1, 5)
    task1_hf.sharpening(noisy_high1)

    # TODO Implementacja obrazu lf

    # TASK2 for the high frequency image
    task2_hf = T2(img_high, 'High freq.')
    task2_lf = T1(img_low, 'Low Freq.')

    gauss_high1 = task2_hf.addGaussian()
    gauss_high1 = task2_hf.GaussianFiltering(gauss_high1)
    task2_hf.sharpening(gauss_high1)


if __name__ == "__main__":
    main()
