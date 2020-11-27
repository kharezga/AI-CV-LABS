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
        final = cv.addWeighted(self.image, 1.5, image, -0.5, 0)
        self.__display(final, 'Sharpened Image')


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
        # Another approach
        #  kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        # im = cv.filter2D(image, -1, kernel)

        final = cv.addWeighted(self.image, 1.5, image, -0.5, 0)
        self.__display(final, 'Sharpened Image')


class T3:
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

    def sobelFilter(self, image):
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)

        grad_x = cv.Sobel(gray_image, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
        grad_y = cv.Sobel(gray_image, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)

        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)

        grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        # grad = cv.addWeighted(abs_grad_x, 1.5, abs_grad_y, -0.5, 0)

        self.__display(grad, 'Sobel filter')


class T4:
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

    def laplaceFilter(self, kernel):

        img_bw = cv.cvtColor(self.image, cv.COLOR_RGB2GRAY)
        m, n = kernel.shape

        d = int((m - 1) / 2)
        h, w = img_bw.shape[0], img_bw.shape[1]

        dst = np.zeros((h, w))

        for y in range(d, h - d):
            for x in range(d, w - d):
                dst[y][x] = np.sum(img_bw[y - d:y + d + 1, x - d:x + d + 1] * kernel)

        self.__display(dst, 'My Laplace')

    def buildInFilter(self):
        image = cv.GaussianBlur(self.image, (3, 3), 0)
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        kernel = np.array([[1, 1, 1],
                           [1, -8, 1],
                           [1, 1, 1]])

        dst = cv.Laplacian(image_gray, cv.CV_64F, kernel)

        self.__display(dst, 'Build In Laplace')


def main():
    img_high = cv.imread('Photo_HF.jpeg', 1)
    img_low = cv.imread('Photo_LOW.jpg', 1)

    # Objects for Task 1
    task1_hf = T1(img_high, 'High Freq.')
    task1_lf = T1(img_low, 'Low Freq.')

    noisy_high1 = task1_hf.addSNP(0.01)
    noisy_high1 = task1_hf.medianFiltering(noisy_high1, 5)
    task1_hf.sharpening(noisy_high1)

    # TODO Implementacja obrazu lf

    # Objects for Task 2
    task2_hf = T2(img_high, 'High freq.')
    task2_lf = T2(img_low, 'Low Freq.')

    gauss_high1 = task2_hf.addGaussian()
    gauss_high1 = task2_hf.GaussianFiltering(gauss_high1)
    task2_hf.sharpening(gauss_high1)

    # TODO Implementacja obrazu lf

    # Objects for the Task 3/4
    task3_hf = T3(img_high, 'High freq.')
    task3_lf = T3(img_low, 'Low Freq.')

    # TASK 3
    task3_hf.sobelFilter(img_high)

    gauss_sobel = task2_hf.addGaussian()
    task3_hf.sobelFilter(gauss_sobel)

    # TASK 4
    task4_hf = T4(img_high, 'High freq.')

    task4_hf.buildInFilter()

    t4_kernel = np.array([[1, 1, 1],
                          [1, -8, 1],
                          [1, 1, 1]])
    
    task4_hf.laplaceFilter(t4_kernel)


if __name__ == "__main__":
    main()
