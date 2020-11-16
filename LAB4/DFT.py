import cv2 as cv
import numpy as np
from matplotlib import pyplot as plot


class DFT:
    def __init__(self, image):
        self.image = image


def discrete_fourier_transform_np(self):
    f = np.fft.fft2(self.image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    plot.subplot(121), plot.imshow(self.image, cmap='gray')
    plot.title('Input Image'), plot.xticks([]), plot.yticks([])
    plot.subplot(122), plot.imshow(magnitude_spectrum, cmap='gray')
    plot.title('Magnitude Spectrum'), plot.xticks([]), plot.yticks([])
    plot.show()


