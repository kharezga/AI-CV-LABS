# AI&CV Lab 4 - Kacper Harezga 249111

import DFT
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plot


def display(out, name):
    cv.imshow(name, out)
    cv.waitKey(0)
    cv.destroyAllWindows()


def histogram(fname, file):
    cv.imshow(fname, file)  # display the current image
    plot.hist(file.ravel(), 256, [0, 256])  # histogram calculation
    plot.title(fname + ' - histogram', fontdict=None, loc='center', pad=None)  # setting figure title
    plot.show()  # plotting histogram
    cv.waitKey(0)
    cv.destroyAllWindows(),



def equal_hist(fname, file):
    img = cv.cvtColor(file, cv.COLOR_BGR2GRAY)  # change colour to the black and white
    eql = cv.equalizeHist(img, None)  # histogram equalization
    cv.imshow(fname + 'black_n_white', img)  # displaying original image
    cv.waitKey(0)
    cv.destroyAllWindows()
    histogram('Equalized ' + fname, eql)  # displaying equalized image


def quantizator(fname, level=32):
    out = np.zeros((fname.shape[0], fname.shape[1]), dtype='uint8')
    max = np.max(fname)
    min = np.min(fname)

    difference = (max - min) // level
    count = 0

    for x in range(fname.shape[0]):
        for y in range(fname.shape[1]):
            out[x, y] = fname[x, y]
            while True:
                if out[x, y] <= min + difference or count >= level:
                    break
                else:
                    out[x, y] = out[x, y] - difference
                    count += 1
            out[x, y] = count * difference + min + difference // 2
            count = 0
    return out


def img_thresholding(img, type_of_thresh, title):
    ret, thresh = cv.threshold(img, 127, 255, type_of_thresh)
    histogram(title, thresh)


def DFT(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    plot.subplot(121), plot.imshow(img, cmap='gray')
    plot.title('Input Image'), plot.xticks([]), plot.yticks([])
    plot.subplot(122), plot.imshow(magnitude_spectrum, cmap='gray')
    plot.title('Magnitude Spectrum'), plot.xticks([]), plot.yticks([])
    plot.show()


def inv_DFT(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    rows, cols = img.shape
    crow, ccol = rows / 2, cols / 2
    fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plot.subplot(131), plot.imshow(img, cmap='gray')
    plot.title('Input Image'), plot.xticks([]), plot.yticks([])
    plot.subplot(132), plot.imshow(img_back, cmap='gray')
    plot.title('Image after HPF'), plot.xticks([]), plot.yticks([])
    plot.subplot(133), plot.imshow(img_back)
    plot.title('Result in JET'), plot.xticks([]), plot.yticks([])

    plot.show()




def main():
    # PATH DEFINITIONS
    image_high = 'Photo_HF.jpeg'
    image_low = 'Photo_LOW.jpg'
    image_lena = 'lena.png'

    # FILES LOADING
    img_high = cv.imread(image_high, 0)  # loading the file
    img_low = cv.imread(image_low, 0)  # loading the file
    img_lena = cv.imread(image_lena, 1)
    img_lena_bw = cv.imread(image_lena, 0)

    DFT(img_lena_bw)
    #  inv_DFT(img_lena_bw)

    histogram('High Histogram', img_high)
    histogram('Low Histogram', img_low)

    img_thresholding(img_lena, cv.THRESH_BINARY, 'Binary Thresholding')
    img_thresholding(img_lena, cv.THRESH_BINARY_INV, 'Binary Inverted Thresholding')
    img_thresholding(img_lena, cv.THRESH_TRUNC, 'Trunc Thresholding')

    histogram('Quantized High - 32', quantizator(img_high, 32))
    histogram('Quantized Low - 32', quantizator(img_low, 32))

    # histogram('Quantized High - 64', quantizator(img_high, 64))
    # histogram('Quantized Low - 64', quantizator(img_low, 64))

    # histogram('Quantized High - 128', quantizator(img_high, 128))
    # histogram('Quantized Low - 128', quantizator(img_low, 128))

if __name__ == "__main__":
    main()
