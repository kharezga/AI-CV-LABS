# AI&CV Lab 4 - Kacper Harezga 249111


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
    dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    plot.subplot(122), plot.imshow(magnitude_spectrum, cmap='gray')
    plot.title('DFT '), plot.xticks([]), plot.yticks([])

    row = img.shape[0]
    col = img.shape[1]
    rrow = row // 2
    ccol = col // 2


    mask = np.ones((row, col, 2), np.uint8)
    mask[rrow:rrow, ccol:ccol] = 0

    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)


    res = cv.idft(f_ishift)
    res = cv.magnitude(res[:, :, 0], res[:, :, 1])


    plot.subplot(223), plot.imshow(res, cmap='gray')
    plot.title('Repaired image'), plot.xticks([]), plot.yticks([])

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


    histogram('High Histogram', img_high)
    histogram('Low Histogram', img_low)

    img_thresholding(img_lena, cv.THRESH_BINARY, 'Binary Thresholding')
    img_thresholding(img_lena, cv.THRESH_BINARY_INV, 'Binary Inverted Thresholding')
    img_thresholding(img_lena, cv.THRESH_TRUNC, 'Trunc Thresholding')

    DFT(img_lena_bw)

    histogram('Quantized High - 32', quantizator(img_high, 32))
    histogram('Quantized Low - 32', quantizator(img_low, 32))

    # histogram('Quantized High - 64', quantizator(img_high, 64))
    # histogram('Quantized Low - 64', quantizator(img_low, 64))

    # histogram('Quantized High - 128', quantizator(img_high, 128))
    # histogram('Quantized Low - 128', quantizator(img_low, 128))

if __name__ == "__main__":
    main()
