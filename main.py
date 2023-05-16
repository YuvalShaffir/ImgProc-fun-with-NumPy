import matplotlib.pyplot as plt
import numpy as np
import imageio
import scipy
from skimage.color import rgb2gray


GRAYSCALE = 1
RGB = 2
RGB_YIQ_TRANSFORMATION_MATRIX = np.array([[0.299, 0.587, 0.114],
                                          [0.596, -0.275, -0.321],
                                          [0.212, -0.523, 0.311]])
x = np.hstack([np.repeat(np.arange(0, 50, 2), 10)[None, :], np.array([255]*6)[None, :]])
grad = np.tile(x, (256, 1))


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    # read the image - check if it is greyscale or RGB image:
    img = imageio.imread(filename)

    # greyscale representation
    if representation == GRAYSCALE:
        img_g = rgb2gray(img) # incase it is an RGB image, turn it into grayscale
        img_g = img_g.astype('float64')
        img_g_norm = img_g / 255 
        return img_g

    # RGB representation
    if representation == RGB:
        img_rgb = img.astype('float64')
        img_rgb_norm = img_rgb / 255
        return img_rgb_norm


def imdisplay(filename, representation):
    """
    Reads an image and displays it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    """
    img = read_image(filename, representation)
    if representation == GRAYSCALE:
        plt.imshow(img, cmap="gray")
        plt.show()
    elif representation == RGB:
        plt.imshow(img)
        plt.show()


def rgb2yiq(imRGB):
    """
    Transform an RGB image into the YIQ color space
    :param imRGB: height X width X 3 np.float64 matrix in the [0,1] range
    :return: the image in the YIQ space
    """
    # We use the dot product to multipley the R,G,B channels with the R,G,B channels of the YIQ matrix.
    return np.dot(imRGB, RGB_YIQ_TRANSFORMATION_MATRIX.T)


def yiq2rgb(imYIQ):
    """
    Transform a YIQ image into the RGB color space
    :param imYIQ: height X width X 3 np.float64 matrix in the [0,1] range for
        the Y channel and in the range of [-1,1] for the I,Q channels
    :return: the image in the RGB space
    """
    # create inverse transformation matrix of the YIQ matrix
    yiq_to_rgb_matrix = np.linalg.inv(RGB_YIQ_TRANSFORMATION_MATRIX)
    return np.dot(imYIQ, yiq_to_rgb_matrix.T)


def check_if_rgb(img):
    """
    checks if image is RGB or grayscale
    :param img: original image
    :return: True if image is RGB, False if image is grayscale
    """
    if len(img.shape) < 3:
        return False
    elif len(img.shape) == 3: # has 3 channels: RGB
        return True


def histogram_equalize(im_orig):
    """
    Perform histogram equalization on the given image
    :param im_orig: Input float64 [0,1] image
    :return: [im_eq, hist_orig, hist_eq]
    """
    # get histogram and cumsum
    intensity_mat, is_rgb = get_intensity_arr(im_orig)
    hist_c, hist_orig, bin_edges = create_histograms(intensity_mat)
    min_level = np.min(hist_c[np.nonzero(hist_c)])
    hist_num_of_indxs = (len(hist_c)-1)
    
    # create the lookup table by multiplying: (max index) x (probabilty to get each index), then round it and make it an integer.
    look_up_tble = np.round_(hist_num_of_indxs * np.subtract(hist_c, min_level)/(hist_c[-1] - min_level)).astype(int)
    
    # generate new image from the lookup table
    hist_eq = get_hist_eq(hist_num_of_indxs, hist_orig, look_up_tble)
    img_eq = show_img(im_orig, intensity_mat, is_rgb, look_up_tble)
    
    return [img_eq, hist_orig, hist_eq]


def get_hist_eq(hist_num_of_indxs, hist_orig, look_up_tble):
    """
    map the original histogram with the mapping of the look-up table :
    :return: new equalized histogram
    """
    hist_eq = np.zeros_like(hist_orig)
    for indx in range(hist_num_of_indxs + 1):
        hist_eq[look_up_tble[indx]] += hist_orig[indx]
    return hist_eq


def create_histograms(intensity_mat):
    """
    create basic histogram and cumsum histogram based on the intensity matrix
    :param intensity_mat
    :return: the new historgrams
    """
    hist_orig, bin_edges = np.histogram(intensity_mat, bins=256, range=(0, 1))
    # compute the cumulative histogram
    hist_c = np.cumsum(hist_orig)
    return hist_c, hist_orig, bin_edges


def get_intensity_arr(im_orig):
    """
    create an intensity matrix based on the type of the image (RGB or grayscale)
    :param im_orig:
    :return: intensity matrix
            , is_rgb: True - RGB/ False - grayscale
    """
    # returns only the intencity channel of YIQ
    is_rgb = check_if_rgb(im_orig)
    if is_rgb:
        yiq_img = rgb2yiq(im_orig)
        intensity_mat = yiq_img[:, :, 0]
    else:
        intensity_mat = im_orig
    return intensity_mat, is_rgb


def show_img(im_orig, intensity_mat, is_rgb, look_up_tble, plot=False):
    """
    creates a new image from look up table
    :param plot: plots the new image if True.
    :return: the new image
    """
    unorm_img = (intensity_mat * 255).astype(int)

    if is_rgb:
        yiq_img = rgb2yiq(im_orig)
        yiq_img[:, :, 0] = look_up_tble[unorm_img] / 255
        new_image = yiq2rgb(yiq_img)
        if plot:
            plt.imshow(new_image)
            plt.show()
        return new_image
    else:
        new_image = look_up_tble[unorm_img] / 255
        if plot:
            plt.imshow(new_image, cmap="gray", vmin=0, vmax=1)
            plt.show()
        return new_image


def print_hist(hist_c, hist_eq, hist_orig, look_up_tble, plot=False):
    """
    prints the histograms and plots them if plot == True.
    """
    print(f"original histogram: {hist_orig.size}")
    print(f"cumulative histogram: {hist_c.size}")
    print(f"T array: {look_up_tble.size}")
    print(f"equalized histogram: {hist_eq.size}")
    if plot:
        plt.plot(hist_eq)
        plt.show()


def quantize_hist(hist_orig, n_quant, q_arr, z_arr):
    """
    returns a new histogram based on the q_arr and z_arr
    """
    new_hist = np.zeros_like(hist_orig)
    for i in range(n_quant):
        new_hist[z_arr[i]: z_arr[i + 1] + 1] = q_arr[i]
    return new_hist.astype(int)


def calc_quantize(hist_orig, n_iter, n_quant, z_arr):
    """
    calculates the error, the q array, and the z array, until
    error is stable.
    :return: all the calculated arrays.
    """
    q_arr = np.zeros(n_quant, dtype=float)
    error = np.zeros(n_iter, dtype=float)
    for itr in range(n_iter):
        for i in range(0, n_quant):
            numerator = 0
            denominator = 0
            for g in range(np.floor(z_arr[i]).astype(int) + 1, np.floor(z_arr[i + 1]).astype(int) + 1):
                numerator += (g * hist_orig[g])
                denominator += hist_orig[g]
                error[itr] += (((q_arr[i] - g) ** 2) * hist_orig[g])
            if denominator == 0:
                q_arr[i] = 0
            else:
                q_arr[i] = numerator / denominator
            if 0 < i < n_quant:
                z_arr[i] = (q_arr[i - 1] + q_arr[i]) / 2
        if itr > 0:
            if error[itr] == error[itr - 1]:
                break
    return error, q_arr, z_arr


def create_z_arr(hist_c, hist_orig, n_quant):
    """
    create z array, each value is an index of a segment in the histogram.
    each segment must have equal number of pixels.
    :return: new z array
    """
    partition = np.round(hist_c[-1] / n_quant).astype(int)
    z_arr = np.zeros(n_quant + 1, dtype=int)
    tmp = hist_c.copy()
    z_arr[0] = 0
    z_arr[-1] = 255
    for seg in range(1, n_quant):
        z_arr[seg] = np.where(tmp >= partition)[0][0]
        tmp -= partition
    # pixels = 0
    # segement = 1
    # z_arr_test = np.empty(n_quant + 1, dtype=int)
    # for indx in range(len(hist_orig)-1):
    #     if pixels < partition:
    #         pixels += hist_orig[indx]
    #     elif segement != len(z_arr_test) - 1:
    #         z_arr_test[segement] = indx
    #         segement += 1
    #         pixels = 0
    # z_arr_test[0] = -1
    # z_arr_test[-1] = 255
    # print(f"z array: {z_arr}")
    # print(f"z test: {z_arr_test}")
    return z_arr


def plot_hist(hist):
    plt.plot(hist)
    plt.show()


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input float64 [0,1] image
    :param n_quant: Number of intensities im_quant image will have
    :param n_iter: Maximum number of iterations of the optimization
    :return:  im_quant - is the quantized output image
              error - is an array with shape (n_iter,) (or less) of
                the total intensities error for each iteration of the
                quantization procedure
    """
    intensity_mat, is_rgb = get_intensity_arr(im_orig)
    hist_c, hist_orig, bin_edges = create_histograms(intensity_mat)
    z_arr = create_z_arr(hist_c, hist_orig, n_quant)
    error, q_arr, z_arr = calc_quantize(hist_orig, n_iter, n_quant, z_arr)
    lookup_table = quantize_hist(hist_orig, n_quant, q_arr, z_arr)
    im_quant = show_img(im_orig, intensity_mat, is_rgb, lookup_table)
    return [im_quant, error]


def quantize_rgb(im_orig, n_quant): # Bonus - optional
    """
    Performs optimal quantization of a given greyscale or RGB image
    :param im_orig: Input RGB image of type float64 in the range [0,1]
    :param n_quant: Number of intensities im_quant image will have
    :return:  im_quant - the quantized output image
    """
    pass
