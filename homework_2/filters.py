from numpy.lib.stride_tricks import as_strided
import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """

    kernel = resize(kernel)
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape

    out = np.zeros((Hi, Wi))

    for i in range(Hi):
        for j in range(Wi):
            for k in range(Hk):
                for m in range(Wk):
                    img_i, img_j = i + k - Hk // 2, j + m - Wk // 2
                    if img_i < 0 or img_i >= Hi or img_j < 0 or img_j >= Wi:
                        continue

                    value = image[img_i, img_j] * kernel[Hk - k - 1, Wk - m - 1]
                    out[i, j] += value

    return out


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (h, w).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (h+2*pad_height, w+2*pad_width).
    """

    h, w = image.shape
    out = np.zeros((h + 2 * pad_height, w + 2 * pad_width))
    out[pad_height: h + pad_height, pad_width: w + pad_width] = image

    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """

    rkernel = np.flip(kernel)
    rkernel = resize(rkernel)

    Hi, Wi = image.shape
    Hk, Wk = rkernel.shape
    out = np.zeros((Hi, Wi))

    Hb = Hk // 2
    Wb = Wk // 2
    conv_img = zero_pad(image, Hb, Wb)

    for i in range(Hb, Hb + Hi):
        for j in range(Wb, Wb + Wi):
            res = conv_img[i - Hb: i + Hb + 1, j - Wb: j + Wb + 1]
            out[i - Hb, j - Wb] = np.sum(res * rkernel)

    return out


def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """

    Hi, Wi = image.shape
    rkernel = np.flip(kernel)
    rkernel = resize(rkernel)
    Hk, Wk = rkernel.shape

    conv_img = zero_pad(image, Hk // 2, Wk // 2)
    padded_strided = as_strided(conv_img, (Hi, Wi, Hk, Wk), strides=conv_img.strides * 2)
    out = np.sum(padded_strided * rkernel, axis=(2, 3))

    return out


def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    return conv_faster(f, np.flip(g))


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    return conv_faster(f, np.flip(g) - g.mean())


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    kernel = resize(g)

    hi, wi = f.shape
    hk, wk = kernel.shape

    padded = zero_pad(f, hk // 2, wk // 2)

    strided_shape = (hi, wi, hk, wk)
    strided = as_strided(padded, strided_shape, strides=padded.strides * 2)

    strided_mean = strided.mean(axis=(2, 3))[:, :, np.newaxis, np.newaxis]
    strided_std = strided.std(axis=(2, 3))[:, :, np.newaxis, np.newaxis]
    strided = (strided - strided_mean) / strided_std
    kernel = (kernel - kernel.mean()) - kernel.std()

    out = np.sum(strided * kernel, axis=(2, 3))

    return out


def resize(kernel):
    if kernel.shape[0] % 2 == 0:
        kernel = np.vstack((kernel, np.zeros(kernel.shape[1])))

    if kernel.shape[1] % 2 == 0:
        kernel = np.hstack((kernel, np.zeros(kernel.shape[0])))

    return kernel