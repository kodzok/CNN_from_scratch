import numpy as np


def conv_single_step(image_slice, filt, bias):
    """
    Apply one filter defined by parameters filt on a single slice (image_slice) of the output activation
    of the previous layer.

    Arguments:
    image_slice -- slice of input data of shape (f, f, n_C_prev)
    filt -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    bias -- Bias parameters contained in a window - matrix of shape (1, 1, 1)

    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    # Element-wise product between a_slice_prev and W
    s = np.multiply(image_slice, filt)
    # Sum over all entries of the volume s
    Z = np.sum(s)
    # Adding bias b to Z
    Z = Z + bias

    return Z


def convolution(image, filt, bias, stride=1, padding=0):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    image -- output activations of the previous layer,
        numpy array of shape (n_H_prev, n_W_prev, n_C_prev)
    filt -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    bias -- Biases, numpy array of shape (1, 1, 1, n_C)

    Returns:
    Z -- conv output, numpy array of shape (n_H, n_W, n_C)
    """

    (f, f, n_C_prev, n_C) = filt.shape

    (n_H_prev, n_W_prev, n_C_prev) = image.shape

    # Computation of the CONV output volume dimensions
    n_H = int((n_H_prev + 2 * padding - f) / stride + 1)
    n_W = int((n_W_prev + 2 * padding - f) / stride + 1)

    Z = np.zeros((n_H, n_W, n_C))

    for h in range(n_H):  # loop over vertical axis of the output volume
        # Vertical start and end of the current "slice"
        vert_start = h * stride
        vert_end = h * stride + f

        for w in range(n_W):  # loop over horizontal axis of the output volume
            # Horizontal start and end of the current "slice"
            horiz_start = w * stride
            horiz_end = w * stride + f

            for c in range(n_C):  # loop over channels (= #filters) of the output volume

                # 3D slice of a_prev_pad
                image_slice = image[vert_start:vert_end, horiz_start:horiz_end, :]

                # Convolution the (3D) slice with the correct filter W and bias b, to get back one output neuron
                weights = filt[:, :, :, c]
                biases = bias[:, :, :, c]
                Z[h, w, c] = conv_single_step(image_slice, weights, biases)

    # Making sure output shape is correct
    assert (Z.shape == (n_H, n_W, n_C))

    return Z


def conv_backward(dZ, A_prev, filt, stride):
    """
    Implementation the backward propagation for a convolution function

    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (n_H, n_W, n_C)

    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (n_H_prev, n_W_prev, n_C_prev)
    dfilt -- gradient of the cost with respect to the weights of the conv layer (filt)
          numpy array of shape (f, f, n_C_prev, n_C)
    dbias -- gradient of the cost with respect to the biases of the conv layer (bias)
          numpy array of shape (1, 1, 1, n_C)
    """

    (n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = filt.shape
    (n_H, n_W, n_C) = dZ.shape

    # Initialization dA_prev, dW, db with the correct shapes
    dA_prev = np.zeros((n_H_prev, n_W_prev, n_C_prev))
    dfilt = np.zeros((f, f, n_C_prev, n_C))
    dbias = np.zeros((1, 1, 1, n_C))

    for h in range(n_H):  # loop over vertical axis of the output volume
        for w in range(n_W):  # loop over horizontal axis of the output volume
            for c in range(n_C):  # loop over the channels of the output volume

                # Corners of the current "slice"
                vert_start = h * stride
                vert_end = h * stride + f
                horiz_start = w * stride
                horiz_end = w * stride + f

                # Using the corners to define the slice from a_prev_pad
                a_slice = A_prev[vert_start:vert_end, horiz_start:horiz_end, :]

                # Updating gradients for the window and the filter's parameters
                dA_prev[vert_start:vert_end, horiz_start:horiz_end, :] += filt[:, :, :, c] * dZ[h, w, c]
                dfilt[:, :, :, c] += a_slice * dZ[h, w, c]
                dbias[:, :, :, c] += dZ[h, w, c]

    # Making sure your output shape is correct
    assert (dA_prev.shape == (n_H_prev, n_W_prev, n_C_prev))

    return dA_prev, dfilt, dbias


