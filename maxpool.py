import numpy as np


def maxpool(image, f=2, stride=2):
    """
    Implements the forward pass of the pooling layer

    Arguments:
    image -- Input data, numpy array of shape (n_H_prev, n_W_prev, n_C_prev)
    f - pooling_size

    Returns:
    A -- output of the pool layer, a numpy array of shape (n_H, n_W, n_C)
    """

    (n_H_prev, n_W_prev, n_C_prev) = image.shape

    # Defining the dimensions of the output
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initializing output matrix A
    A = np.zeros((n_H, n_W, n_C))

    for h in range(n_H):  # loop on the vertical axis of the output volume
        # Vertical start and end of the current "slice"
        vert_start = h * stride
        vert_end = h * stride + f

        for w in range(n_W):  # loop on the horizontal axis of the output volume
            # Horizontal start and end of the current "slice"
            horiz_start = w * stride
            horiz_end = w * stride + f

            for c in range(n_C):  # loop over the channels of the output volume

                # Use the corners to define the current slice on the ith training example of A_prev, channel c
                image_slice = image[vert_start:vert_end, horiz_start:horiz_end, c]

                # Computation the pooling operation on the slice.
                A[h, w, c] = np.max(image_slice)

    # Making sure output shape is correct
    assert (A.shape == (n_H, n_W, n_C))

    return A


def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.

    Arguments:
    x -- Array of shape (f, f)

    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    mask = (x == np.max(x))

    return mask


def maxpool_backward(dA, A_prev, f, stride):
    """
    Implements the backward pass of the pooling layer

    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    f - pooling size
    A_prev - 

    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """

    # Retrieving dimensions from A_prev's shape and dA's shape
    n_H_prev, n_W_prev, n_C_prev = A_prev.shape
    n_H, n_W, n_C = dA.shape

    # Initialization dA_prev with zeros
    dA_prev = np.zeros((n_H_prev, n_W_prev, n_C_prev))

    for h in range(n_H):  # loop on the vertical axis
        for w in range(n_W):  # loop on the horizontal axis
            for c in range(n_C):  # loop over the channels (depth)

                # corners of the current "slice"
                vert_start = h * stride
                vert_end = h * stride + f
                horiz_start = w * stride
                horiz_end = w * stride + f

                # Using the corners and "c" to define the current slice from a_prev
                a_prev_slice = A_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                # Creating the mask from a_prev_slice
                mask = create_mask_from_window(a_prev_slice)
                # Setting dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA)
                dA_prev[vert_start: vert_end, horiz_start: horiz_end, c] += np.multiply(mask, dA[h, w, c])

    # Making sure output shape is correct
    assert (dA_prev.shape == A_prev.shape)

    return dA_prev
