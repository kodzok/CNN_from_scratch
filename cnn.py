import numpy as np
import pickle
from tqdm import tqdm

from convolution import *
from maxpool import *
from utils import *
from extract_data import *


def cnn(image, label, params, conv_stride, pool_size, pool_stride):
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    ########### forward propogate ###########

    image = image.reshape(image.shape[1], image.shape[2], image.shape[0])
    conv1 = convolution(image, f1, b1, conv_stride)  # convolution operation
    conv1 = relu(conv1)  # relu activation

    conv2 = convolution(conv1, f2, b2, conv_stride)  # convolution operation
    conv2 = relu(conv2)  # relu activation

    pooling = maxpool(conv2, pool_size, pool_stride)  # maxpooling operation

    n_H2, n_W2, n_C2 = pooling.shape
    flatten = pooling.reshape((n_H2 * n_W2 * n_C2, 1))  # flatten maxpooling layer

    # first dense layer
    Z = np.dot(w3, flatten) + b3  # first dense layer
    A = relu(Z)  # relu activation

    # second dense layer
    out = np.dot(w4, A) + b4  # second dense layer
    probs = softmax(out)  # predict class probabilities with the softmax activation function

    loss = categoricalCrossEntropy(probs, label)  # categorical cross-entropy loss

    ########### backward propogate ###########

    dout = probs - label  # derivative of loss w.r.t. final dense layer output
    dw4 = np.dot(dout, A.T)
    db4 = np.sum(dout, axis=1, keepdims=True)
    dA = np.dot(w4.T, dout)  # loss gradient of first dense layer outputs

    dZ = np.multiply(dA, np.int64(A > 0))  # relu backward
    dw3 = np.dot(dZ, flatten.T)
    db3 = np.sum(dZ, axis=1, keepdims=True)
    dflatten = np.dot(w3.T, dZ)

    dpool = dflatten.reshape(pooling.shape)  # reshape fully connected into dimensions of pooling layer

    dconv2 = maxpool_backward(dpool, conv2, pool_size, pool_stride)
    dconv2 = relu_backward(dconv2, conv2)

    dconv1, df2, db2 = conv_backward(dconv2, conv1, f2, conv_stride)
    dconv1 = relu_backward(dconv1, conv1)

    dimage, df1, db1 = conv_backward(dconv1, image, f1, conv_stride)

    grads = [df1, df2, dw3, dw4, db1, db2, db3, db4]

    return grads, loss


def adamGD(batch, num_classes, lr, dim, n_c, beta1, beta2, params, cost):
    '''
    update the parameters through Adam gradient descent.
    '''
    [f1, f2, w3, w4, b1, b2, b3, b4] = params

    X = batch[:, 0:-1]  # get batch inputs
    X = X.reshape(len(batch), n_c, dim, dim)
    Y = batch[:, -1]  # get batch labels

    cost_ = 0
    batch_size = len(batch)

    # initialize gradients and momentum,RMS params
    df1 = np.zeros(f1.shape)
    df2 = np.zeros(f2.shape)
    dw3 = np.zeros(w3.shape)
    dw4 = np.zeros(w4.shape)
    db1 = np.zeros(b1.shape)
    db2 = np.zeros(b2.shape)
    db3 = np.zeros(b3.shape)
    db4 = np.zeros(b4.shape)

    v1 = np.zeros(f1.shape)
    v2 = np.zeros(f2.shape)
    v3 = np.zeros(w3.shape)
    v4 = np.zeros(w4.shape)
    bv1 = np.zeros(b1.shape)
    bv2 = np.zeros(b2.shape)
    bv3 = np.zeros(b3.shape)
    bv4 = np.zeros(b4.shape)

    s1 = np.zeros(f1.shape)
    s2 = np.zeros(f2.shape)
    s3 = np.zeros(w3.shape)
    s4 = np.zeros(w4.shape)
    bs1 = np.zeros(b1.shape)
    bs2 = np.zeros(b2.shape)
    bs3 = np.zeros(b3.shape)
    bs4 = np.zeros(b4.shape)

    for i in range(batch_size):
        x = X[i]
        y = np.eye(num_classes)[int(Y[i])].reshape(num_classes, 1)  # convert label to one-hot
        # Collect Gradients for training example
        grads, loss = cnn(x, y, params, 1, 2, 2)
        [df1_, df2_, dw3_, dw4_, db1_, db2_, db3_, db4_] = grads

        df1 += df1_
        db1 += db1_
        df2 += df2_
        db2 += db2_
        dw3 += dw3_
        db3 += db3_
        dw4 += dw4_
        db4 += db4_

        cost_ += loss

    # Parameter Update

    v1 = beta1 * v1 + (1 - beta1) * df1 / batch_size  # momentum update
    s1 = beta2 * s1 + (1 - beta2) * (df1 / batch_size) ** 2  # RMSProp update
    f1 -= lr * v1 / np.sqrt(s1 + 1e-7)  # combine momentum and RMSProp to perform update with Adam

    bv1 = beta1 * bv1 + (1 - beta1) * db1 / batch_size
    bs1 = beta2 * bs1 + (1 - beta2) * (db1 / batch_size) ** 2
    b1 -= lr * bv1 / np.sqrt(bs1 + 1e-7)

    v2 = beta1 * v2 + (1 - beta1) * df2 / batch_size
    s2 = beta2 * s2 + (1 - beta2) * (df2 / batch_size) ** 2
    f2 -= lr * v2 / np.sqrt(s2 + 1e-7)

    bv2 = beta1 * bv2 + (1 - beta1) * db2 / batch_size
    bs2 = beta2 * bs2 + (1 - beta2) * (db2 / batch_size) ** 2
    b2 -= lr * bv2 / np.sqrt(bs2 + 1e-7)

    v3 = beta1 * v3 + (1 - beta1) * dw3 / batch_size
    s3 = beta2 * s3 + (1 - beta2) * (dw3 / batch_size) ** 2
    w3 -= lr * v3 / np.sqrt(s3 + 1e-7)

    bv3 = beta1 * bv3 + (1 - beta1) * db3 / batch_size
    bs3 = beta2 * bs3 + (1 - beta2) * (db3 / batch_size) ** 2
    b3 -= lr * bv3 / np.sqrt(bs3 + 1e-7)

    v4 = beta1 * v4 + (1 - beta1) * dw4 / batch_size
    s4 = beta2 * s4 + (1 - beta2) * (dw4 / batch_size) ** 2
    w4 -= lr * v4 / np.sqrt(s4 + 1e-7)

    bv4 = beta1 * bv4 + (1 - beta1) * db4 / batch_size
    bs4 = beta2 * bs4 + (1 - beta2) * (db4 / batch_size) ** 2
    b4 -= lr * bv4 / np.sqrt(bs4 + 1e-7)

    cost_ = cost_ / batch_size
    cost.append(cost_)

    params = [f1, f2, w3, w4, b1, b2, b3, b4]

    return params, cost


def train(num_classes=10, lr=0.01, beta1=0.95, beta2=0.99, img_dim=28, img_depth=1, f=5, num_filt1=8, num_filt2=8,
          batch_size=32, num_epochs=2, save_path='params.pkl'):
    # Get training data
    m = 50000
    X = extract_data('train-images-idx3-ubyte.gz', m, img_dim)
    y_dash = extract_labels('train-labels-idx1-ubyte.gz', m).reshape(m, 1)
    X -= int(np.mean(X))
    X /= int(np.std(X))
    train_data = np.hstack((X, y_dash))

    np.random.shuffle(train_data)

    ## Initializing all the parameters
    f1, f2, w3, w4 = (f, f, img_depth, num_filt1), (f, f, num_filt1, num_filt2), (128, 800), (10, 128)
    f1 = initializeFilter(f1)
    f2 = initializeFilter(f2)
    w3 = initializeWeight(w3)
    w4 = initializeWeight(w4)

    b1 = np.zeros((1, 1, 1, f1.shape[3]))
    b2 = np.zeros((1, 1, 1, f2.shape[3]))
    b3 = np.zeros((w3.shape[0], 1))
    b4 = np.zeros((w4.shape[0], 1))

    params = [f1, f2, w3, w4, b1, b2, b3, b4]

    cost = []

    print("LR:" + str(lr) + ", Batch Size:" + str(batch_size))

    for epoch in range(num_epochs):
        np.random.shuffle(train_data)
        batches = [train_data[k:k + batch_size] for k in range(0, train_data.shape[0], batch_size)]

        t = tqdm(batches)
        for x, batch in enumerate(t):
            params, cost = adamGD(batch, num_classes, lr, img_dim, img_depth, beta1, beta2, params, cost)
            t.set_description("Cost: %.2f" % (cost[-1]))

    with open(save_path, 'wb') as file:
        pickle.dump(params, file)

    return cost


def predict(image, f1, f2, w3, w4, b1, b2, b3, b4, conv_stride=1, pool_size=2, pool_stride=2):
    '''
    Make predictions with trained filters/weights.
    '''
    image = image.reshape(image.shape[1], image.shape[2], image.shape[0])
    conv1 = convolution(image, f1, b1, conv_stride)  # convolution operation
    conv1 = relu(conv1)  # relu activation

    conv2 = convolution(conv1, f2, b2, conv_stride)  # convolution operation
    conv2 = relu(conv2)  # relu activation

    pooling = maxpool(conv2, pool_size, pool_stride)  # maxpooling operation

    n_H2, n_W2, n_C2 = pooling.shape
    flatten = pooling.reshape((n_H2 * n_W2 * n_C2, 1))  # flatten maxpooling layer

    Z = np.dot(w3, flatten) + b3  # first dense layer
    A = relu(Z)  # relu activation

    out = np.dot(w4, A) + b4  # second dense layer
    probs = softmax(out)  # predict class probabilities with the softmax activation function

    return np.argmax(probs), np.max(probs)
