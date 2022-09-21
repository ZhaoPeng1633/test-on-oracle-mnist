from sklearn.metrics import mean_squared_error
import numpy as np
import pickle
import cv2
import os
import iisignature
datasets = 'mnist'
if datasets == 'mnist':
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    path = './mnist/training/'
    n_signatures = 10
    N_truncated = 3
    d = 28
    begin_validate = 2000
    end_validate = 2100

categories = len(labels)
folder = np.empty(categories, dtype='object')
for c in range(0, categories):
    folder[c] = path + labels[c] + '/'


# Compute signature.
def signature_cyz(folder, filename):
    image = cv2.imread(os.path.join(folder, filename))
    if image is not None:
        image = cv2.resize(image, (d, d))
        image = np.reshape(image, (image.shape[0], image.shape[1] * image.shape[2]))
        image = iisignature.sig(image, N_truncated)
        return image

# Compute a class representative for each category using (0:n_signatures) from train.
# e.g. In AFHQ we use 100 signatures per class, that is a total of 300 train samples.
supermeanA = np.empty(categories, dtype='object')
for c in range(0, categories):
    dataA= []
    a = os.listdir(folder[c])
    for filename in a[0:n_signatures]:
        dataA.append([signature_cyz(folder[c], filename), folder[c] + filename])

    featuresA, imagesA = zip(*dataA)
    supermeanA[c] = np.mean(featuresA, axis=0)


# Load validation instances from train (begin:end) and compute signatures to tune the weights.
# e.g. In AFHQ we use 500 signatures per class, that is a total of 1500 validation samples.

for c in range(0, categories):
    dataAA = []
    a = os.listdir(folder[c])
    for filename in a[begin_validate:end_validate]:
        dataAA.append([signature_cyz(folder[c], filename), folder[c] + filename])

    featuresAA, imagesAA = zip(*dataAA)

    # Estimate optimal \lambda_{*}
    # e.g. In AFHQ we solve the inverse problem lambda * supermeanA = featuresAA[z] z:0..500
    c_0 = supermeanA[c]
    c_0[c_0 == 0] = 1
    l = (1. / c_0) * featuresAA
    globals()['supermeanl_' + str(c)] = np.mean(l, axis=0)

if datasets == 'mnist':
    path = './mnist/testing/'

# Path where test instances can be found.
for c in range(0, categories):
    folder[c] = path + labels[c] + '/'

# Compute RMSE Signature and print accuracy. Load test instances inside the loop, compute signatures and evaluate.
# e.g. We use the full AFHQ validation set as test, that is a total of 1500 samples.
count = np.zeros(categories, dtype='object')

for c2 in range(0, categories):
    a = os.listdir(folder[c2])
    for z in range(0, len(a)):
        rmse_c = np.empty(categories, dtype='object')
        for c in range(0, categories):
            rmse_c[c] = mean_squared_error(globals()['supermeanl_' + str(c2)] * supermeanA[c], signature_cyz(folder[c2], a[z]), squared=False)
        min_rmse = np.argmin(rmse_c)
        if min_rmse != c2:
            count[c2] += 1

    print('RMSE ' + labels[c2])
    print('# of errors:', count[c2])
    print('Accuracy:', 1 - count[c2] / len(a))
    print('\n')