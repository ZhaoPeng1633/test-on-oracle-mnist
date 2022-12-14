from sklearn import svm
import numpy as np
from time import time
from sklearn.metrics import accuracy_score
from struct import unpack
from sklearn.model_selection import GridSearchCV


def readimage(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        img = np.fromfile(f, dtype=np.uint8).reshape(num, 784)
    return img


def readlabel(path):
    with open(path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        lab = np.fromfile(f, dtype=np.uint8)
    return lab


def main():
    train_data = readimage("train-images-idx3-ubyte")
    train_label = readlabel("train-labels-idx1-ubyte")
    test_data = readimage("test-images-idx3-ubyte")
    test_label = readlabel("test-labels-idx1-ubyte")
    print(test_label)

    svc = svm.SVC()
    parameters = {'kernel': ['rbf'], 'C': [10]}
    print("Train...")
    clf = GridSearchCV(svc, parameters, n_jobs=-1)
    start = time()
    clf.fit(train_data, train_label)
    end = time()
    t = end - start
    print('Train：%dmin%.3fsec' % (t // 60, t - 60 * (t // 60)))
    prediction = clf.predict(test_data)
    print("accuracy: ", accuracy_score(prediction, test_label))
    accurate = [0] * 10
    sumall = [0] * 10
    s = (10, 10)
    error = np.zeros(s, int)
    i = 0
    while i < len(test_label):
        sumall[test_label[i]] += 1
        if prediction[i] == test_label[i]:
            accurate[test_label[i]] += 1
        else:
            error[test_label[i], prediction[i]] += 1
        i += 1
    print("分类正确的：", accurate)
    print("总的测试标签：", sumall)
    print(error)


if __name__ == '__main__':
    main()