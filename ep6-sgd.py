from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score


def display_image(x):
    plt.imshow(x.reshape(28, 28), cmap=plt.cm.binary, interpolation='nearest')
    plt.show()


def display_predict(clf, actually_y, x):
    print('Actually = ', actually_y)
    print('Prediction = ', clf.predict([x])[0])


mnist_raw = loadmat('D:\\Skill\\python\\kongsuksiam\\mnist-original.mat')
mnist = {
    "data": mnist_raw['data'].T,
    "target": mnist_raw['label'][0]
}
x, y = mnist['data'], mnist['target']
# print(mnist['data'].shape)
# print(mnist['target'].shape)

# training 80% , test 20%
# train_set : 0- 60000 , test_set: 60001 - 70000
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
'''print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)'''

predict_number = 100
y_train_5 = (y_train == 5)
#print(y_train_0.shape, y_train_0)
y_test_5 = (y_test == 5)
#print(y_test_0.shape, y_test_0)

sgd_clf = SGDClassifier()
sgd_clf.fit(x_train, y_train_5)

#display_predict(sgd_clf, y_test_0[predict_number], x_test[predict_number])
# display_image(x_test[predict_number])
score = cross_val_score(sgd_clf, x_train, y_train_5, cv=5, scoring='accuracy')
print(score)
