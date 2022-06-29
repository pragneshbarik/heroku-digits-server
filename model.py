from PIL import Image
import mnist
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import joblib

X_train = mnist.train_images()
y_train = mnist.train_labels()
X_test =  mnist.test_images()
y_test = mnist.test_labels()

X_train = X_train.reshape((-1, 28*28))
X_test=X_test.reshape((-1, 28*28))

X_train = (X_train/256)
X_test  = (X_test/256)

clf = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(100,100))
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
acc = confusion_matrix(y_test, predictions)
def accuracy(con_matrix) :
  diagonal = con_matrix.trace()
  elements = np.sum(np.sum(con_matrix))
  return diagonal/elements
print("Model Accuracy:", 100*accuracy(acc))

joblib.dump(clf, "digit_classifier.sav")

