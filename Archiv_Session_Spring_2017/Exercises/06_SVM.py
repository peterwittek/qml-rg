"""
Created on Sun Mar 26 14:13:45 2017

@author: phuem
"""

#http://scikit-learn.org/stable/modules/svm.html

from sklearn import svm, metrics
import tools
import numpy as np
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
#Load and reshape Images with tools.py
training_set, training_labels= tools.load_images("images/train/")
test_set, test_labels = tools.load_images("images/test/")

resize_set, resize_labels= tools.prep_datas(training_set, training_labels)
resize_test_set, resize_test_labels= tools.prep_datas(test_set, test_labels)

#---------------------------------------------------------------
#very ugly way to bring vectors to the right shape for SVC fit()
a = []
for x in resize_set:
    a.append(x.tolist())
#----------------------------------------------------------------
X = a               #reshaped images (training)
y = resize_labels   #labels

clf = svm.SVC(gamma=1.0)  #load SVC
clf.fit(X, y)               #fit SVC

#-------------------------------------------------------------------
#very ugly way to bring vectors to the right shape for SVC predict()
a = []
for x in resize_test_set:
    a.append(x.tolist())
#-------------------------------------------------------------------    

predicted   = clf.predict(a)        #predict labels for new data
expected    = resize_test_labels    #this are the real labels that we expect

for index in range(0,len(resize_labels)): #Print Training Data
    plt.subplot(5, 5, index + 1)
    plt.axis('off')
    plt.imshow(training_set[index], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % resize_labels[index])
plt.show()

for index in range(0,len(resize_test_labels)): #Print Test Result
    plt.subplot(3, 2, index + 1)
    plt.axis('off')
    plt.imshow(test_set[index], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Predicted_Label: %i' % predicted[index])
plt.show()
#-----------------------------------------------------------------------------
#Print Results
print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
#See wiki article about confusion matrix. If Matrix is diagonal, prediction worked well!

