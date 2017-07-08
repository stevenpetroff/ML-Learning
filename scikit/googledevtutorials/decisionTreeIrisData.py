from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np


iris = load_iris()
test_index = [0,50,100]

# print iris.feature_names
# print iris.target_names
#print iris.data[0]
#print iris.target[0]

#get training data
#np.delete returns a new array containing values 
#from every array index except [0], [50] & [100]
train_target = np.delete(iris.target, test_index)
#this should populate train_data with iris[1-49, 51-99]
train_data = np.delete(iris.data, test_index, axis=0)


#lets get test data from indecies 0,50,100
test_target = iris.target[test_index]
#passing an array of indecies as an index returns an 
#an array of values from iris.target from those indecies! really cool
test_data = iris.data[test_index]

#lets setup our classifier and train it
classifier = tree.DecisionTreeClassifier()
#pass in our data(features) and target (labels)
classifier = classifier.fit(train_data, train_target)

#lets pass in our test data and see if it can predict!
print test_target
print classifier.predict(test_data)