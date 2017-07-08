from sklearn import datasets

iris = datasets.load_iris()

# f(x) =y
#x = features
#y = labels
X = iris.data
y = iris.target

from sklearn.cross_validation import train_test_split

# test_size means an even split with the test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)


#decision tree
# from sklearn import tree
# my_classifier = tree.DecisionTreeClassifier()

#K Nearest Neighbors classifier
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

# i=0
# for prediction in predictions:
# 	if(prediction == y_test[i]):
# 		print prediction ,' == ', y_test[i]
# 	else:
# 		print prediction ,' != ', y_test[i]
# 	i = i+1

print predictions

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)

