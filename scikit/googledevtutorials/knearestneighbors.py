from scipy.spatial import distance

#KNN takeaway
#pros: easy to understand
#cons: slow, hard to represent relationships between features
def euc(a, b):
	return distance.euclidean(a,b)

class GoodEnoughKNN():
	def fit (self, X_train, y_train):
		self.X_train = X_train
		self.y_train = y_train

	def predict(self, X_test):
		predictions = []
		for row in X_test:
			label = self.closest(row)
			predictions.append(label)
		return predictions

	#loop through all of our X_train data to return the label that is closest to our row point
	def closest(self, row):
		best_dist = euc(row, self.X_train[0])
		best_index = 0
		for i in range(1, len(self.X_train)):
			new_dist = euc(row, self.X_train[i])
			if  new_dist< best_dist:
				best_dist = new_dist
				best_index = i
		return self.y_train[best_index]


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
# from sklearn.neighbors import KNeighborsClassifier
my_classifier = GoodEnoughKNN()

my_classifier.fit(X_train, y_train)

predictions = my_classifier.predict(X_test)

# i=0
# for prediction in predictions:
# 	if(prediction == y_test[i]):
# 		print prediction ,' == ', y_test[i]
# 	else:
# 		print prediction ,' != ', y_test[i]
# 	i = i+1

# print predictions

from sklearn.metrics import accuracy_score
print accuracy_score(y_test, predictions)

