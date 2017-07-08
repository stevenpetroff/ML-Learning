from sklearn import tree

#Training Data
# Grams    Texture    Fruit
# 140      smooth     pear
# 130      smooth     pear
# 150      fuzzy      kiwi
# 180      fuzzy      kiwi

# feature[i] = [grams, texture]
# texture values: 0 for smooth, 1 for fuzzy
features = [[140, 0], [130, 0], [150, 1], [180, 1]]

#label values: 1 for pear, 0 for kiwi
labels = [1, 1, 0, 0]

#initialize decision tree
classifier = tree.DecisionTreeClassifier()

#train our classifier with .fit() and passing in our training data and labels
classifier = classifier.fit(features, labels)

#use classifier to predict by passing in data it has not seen before
print classifier.predict([[160, 0]])
#this prints both [0] and [1] equally on subsequent executions.
#probably because training data is not sufficient