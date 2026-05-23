##Importing required libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#Collecting the data and splitting it into test and train sets
data = fetch_20newsgroups()
data.target_names
train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')



##Classification using Multinomial Naive Bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(train.data, train.target)
pred_labels = model.predict(test.data)
mat = confusion_matrix(test.target, pred_labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.title('Multinomial Naive Bayes')
print("Accuracy: {:.2f}%".format(100 * accuracy_score(test.target, pred_labels)))



##Classification using Random Forest Classifier
model = make_pipeline(TfidfVectorizer(), RandomForestClassifier())
model.fit(train.data, train.target)
pred_labels = model.predict(test.data)
mat = confusion_matrix(test.target, pred_labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.title('Random Forest Classifier')
print("Accuracy: {:.2f}%".format(100 * accuracy_score(test.target, pred_labels)))



##Classification using KNN Classifier
model = make_pipeline(TfidfVectorizer(), KNeighborsClassifier(n_neighbors=3))
model.fit(train.data, train.target)
pred_labels = model.predict(test.data)
mat = confusion_matrix(test.target, pred_labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.title('KNN Classifier')
print("Accuracy: {:.2f}%".format(100 * accuracy_score(test.target, pred_labels)))


