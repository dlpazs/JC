from sklearn.ensemble import RandomForestClassifier
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, neighbors, svm

df = pd.read_csv()#
x = df.drop(['job_title', 'description'], 1)#drop the Y output
y = df['job_title'] # Y output label
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

clf = svm.SVC()
clf.fit(x_train, y_train)
accuracy_svm = clf.score(x_test, y_test)

forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(x_train, y_train)

accuracy = forest_clf.score(x_test, y_test)
example = [[0.2,0.5,0.1,0.7,0.2,0.8,0.9,0.4]]

prediction = clf.predict(example)
print(prediction)
print(example)
print(accuracy_svm)
