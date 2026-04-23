import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
data = pd.read_csv('IRIS.csv')
print(data.head())
x = data.drop('species', axis=1)
y = data['species']
le = LabelEncoder()
y =le.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
                      columns=x.columns)
prediction = model.predict(sample)
print("\nPredicted Species:", le.inverse_transform(prediction))
