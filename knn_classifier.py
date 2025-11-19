import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("ccdefault.csv")

# Separate features and target
y = df["DEFAULT"]
X = df.drop(["DEFAULT", "ID"], axis=1)

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalize the data
scaler = Normalizer()
x_train_norm = scaler.fit_transform(x_train)
x_test_norm = scaler.transform(x_test)

# Train the KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train_norm, y_train)

# Predict on the test set
y_pred = model.predict(x_test_norm)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
