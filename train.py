import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib # Used to "Freeze" the model

# 1. Import Dataset
df = pd.read_csv('data.csv')

# 2. Preparing Training and Testing data
X = df[['sqft']] # Features
y = df['is_expensive'] # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train a Simple Model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Evaluate the Model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100}%")

# 5. Freeze (Save) the model
# This saves the "brain" of your app to a file called 'house_model.pkl'
joblib.dump(model, 'house_model.pkl')
print("Model frozen and saved as house_model.pkl")