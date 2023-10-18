import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('IRIS.csv')


le = LabelEncoder()
data['species'] = le.fit_transform(data['species'])

X = data.drop('species', axis=1)
y = data['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Model = RandomForestClassifier(n_estimators=100, random_state=42)

Model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = Model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)


print(f"Accuracy: {accuracy:.2f}")

# Create a new DataFrame with the Iris flower data
new_data = pd.DataFrame({
    'sepal_length': [5.6],  
    'sepal_width': [3.5],  
    'petal_length': [1.29], 
    'petal_width': [0.95]   
})

predicted_species = Model.predict(new_data)

# Inverse transform to get the actual species name
predicted_species_name = le.inverse_transform(predicted_species)

print("Predicted Species:", predicted_species_name)
