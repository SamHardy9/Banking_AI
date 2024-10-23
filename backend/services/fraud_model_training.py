# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib 
import pickle

# Load the dataset
file_path = 'Balanced_ML_Training_Dataset_for_Fraud_Detection.csv'
data = pd.read_csv(file_path)

# Drop columns with too many missing values (more than 50% missing)
data_reduced = data.drop(columns=['contact_type', 'contact_value', 'ssn', 'address_line1', 'city', 'zip_code', 'account_status'])

# Impute missing values in account_id using the median
data_reduced['account_id'].fillna(data_reduced['account_id'].median(), inplace=True)

# Select features for training
X_reduced = data_reduced[['customer_id', 'account_id', 'transaction_amount', 'currency']]
y_reduced = data_reduced['outcome']

# Add discrepancy_type to the dataset
discrepancy_type = data_reduced['discrepancy_type']

# One-hot encode the 'currency' column
X_encoded_reduced = pd.get_dummies(X_reduced, columns=['currency'], drop_first=True)

# Combine X_encoded_reduced, y_reduced, and discrepancy_type into one DataFrame for splitting
combined_data = pd.concat([X_encoded_reduced, y_reduced, discrepancy_type], axis=1)

# Split the combined data into train, validation, and test sets (60% train, 20% validation, 20% test)
train_data, temp_data = train_test_split(combined_data, test_size=0.4, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Separate X, y, and discrepancy_type from the split datasets
X_train = train_data.drop(columns=['outcome', 'discrepancy_type'])
y_train = train_data['outcome']
X_val = val_data.drop(columns=['outcome', 'discrepancy_type'])
y_val = val_data['outcome']
X_test = test_data.drop(columns=['outcome', 'discrepancy_type'])
y_test = test_data['outcome']

# Also, get the discrepancy_type for validation and test data
discrepancy_test = test_data['discrepancy_type']

joblib.dump(discrepancy_test, '..\models\discrepancy_test.pkl')

# Set up hyperparameter grid for RandomForest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the model with GridSearchCV using the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters from the grid search
best_rf_model = grid_search.best_estimator_

# Predict on the test set
y_test_pred = best_rf_model.predict(X_test)

# Evaluate the model on the test set
accuracy_test = accuracy_score(y_test, y_test_pred)
classification_rep_test = classification_report(y_test, y_test_pred)
#print(f"\nTest Set Performance:\nAccuracy: {accuracy_test}")
#print(classification_rep_test)

# Print discrepancy_type and prediction for each test data point
#print("\nPredictions and Corresponding Discrepancy Types on Test Data:")
#for i in range(len(X_test)):
#    print(f"Test Data Point {i + 1}:")
#    print(f"Discrepancy Type: {discrepancy_test.iloc[i]}")
#    print(f"Predicted Outcome: {y_test_pred[i]}")
#    print("")

import os
# Specify the path where you want to save the model
model_directory = '..\models'
model_filename = 'fraud_model.pkl'
model_filepath = os.path.join(model_directory, model_filename)

# Check if the directory exists, and if not, create it
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Save the trained model to the specified file
with open(model_filepath, 'wb') as file:
    pickle.dump(best_rf_model, file)

print(f"Model saved as {model_filepath}")


