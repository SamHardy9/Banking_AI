import joblib
import numpy as np
from datetime import datetime

# Load the pre-trained fraud model
model = joblib.load('./models/fraud_model.pkl')

discrepancy_test = joblib.load('./models/discrepancy_test.pkl')

def predict_fraud(transaction_data, connection):
    """
    Predict fraud based on transaction data and return the fraud detection result
    along with the discrepancy type.
    """
    
    # Handle merchant_location by hashing the string to get a numeric value
    merchant_location_numeric = hash(transaction_data['merchant_location']) % 10**6  # Simple hash to a 6-digit number
    
    # Convert the transaction_date to a numerical format (e.g., timestamp or ordinal date)
    try:
        transaction_date = datetime.strptime(transaction_data['transaction_date'], "%Y-%m-%d")
        transaction_date_numeric = transaction_date.toordinal()  # Convert to an ordinal number
    except ValueError:
        return {"error": "Invalid transaction date format. Expected YYYY-MM-DD."}
    
    # Extract all required features from transaction_data
    features = np.array([
        transaction_data['account_id'],          # account_id
        transaction_data['transaction_amount'],  # transaction_amount
        merchant_location_numeric,               # Hashed merchant_location
        transaction_data['transaction_type'],    # transaction_type (already numeric)
        transaction_date_numeric                 # transaction_date as a numeric value
    ]).reshape(1, -1)  # Reshape to match the model's input format
    
    # Get fraud prediction and discrepancy type from the model
    prediction = model.predict(features)[0]  # True = Fraud, False = Not Fraud
    
    # Get the discrepancy type associated with the prediction from the model
    discrepancy_type = discrepancy_test.iloc[0]  # Hypothetical function to get discrepancy type
    
    return {
        'prediction': bool(prediction),  # True = Fraud, False = Not Fraud
        'discrepancy_type': discrepancy_type  # Return discrepancy type
    }
