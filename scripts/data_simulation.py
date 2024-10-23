import time
import mysql.connector

# Function to simulate data inserts into the database
def simulate_data_inserts(connection):
    cursor = connection.cursor()
    
    # Example of inserting new customer data every few seconds
    for i in range(5):  # Simulate 5 new customers and transactions
        cursor.execute("""
            INSERT INTO customers (first_name, last_name, ssn, phone_number, address_line1, city, state, zip_code, account_creation_date, account_status, fraud_flag)
            VALUES ('John', 'Doe', '123-45-6789', '555-555-5555', '123 Main St', 'Cityville', 'State', '12345', NOW(), 'active', 'FALSE')
        """)
        connection.commit()
        print(f"Inserted customer {i+1}")
        
        cursor.execute("""
            INSERT INTO transactions (account_id, customer_id, transaction_type, transaction_amount, transaction_date, merchant_name, merchant_location, transaction_status, fraud_flag)
            VALUES (LAST_INSERT_ID(), LAST_INSERT_ID(), 'purchase', 100.50, NOW(), 'StoreX', 'LocationX', 'completed', 'FALSE')
        """)
        connection.commit()
        print(f"Inserted transaction {i+1}")
        
        time.sleep(5)  # Wait 5 seconds between each insert

# Connect to the MySQL database
def create_connection():
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='your_password',  # Use your actual password
        database='banking_data'     # Ensure the correct database is used
    )
    return connection

if __name__ == "__main__":
    connection = create_connection()
    if connection.is_connected():
        print("Connected to MySQL for simulation.")
        simulate_data_inserts(connection)
        connection.close()
        print("Simulation complete and connection closed.")
    else:
        print("Unable to connect to MySQL.")
