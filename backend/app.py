from flask import Flask, jsonify, request
import mysql.connector
from mysql.connector import Error
from services.discrepancy_detection import check_for_discrepancies
from services.fraud_detection import predict_fraud
from services.llm_integration import (
    generate_sql_query_with_context,
    execute_sql_query,
    summarize_query_with_results,
    retrieve_context_for_query,
    get_schema_info,
    parse_sql_query,
    kg,
    engine,
    metadata,
    build_knowledge_graph,
    chat_openai
)
import datetime

app = Flask(__name__)


build_knowledge_graph(metadata, kg) 

transaction_type_mapping = {
    "Online Purchase": 0,
    "ATM Withdrawal": 1,
    "POS Payment": 2,
    "Transfer": 3
}

# Database connection setup
def create_connection():
    try:
        connection = mysql.connector.connect(
            host='',
            user='',
            password='',
            database=''
        )
        if connection.is_connected():
            print("Connected to MySQL database")
        return connection
    except Error as e:
        print(f"Error while connecting to MySQL: {e}")
        return None

# Close database connection
def close_connection(connection):
    if connection.is_connected():
        connection.close()
        print("MySQL connection closed")

# API route for getting discrepancies
@app.route('/api/discrepancies', methods=['GET'])
def get_discrepancies():
    connection = create_connection()
    if connection:
        discrepancies = check_for_discrepancies(connection)
        close_connection(connection)
        return jsonify(discrepancies), 200
    else:
        return jsonify({"error": "Unable to connect to the database"}), 500
    

@app.route('/api/fraud_detection', methods=['POST'])
def fraud_detection():
    data = request.get_json()  # Transaction data sent from frontend
    
    # Handle string inputs for transaction_type by converting to numeric (label encoding)
    transaction_type_numeric = transaction_type_mapping.get(data["transaction_type"], -1)
    
    # If transaction_type is not found in the mapping, return an error
    if transaction_type_numeric == -1:
        return jsonify({"error": "Invalid transaction type"}), 400

    # Prepare the formatted data
    formatted_data = {
        "account_id": int(data["account_id"]),
        "transaction_amount": float(data["transaction_amount"]),
        "transaction_type": transaction_type_numeric,  # Now it's numeric
        "transaction_date": str(data["transaction_date"]),  # Pass transaction_date as a string
        "merchant_location": str(data["merchant_location"]) # Include merchant location as string
    }
    
    connection = create_connection()
    if connection:
        result = predict_fraud(formatted_data, connection)  # Use the model to predict fraud and get discrepancy type
        close_connection(connection)
        return jsonify(result), 200
    else:
        return jsonify({"error": "Unable to connect to the database"}), 500


DB_HOST = ''
DB_USER = ''
DB_PASSWORD = ''
DB_NAME = ''

def convert_sets_to_lists(data):
    """
    Recursively convert all sets to lists within a dictionary or list.
    """
    if isinstance(data, dict):
        return {key: convert_sets_to_lists(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_sets_to_lists(item) for item in data]
    elif isinstance(data, set):
        return list(data)
    else:
        return data


# API for LLM-based queries
@app.route('/api/query', methods=['POST'])
def process_query():
    data = request.get_json()
    question = data.get('query')
    
    try:
        schema_info = get_schema_info(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)
        
        # Retrieve full context with datatypes and sample data
        relevant_tables = ['customers','accounts','contact_info','address','employment_info','fraud_investigation','transactions']  # Modify this to limit tables based on your logic
        context = retrieve_context_for_query(kg, engine, relevant_tables)
        
        # Generate SQL query using GPT-4
        sql_query = generate_sql_query_with_context(question, relevant_tables, chat_openai)
        
        if sql_query:
            print(f"Generated SQL Query: {sql_query}")
            
            # Execute the SQL query
            query_results = execute_sql_query(engine, sql_query)
            
            if isinstance(query_results, str):
                return jsonify({"error": query_results}), 500
            
            # Summarize results
            summary = summarize_query_with_results(question, query_results, chat_openai)
            
            # Parse the SQL query to extract used tables and columns
            used_elements = parse_sql_query(sql_query, schema_info)

            # Convert any set to list for JSON serialization
            if isinstance(used_elements.get("tables"), set):
                used_elements["tables"] = list(used_elements["tables"])
            if isinstance(used_elements.get("columns"), set):
                used_elements["columns"] = list(used_elements["columns"])

            # Prepare columns list
            columns = query_results[0].keys() if query_results else []  

            print(f"Columns: {list(columns)}")

            print("summary_type", type(summary))
            print("result_type", type(query_results))
            print("column_type", type(columns))
            print("used_elements_type", type(used_elements))
            print("schema_info_type", type(schema_info))

            # Recursively convert any set to list for JSON serialization
            used_elements = convert_sets_to_lists(used_elements)
            schema_info = convert_sets_to_lists(schema_info)

            print("used_elements_type", type(used_elements))
            print("schema_info_type", type(schema_info))

            return jsonify({
                "summary": summary,
                "result": query_results,
                "columns": list(columns),
                "used_elements": used_elements,
                "schema_info": schema_info
            }), 200
        
        else:
            return jsonify({"error": "Failed to generate SQL query"}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
