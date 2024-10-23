import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
import datetime
import sys
import os

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (project root)
parent_dir = os.path.dirname(current_dir)

# Add the backend directory to the Python path
backend_dir = os.path.join(parent_dir, 'backend')
sys.path.append(backend_dir)

# Now you can import from services.llm_integration
from services.llm_integration import get_schema_info

# Your fapp.py code continues here...


# Backend URLs
DISCREPANCIES_API_URL = "http://localhost:5000/api/discrepancies"
FRAUD_DETECTION_API_URL = "http://localhost:5000/api/fraud_detection"
LLM_QUERY_API_URL = "http://localhost:5000/api/query"


DB_HOST = ''
DB_USER = ''
DB_PASSWORD = ''
DB_NAME = ''

# Navigation Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "View Discrepancies", "Fraud Detection", "Ask a Question (LLM)"])

# Define different pages
if page == "Home":
    st.title("Welcome to the Banking Data Monitoring Dashboard")
    st.write("This is your central dashboard for detecting discrepancies, predicting fraud, and interacting with AI.")

elif page == "View Discrepancies":
    st.title("View Discrepancies")
    
    # Define buttons for each type of discrepancy
    discrepancy_types = [
        "Duplicate SSN", "Duplicate Contact Information", "Transactions on Inactive Accounts", 
        "Multiple Accounts with Same Address", "Suspiciously High Transactions", 
        "Mismatched Customer and Account Data", "Multiple Location Transactions", 
        "Unusual Balance Changes", "Inconsistent KYC with High Transaction Volumes", 
        "Multiple Failed Transactions Followed by Success"
    ]

    # Display buttons for each discrepancy type and handle button clicks
    for discrepancy in discrepancy_types:
        if st.button(discrepancy):
            response = requests.get(DISCREPANCIES_API_URL).json()
            
            # Filter the response based on the discrepancy type
            filtered_discrepancies = [d for d in response if d['type'] == discrepancy]
            
            if filtered_discrepancies:
                st.write(f"Discrepancies for {discrepancy}:")
                
                # Convert the list of dictionaries to a pandas DataFrame for tabular display
                df = pd.DataFrame(filtered_discrepancies[0]['details'])
                st.dataframe(df)
            else:
                st.write(f"No {discrepancy} found.")

elif page == "Fraud Detection":
    st.title("Fraud Detection")
    # Account ID (int)
    account_id = st.number_input("Account ID", min_value=0, value=0, step=1)

    # Transaction Amount (float)
    transaction_amount = st.number_input("Transaction Amount", min_value=0.0, value=0.0, step=0.01)

    # Transaction Type (categorical - string)
    transaction_type = st.selectbox(
        "Transaction Type", 
        ["Online Purchase", "ATM Withdrawal", "POS Payment", "Transfer"]
    )

    # Transaction Date (string or date)
    transaction_date = st.date_input("Transaction Date")

    # Merchant Location (string)
    merchant_location = st.text_input("Merchant Location")

    # When the user clicks the "Submit" button, send the data to the backend for fraud prediction
    if st.button("Submit"):
        # Prepare the data for the backend
        data = {
            "account_id": int(account_id),
            "transaction_amount": float(transaction_amount),
            "transaction_type": transaction_type,
            "transaction_date": str(transaction_date),  # Convert date to string format
            "merchant_location": merchant_location      # Include merchant location
        }

        # Send data to backend API
        response = requests.post(FRAUD_DETECTION_API_URL, json=data)

        if response.status_code == 200:
            result = response.json()
            fraud_prediction = result.get("prediction")
            discrepancy_type = result.get("discrepancy_type")  # Get discrepancy type from backend
            
            if fraud_prediction:
                st.error(f"Fraudulent Transaction Detected! Discrepancy Type: {discrepancy_type}")
            else:
                st.success("Transaction is Legitimate. Discrepancy Type: {discrepancy_type}")
        else:
            st.error("Error: Could not process the request.")


elif page == "Case Study Simulation":
    st.title("Simulate a Fraud Case Study")
    account_id = st.number_input("Account ID", min_value=1)
    transaction_amount = st.number_input("Transaction Amount", min_value=0.0)
    transaction_type = st.selectbox("Transaction Type", ["deposit", "withdrawal", "purchase"])
    if st.button("Simulate"):
        data = {
            "account_id": account_id,
            "transaction_amount": transaction_amount,
            "transaction_type": transaction_type
        }
        response = requests.post(FRAUD_DETECTION_API_URL, json=data).json()
        st.write("Fraud Prediction:", response['prediction'])
        st.write("SHAP Explanation:", response['explanation'])


elif page == "Ask a Question (LLM)":
    # Create Tabs for the sections
    tabs = st.tabs(["Data Insights", "Data Visualization", "Data Lineage"])
    with tabs[0]:
        st.title("Data Insights")
        user_query = st.text_input("Enter your question")
        if st.button("Submit Query"):
            if user_query:
                response = requests.post(LLM_QUERY_API_URL, json={"query": user_query})
                if response.status_code == 200:
                    result = response.json()
                    st.write(f"Summary: {result['summary']}")
                    st.session_state['result'] = result['result']
                    st.session_state['columns'] = result['columns']
                    st.session_state['used_elements'] = result['used_elements']
                    st.session_state['schema_info'] = result['schema_info']
                    df = pd.DataFrame(result['result'], columns=result['columns'])
                    st.table(df)
                else:
                    st.error("Failed to get a response from the server.")
            else:
                st.warning("Please enter a question.")

    with tabs[1]:
        st.title("Data Visualization")
        if 'result' in st.session_state and 'columns' in st.session_state:
            df = pd.DataFrame(st.session_state['result'], columns=st.session_state['columns'])
            chart_type = st.selectbox("Select Chart Type", ["line", "bar", "pie", "scatter"])
            if chart_type == "line":
                x_axis = st.selectbox('Select X-axis', df.columns)
                y_axis = st.selectbox('Select Y-axis', df.columns)
                fig = px.line(df, x=x_axis, y=y_axis)
                st.plotly_chart(fig)
            elif chart_type == "bar":
                x_axis = st.selectbox('Select X-axis', df.columns)
                y_axis = st.selectbox('Select Y-axis', df.columns)
                fig = px.bar(df, x=x_axis, y=y_axis)
                st.plotly_chart(fig)
            elif chart_type == "pie":
                names = st.selectbox('Select "Names" for Pie Chart', df.columns)
                values = st.selectbox('Select "Values" for Pie Chart', df.columns)
                fig = px.pie(df, names=names, values=values)
                st.plotly_chart(fig)
            elif chart_type == "scatter":
                x_axis = st.selectbox('Select X-axis', df.columns)
                y_axis = st.selectbox('Select Y-axis', df.columns)
                fig = px.scatter(df, x=x_axis, y=y_axis)
                st.plotly_chart(fig)
        else:
            st.warning("No query results available for visualization.")

    with tabs[2]:  # Data Lineage Tab
        st.subheader("Data Lineage Graph")

                # Ensure schema_info is loaded
        if 'schema_info' not in st.session_state:
            st.session_state['schema_info'] = get_schema_info(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)

        schema_info = st.session_state['schema_info']

        if 'used_elements' in st.session_state:
            used_elements = st.session_state['used_elements']
            print(used_elements)
            G = nx.DiGraph()

            # Add table nodes and edges between table and columns
            for table in used_elements["tables"]:
                G.add_node(table, label=table, type='table')

                # Add column nodes and edges only between table and column
                for column in used_elements["columns"]:
                    if column in schema_info.get(table, []):
                        G.add_node(column, label=column, type='column')
                        G.add_edge(table, column)  # Edge only from table to column

            pos = {}

            # Set position for tables, centered horizontally
            pos_y = 0
            table_pos = {}
            pos_x = -len(used_elements["tables"]) * 2

            for node, data in G.nodes(data=True):
                if data['type'] == 'table':
                    pos[node] = (pos_x, pos_y)
                    table_pos[node] = pos_x
                    pos_x += 5  # Spread tables horizontally

            # Set position for columns horizontally under each table
            pos_y -= 2  # Move down the y-axis for columns
            for node, data in G.nodes(data=True):
                if data['type'] == 'column':
                    table = list(G.predecessors(node))[0]
                    pos[node] = (table_pos[table], pos_y)
                    table_pos[table] += 2

            plt.figure(figsize=(12, 6))

            # Draw the edges
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v) for u, v in G.edges()],
                edge_color='lightcoral', arrows=False, width=1.0, alpha=0.6
            )

            # Draw the labels
            for node, (x, y) in pos.items():
                if G.nodes[node]['type'] == 'table':
                    plt.text(x, y, G.nodes[node]['label'], ha="center", va="center",
                            fontsize=12, fontweight="bold")
                    plt.text(x, y + 0.1, "Table", ha="center", va="center",
                            fontsize=12, fontweight="bold")
                elif G.nodes[node]['type'] == 'column':
                    plt.text(x, y, G.nodes[node]['label'], ha="center", va="center",
                            fontsize=12, fontweight="bold")
                    plt.text(x, y + 0.1, "Column", ha="center", va="center",
                            fontsize=12, fontweight="bold")

            # Add title and adjust layout
            plt.title("Database Lineage Flow", fontsize=14, fontweight='bold')
            plt.axis('off')
            plt.tight_layout()

            # Render the plot in Streamlit
            st.pyplot(plt)
        else:
            st.info("Submit a query in the 'Query & Result' tab to view the data lineage graph.")
