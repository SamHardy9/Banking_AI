import streamlit as st
from sqlalchemy import create_engine, MetaData, text
from langchain_openai import ChatOpenAI
import networkx as nx

# Replace with your actual MySQL credentials
DATABASE_URL = "mysql+pymysql://root:password@localhost:3306/dbname"
engine = create_engine(DATABASE_URL)
metadata = MetaData()
metadata.reflect(bind=engine)


DB_HOST = ''
DB_USER = ''
DB_PASSWORD = ''
DB_NAME = ''


# Initialize the GPT-4 model using ChatOpenAI from LangChain
chat_openai = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key='')

# Initialize the NetworkX graph
kg = nx.DiGraph()



# SQL Query Execution Function
def execute_sql_query(engine, query):
    """
    Execute the generated SQL query and return the results.
    """
    try:
        with engine.connect() as connection:
            result = connection.execute(text(query))
            # Use row._mapping to ensure compatibility
            result_set = [dict(row._mapping) for row in result]  # Convert the result to a list of dictionaries
        return result_set
    except Exception as e:
        print("1",e)
        return str(e)  # Return the error if any occurs


# Function to get sample data from a table
def get_sample_data(engine, table_name, limit=5):
    """
    Retrieve sample data from a given table.
    """
    query = text(f"SELECT * FROM {table_name} LIMIT {limit}")
    with engine.connect() as connection:
        result = connection.execute(query)
        # Use row._mapping to ensure compatibility
        sample_data = [dict(row._mapping) for row in result]
    
    return sample_data


sql_keywords = {'SUM', 'AS', 'SELECT', 'FROM', 'WHERE', 'GROUP', 'BY', 'ORDER', 'JOIN', 'TIMESTAMPDIFF', 'ON', 'LIMIT', 'DESC', 'ASC', 'COUNT', 'AVG', 'MAX', 'MIN', 'WITH', 'DISTINCT', 'STR_TO_DATE', 'WHEN', 'CROSS', 'GROUP BY', 'CASE'}

import re
# Function to parse SQL query
def parse_sql_query(sql_query, schema_info):
    used_elements = {
        "schemas": {"mysql"},
        "databases": {"banking_data"},
        "tables": set(),
        "columns": set()
    }

    # Normalize the query: remove newlines and extra spaces
    sql_query = ' '.join(sql_query.split())

    # Extract tables from FROM, JOIN, and other clauses
    table_pattern = r'(?:FROM|JOIN|CROSS JOIN)\s+`?(\w+)`?'
    tables = re.findall(table_pattern, sql_query, re.IGNORECASE)

    # Validate the extracted tables against schema_info
    valid_tables = set()
    for table in tables:
        if table in schema_info:
            valid_tables.add(table)
    
    used_elements["tables"].update(valid_tables)

    # Extract columns from the SELECT clause
    column_pattern = r'`(\w+)`|(?<=\.)`?(\w+)`?|\b(\w+)\b'
    potential_columns = re.findall(column_pattern, sql_query)

    for match in potential_columns:
        column = next((col for col in match if col), None)
        if column and column.lower() not in sql_keywords:
            for table in valid_tables:
                if column in schema_info.get(table, []):
                    used_elements["columns"].add(column)

    # Convert sets to lists for JSON serialization
    used_elements["tables"] = list(used_elements["tables"])
    used_elements["columns"] = list(used_elements["columns"])

    print(used_elements)

    return used_elements

import plotly.express as px
import plotly.io as pio

def plot_data_visualization(df, chart_type):
    chart_type = chart_type.lower()
    fig = None

    # Convert all columns to string type except the numeric ones to preserve full numbers
    for col in df.columns:
        if df[col].dtype == 'int64' or df[col].dtype == 'float64':
            df[col] = df[col].apply(lambda x: '{:.0f}'.format(x))  # Format to avoid scientific notation

    # Let the user select the x and y axes from the dataframe columns
    x_axis = st.selectbox('Select X-axis', df.columns)
    y_axis = None
    if chart_type in ["line", "bar", "scatter"]:
        y_axis = st.selectbox('Select Y-axis', df.columns)

    if chart_type == "line":
        fig = px.line(df, x=x_axis, y=y_axis)
    elif chart_type == "bar":
        fig = px.bar(df, x=x_axis, y=y_axis)
    elif chart_type == "pie":
        names = st.selectbox('Select "Names" for Pie Chart', df.columns)
        values = st.selectbox('Select "Values" for Pie Chart', df.columns)
        fig = px.pie(df, names=names, values=values)
    elif chart_type == "scatter":
        fig = px.scatter(df, x=x_axis, y=y_axis)

    if fig:
        # Disable tick abbreviations for both x and y axes
        fig.update_layout(
            xaxis_tickformat="none",
            yaxis_tickformat="none",
            xaxis=dict(tickmode='linear', tick0=0),
            yaxis=dict(tickmode='linear', tick0=0)
        )
        st.plotly_chart(fig)
    else:
        st.error("Unable to create the selected chart type with the given data.")



def get_schema_info(host, user, password, database):
    import mysql.connector
    conn = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database
    )
    
    cursor = conn.cursor()
    query = """
    SELECT TABLE_NAME, COLUMN_NAME 
    FROM INFORMATION_SCHEMA.COLUMNS 
    WHERE TABLE_SCHEMA = %s;
    """
    
    cursor.execute(query, (database,))
    results = cursor.fetchall()
    
    schema_info = {}
    for table, column in results:
        if table not in schema_info:
            schema_info[table] = []
        schema_info[table].append(column)
    
    cursor.close()
    conn.close()
    
    return schema_info

# Function to retrieve full context with sample data
def retrieve_context_for_query(graph, engine, relevant_tables):
    """
    Retrieve relevant schema information for the tables involved in the query,
    including columns, data types, relationships, and sample data.
    """
    context = "Relevant Database schema with column data types and sample data:\n"
    
    for table_name in relevant_tables:
        # Get the table data from the graph
        table_data = graph.nodes.get(table_name, {})
        context += f"Table: {table_name}\n"
        
        # Include columns and their data types
        columns = table_data.get('columns', [])
        for column in columns:
            column_type = table_data.get(f'col_{column}', {}).get('type', 'Unknown')
            context += f"  - Column: {column}, Type: {column_type}\n"
        
        # Add relationships to other tables (if any)
        related_tables = list(graph.successors(table_name))
        if related_tables:
            context += f"  Related tables: {', '.join(related_tables)}\n"
        
        # Include sample data (1-2 rows) for the table
        sample_data = get_sample_data(engine, table_name, limit=2)
        if sample_data:
            context += "  Sample Data:\n"
            for row in sample_data:
                context += f"    {row}\n"
    
    return context




from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# Initialize OpenAI embeddings using your OpenAI API key
embeddings = OpenAIEmbeddings(openai_api_key='')



import ast
# Function to generate SQL query with conversation history and example queries for better understanding
def generate_sql_query_with_context(question, relevant_tables, chat_model):
    """
    Generate an optimized SQL query without using conversation history.
    """
    # Retrieve relevant schema context (limited to the query's relevant tables)
    context = retrieve_context_for_query(kg, engine, relevant_tables)

    # Update the prompt template to focus only on the current user query
    prompt = f"""
    You are an expert MySQL SQL query generator. Based on the user input, generate a comprehensive and efficient SQL query that addresses the request fully. Consider using advanced SQL features such as JOINs, subqueries, window functions, and aggregations where appropriate.

    **Example Queries:**
    1. "Show me accounts with duplicate SSNs."
    - SELECT ssn, COUNT(*) FROM customers GROUP BY ssn HAVING COUNT(*) > 1;

    2. "Find the total number of transactions made by each customer, and their average transaction amount."
    - SELECT customer_id, COUNT(transaction_id) AS total_transactions, AVG(transaction_amount) AS avg_transaction_amount FROM transactions GROUP BY customer_id;

    3. "Retrieve the top 5 customers who have made the highest total transaction amounts in foreign currencies."
    - SELECT t.customer_id, c.first_name, c.last_name, SUM(t.transaction_amount) AS total_foreign_amount FROM transactions t JOIN customers c ON t.customer_id = c.customer_id WHERE t.currency != 'USD' GROUP BY t.customer_id, c.first_name, c.last_name ORDER BY total_foreign_amount DESC LIMIT 5;

    User Input: {question}

    Context of the Database (schemas, columns, relationships, and sample data):
    {context}

    Guidelines:
    1. Use appropriate JOINs to connect related tables.
    2. Utilize subqueries or CTEs (Common Table Expressions) for complex logic.
    3. Apply aggregations (COUNT, SUM, AVG, etc.) when dealing with groups of data.
    4. Use WHERE clauses to filter data effectively.
    5. Consider using window functions for advanced analytics if needed.
    6. Ensure the query is optimized for performance.

    Return only the SQL query without any explanations. The query should be ready to execute in a MySQL environment.
    """

    # Generate the SQL query using the GPT-4 model
    response = chat_model.predict(prompt)
    
    # Log and return the generated query
    print(f"Generated SQL Query: {response}")
    return response.strip()




# Function to summarize SQL query results using GPT-4
def summarize_query_with_results(question, query_results, chat_model):
    """
    Generate a natural language summary based on the user's question and the SQL query results using GPT-4.
    """
    if not query_results:
        return "No results found."
    
    # Format the result into a string
    formatted_results = "Query Results:\n"
    for i, row in enumerate(query_results):
        formatted_results += f"Row {i + 1}: {row}\n"
    
    # Create a prompt that includes the user's question and the query results
    prompt = f"""
    Based on the following user question and SQL query results, provide a concise, general natural language summary without mentioning specific names just the count of the customers. 
    Summarize key insights such as the total number of records, aggregate metrics (e.g., average, total), 
    or notable patterns :

    User Question: {question}
    
    {formatted_results}
    
    Natural Language Summary:
    """
    
    # Generate the summary using GPT-4
    response = chat_model.predict(prompt)
    
    return response


# When creating the knowledge graph
for table_name, table in metadata.tables.items():
    # Add each table as a node
    kg.add_node(table_name, label='table', columns=list(table.columns.keys()))

    # Add column metadata, including data type
    for column in table.columns:
        kg.nodes[table_name][f'col_{column.name}'] = {
            'type': str(column.type),  # Include column data type
            'primary_key': column.primary_key,
            'nullable': column.nullable
        }

    # Add edges based on foreign key relationships
    for column in table.columns:
        if column.foreign_keys:
            for fk in column.foreign_keys:
                related_table = fk.column.table.name
                kg.add_edge(related_table, table_name, relationship='foreign_key')


def build_knowledge_graph(metadata, kg):
    print("Building the knowledge graph...")

    for table_name, table in metadata.tables.items():
        print(f"Adding table: {table_name}")  # This will help identify if 'transactions' is missing

        # Add each table as a node in the graph
        kg.add_node(table_name, label='table', columns=list(table.columns.keys()))

        # Add foreign key relationships as edges between tables
        for column in table.columns:
            if column.foreign_keys:
                for fk in column.foreign_keys:
                    related_table = fk.column.table.name
                    print(f"Adding relationship from {table_name} to {related_table}")
                    kg.add_edge(related_table, table_name, relationship='foreign_key')

    print("Knowledge graph built successfully.")

# Build the knowledge graph and check for 'transactions'
build_knowledge_graph(metadata, kg)

