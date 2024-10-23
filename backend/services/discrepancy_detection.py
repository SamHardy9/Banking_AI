def check_for_discrepancies(connection):
    cursor = connection.cursor(dictionary=True)
    discrepancies = []

    # Case 1: Duplicate SSNs
    cursor.execute("""
        SELECT ssn, COUNT(*) AS occurrences
        FROM customers
        GROUP BY ssn
        HAVING COUNT(*) > 1;
    """)
    duplicate_ssns = cursor.fetchall()
    if duplicate_ssns:
        discrepancies.append({
            "type": "Duplicate SSN",
            "details": duplicate_ssns
        })

    # Case 2: Duplicate Contact Information (Phone Numbers or Emails)
    cursor.execute("""
        SELECT contact_value, COUNT(*) AS occurrences
        FROM contact_info
        GROUP BY contact_value
        HAVING COUNT(*) > 1;
    """)
    duplicate_contacts = cursor.fetchall()
    if duplicate_contacts:
        discrepancies.append({
            "type": "Duplicate Contact Information",
            "details": duplicate_contacts
        })

    # Case 3: Transactions on Inactive or Dormant Accounts
    cursor.execute("""
        SELECT t.transaction_id, t.account_id, a.account_status, t.transaction_amount, t.transaction_date
        FROM transactions t
        JOIN accounts a ON t.account_id = a.account_id
        WHERE a.account_status IN ('inactive', 'dormant');
    """)
    inactive_transactions = cursor.fetchall()
    if inactive_transactions:
        discrepancies.append({
            "type": "Transactions on Inactive/Dormant Accounts",
            "details": inactive_transactions
        })

    # Case 4: Multiple Accounts Using the Same Address
    cursor.execute("""
        SELECT address_line1, city, zip_code, COUNT(DISTINCT customer_id) AS customer_count
        FROM address
        GROUP BY address_line1, city, zip_code
        HAVING COUNT(DISTINCT customer_id) > 1;
    """)
    duplicate_addresses = cursor.fetchall()
    if duplicate_addresses:
        discrepancies.append({
            "type": "Multiple Accounts with the Same Address",
            "details": duplicate_addresses
        })

    # Case 5: Suspiciously High Transaction Amounts
    cursor.execute("""
        SELECT customer_id, transaction_id, transaction_amount, transaction_date
        FROM transactions
        WHERE transaction_amount > 10000;
    """)
    high_transactions = cursor.fetchall()
    if high_transactions:
        discrepancies.append({
            "type": "Suspiciously High Transaction Amounts",
            "details": high_transactions
        })

    # Case 6: Mismatched Customer and Account Data
    cursor.execute("""
        SELECT a.customer_id, c.first_name, c.last_name, a.account_id
        FROM accounts a
        LEFT JOIN customers c ON a.customer_id = c.customer_id
        WHERE c.customer_id IS NULL;
    """)
    mismatched_data = cursor.fetchall()
    if mismatched_data:
        discrepancies.append({
            "type": "Mismatched Customer and Account Data",
            "details": mismatched_data
        })

    # Case 7: Transactions from Multiple Locations in a Short Time Frame
    cursor.execute("""
        SELECT t.customer_id, t.transaction_id, t.merchant_location, t.transaction_date
        FROM transactions t
        JOIN (
            SELECT customer_id, MIN(transaction_date) AS first_transaction, MAX(transaction_date) AS last_transaction
            FROM transactions
            GROUP BY customer_id
            HAVING TIMESTAMPDIFF(MINUTE, MIN(transaction_date), MAX(transaction_date)) < 10
        ) AS suspicious_transactions ON t.customer_id = suspicious_transactions.customer_id;
    """)
    multiple_location_transactions = cursor.fetchall()
    if multiple_location_transactions:
        discrepancies.append({
            "type": "Transactions from Multiple Locations in Short Time",
            "details": multiple_location_transactions
        })

    # Case 8: Accounts with Unusual Balance Changes
    cursor.execute("""
        SELECT a.account_id, a.balance, a.customer_id, a.date_opened
        FROM accounts a
        WHERE a.balance > 50000
        ORDER BY a.balance DESC;
    """)
    unusual_balance_changes = cursor.fetchall()
    if unusual_balance_changes:
        discrepancies.append({
            "type": "Unusual Balance Changes",
            "details": unusual_balance_changes
        })

    # Case 9: Inconsistent KYC Status with High Transaction Volumes
    cursor.execute("""
        SELECT c.customer_id, c.first_name, c.last_name, t.transaction_amount, c.kyc_verification
        FROM customers c
        JOIN transactions t ON c.customer_id = t.customer_id
        WHERE c.kyc_verification = 0
        AND t.transaction_amount > 5000;
    """)
    inconsistent_kyc = cursor.fetchall()
    if inconsistent_kyc:
        discrepancies.append({
            "type": "Inconsistent KYC Status and High Transaction Volumes",
            "details": inconsistent_kyc
        })

    # Case 10: Multiple Failed Transaction Attempts Followed by Success
    cursor.execute("""
    SELECT t.customer_id, t.transaction_id, t.transaction_status, t.transaction_date
    FROM transactions t
    WHERE t.transaction_status = 'failed'
    AND EXISTS (
        SELECT 1
        FROM transactions t2
        WHERE t2.customer_id = t.customer_id
        AND t2.transaction_status = 'successful'
        AND TIMESTAMPDIFF(MINUTE, t.transaction_date, t2.transaction_date) BETWEEN 0 AND 5);
        );
    """)
    failed_transactions = cursor.fetchall()
    if failed_transactions:
        discrepancies.append({
            "type": "Multiple Failed Transaction Attempts Followed by Success",
            "details": failed_transactions
        })

    return discrepancies
