import sqlite3

# Function to check if a video ID already exists
def item_exists(item_id):
    # Connect to the database (creates a new file if it doesn't exist)
    conn = sqlite3.connect('items.db')

    # Create a cursor object to execute SQL statements
    cursor = conn.cursor()

    # Create a table to store video IDs if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY
        )
    ''')
    cursor.execute('SELECT * FROM items WHERE id=?', (item_id,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result is not None

# Function to insert a new item ID
def insert_item(item_id):
    # Connect to the database (creates a new file if it doesn't exist)
    conn = sqlite3.connect('items.db')

    # Create a cursor object to execute SQL statements
    cursor = conn.cursor()

    # Create a table to store video IDs if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS items (
            id TEXT PRIMARY KEY
        )
    ''')
    cursor.execute('INSERT INTO items VALUES (?)', (item_id,))
    conn.commit()
    cursor.close()
    conn.close()