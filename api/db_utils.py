import sqlite3
from datetime import datetime

DB_NAME = "rag_app.db"

def get_db_connection():
    connect = sqlite3.connect(DB_NAME)
    connect.row_factory = sqlite3.Row
    return connect


def create_application_logs():
    connect = get_db_connection()
    connect.execute(
            '''
            CREATE TABLE IF NOT EXISTS app_logs
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                user_query TEXT,
                gpt_response TEXT,
                model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
            '''
            )
    connect.close()

def insert_application_logs(session_id, user_query, gpt_response, model):
    connect = get_db_connection()
    connect.execute(''' 
        INSERT INTO app_logs (session_id, user_query, gpt_response, model) 
                    VALUES (?,?,?,?)
        ''',
        (session_id, user_query, gpt_response, model)
        )
    connect.commit()
    connect.close()

def get_chat_history(session_id):
    connect = get_db_connection()
    cursor = connect.cursor()
    cursor.execute(
        '''
        SELECT 
            user_query,
            gpt_response
        FROM
            app_logs
        WHERE
            session_id = ?
        ORDER BY  created_at
    ''',
    (session_id,)
    )
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {'role':'human', 'content': row['user_query']},
            {'role':'ai', 'content': row['gpt_response']}

        ]) 
    connect.close()
    return messages

def create_document_store():
    connect = get_db_connection()
    connect.execute('''
                    CREATE TABLE IF NOT EXISTS document_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
                    ''')
    connect.close()

def insert_document_record(filename):
    connect = get_db_connection()
    cursor = connect.cursor()
    cursor.execute('INSERT INTO document_store (filename) VALUES (?)', (filename,))
    file_id = cursor.lastrowid
    connect.commit()
    connect.close()
    return file_id

def delete_document_record(file_id):
    connect = get_db_connection()
    connect.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
    connect.commit()
    connect.close()
    return True

def get_all_documents():
    connect = get_db_connection()
    cursor = connect.cursor()
    cursor.execute('SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
    documents = cursor.fetchall()
    connect.close()
    return [dict(doc) for doc in documents]

# Initialize the database tables
create_application_logs()
create_document_store()