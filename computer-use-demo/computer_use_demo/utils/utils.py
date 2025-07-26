import pickle
import sqlite3


def persist_message(state_messages: list, message: dict):
    state_messages.append(message)
    with sqlite3.connect("computer_use_demo/state.db") as connection:
        cursor = connection.cursor()
        params = ("-", pickle.dumps(message))
        cursor.execute("INSERT INTO messages (session_id, message) VALUES (?, ?)", params)
        connection.commit()
