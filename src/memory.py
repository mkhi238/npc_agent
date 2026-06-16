import sqlite3
from .config import DB_PATH

def get_connection():
  return sqlite3.connect(DB_PATH)

def create_npc_table(table_name):
  conn = get_connection()
  conn.execute(f'''
               CREATE TABLE IF NOT EXISTS {table_name} (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 role TEXT,
                 message TEXT,
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
               )
               ''')
  conn.commit()
  conn.close()
  
def store_message(table_name, role, message):
  conn = get_connection()
  conn.execute(
    f"INSERT INTO {table_name} (role, message) VALUES (?, ?)",
    (role, message)
  )
  conn.commit()
  conn.close()

#Retrieves last N exchanges (N*2 rows since each exchange is player + npc). Reversed so oldest is first.
def get_recent_history(table_name, n = 3):
  conn = get_connection()
  rows = conn.execute(
    f"SELECT role, message FROM {table_name} ORDER BY id DESC LIMIT ?",
    (n *2,)
  ).fetchall()
  conn.close()
  return list(reversed(rows))

#counts how many times the player has spoken to this NPC.
def get_message_count(table_name):
  conn = get_connection()
  count = conn.execute(
      f"SELECT COUNT(*) FROM {table_name} WHERE role = ?",
      ("player",)
  ).fetchone()[0]
  conn.close()
  return count

#clear table
def clear_table(table_name):
  conn = get_connection()
  conn.execute(
    f"DELETE FROM {table_name}"
  )
  conn.commit()
  conn.close()

