# DB Management
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()

def create_user():
    c.execute("CREATE TABLE IF NOT EXISTS users(name TEXT, username TEXT, password TEXT)")
    
def add_userdata(name,username,password):
    c.execute('INSERT INTO users(username,password) VALUES (?, ?)', (name, username, password))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * FROM users WHERE username=? AND password=?', (username, password))
    data = c.fetchall()
    return data