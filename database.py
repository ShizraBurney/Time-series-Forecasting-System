import sqlite3

def init_db():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS forecasts (
        id INTEGER PRIMARY KEY,
        dataset TEXT,
        model TEXT,
        date TEXT,
        forecasted_value REAL
    )
    ''')
    conn.commit()
    conn.close()

def insert_forecast(dataset, model, date, forecasted_value):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('''
    SELECT 1 FROM forecasts WHERE dataset = ? AND model = ? AND date = ?
    ''', (dataset, model, date))
    if c.fetchone() is None:
        c.execute('''
        INSERT INTO forecasts (dataset, model, date, forecasted_value) 
        VALUES (?, ?, ?, ?)
        ''', (dataset, model, date, forecasted_value))
        conn.commit()
    conn.close()


def get_forecasts(dataset, model):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('''
    SELECT date, forecasted_value FROM forecasts
    WHERE dataset = ? AND model = ?
    ORDER BY date
    ''', (dataset, model))
    results = c.fetchall()
    conn.close()
    return results
