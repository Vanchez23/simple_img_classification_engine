from pathlib import Path
import sqlite3 as sl
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    df_file = Path(db_file)
    df_file.parent.mkdir(exist_ok=True)
    try:
        con = sl.connect(db_file)
        print(sl.version)
        with con:
            con.execute("""PRAGMA foreign_keys=on""")
            con.execute("""
                CREATE TABLE trains (
                    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                    experiment TEXT,
                    model_path TEXT,
                    loss REAL,
                    accuracy REAL,
                    f1_score REAL,
                    precision REAL,
                    recall REAL,
                    data_version TEXT,
                    duration TEXT,
                    created_at TEXT
                );
            """)
            con.execute("""
                CREATE TABLE predictions (
                    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                    image_link TEXT,
                    image_label INTEGER,
                    image_class TEXT,
                    prediction_time REAL,
                    created_at DATETIME,
                    train_id INTEGER,
                    FOREIGN KEY(train_id) REFERENCES trains(id)
                );
            """)
    except Error as e:
        print(e)
    finally:
        if con:
            con.close()


if __name__ == '__main__':
    create_connection(r"db/pythonsqlite.db")
