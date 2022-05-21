import sqlite3 as sl
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    try:
        con = sl.connect(db_file)
        print(sl.version)
        with con:
            con.execute("""
                CREATE TABLE predictions (
                    id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
                    image_link TEXT,
                    image_label INTEGER,
                    image_class TEXT,
                    prediction_time DATETIME,
                    created_at DATETIME,
                    status TEXT
                );
            """)
    except Error as e:
        print(e)
    finally:
        if con:
            con.close()


if __name__ == '__main__':
    create_connection(r"db/pythonsqlite.db")
