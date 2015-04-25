# -*- coding: utf-8 -*-
import os
import sqlite3


def init_db():
    conn = sqlite3.connect('.umedb')
    cur = conn.cursor()
    cur.execute("""
CREATE TABLE action_log(
    id INTEGER NOT NULL PRIMARY KEY,
    action_type text,
    action text,
    update_date TIMESTAMP DEFAULT (DATETIME('now', 'localtime')));
""")
    cur.execute("""
INSERT INTO action_log(action_type, action)
VALUES('INIT', 'Initialize ume project')
""")

    conn.commit()
    conn.close()


def exists_sqlitedb():
    return os.path.exists(".umedb")
