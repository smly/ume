# -*- coding: utf-8 -*-
import os
import sqlite3

import ume


def add_validation_score(jn_name, version, metric, cv_score):
    conn = sqlite3.connect('.umedb')
    cur = conn.cursor()
    cur.execute("""
INSERT INTO validation_log(version, metric, model, score)
VALUES(?, ?, ?, ?)
""", (version, metric, jn_name, cv_score))
    cur.execute("""
INSERT INTO action_log(action_type, action)
VALUES('VALIDATE', ?)
""", ("Run {0}".format(jn_name),))
    conn.commit()
    conn.close()


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
    cur.execute("""
CREATE TABLE validation_log(
    id INTEGER NOT NULL PRIMARY KEY,
    version text,
    metric text,
    model text,
    score real,
    update_date TIMESTAMP DEFAULT (DATETIME('now', 'localtime')));
""")

    conn.commit()
    conn.close()


def exists_sqlitedb():
    return os.path.exists(".umedb")
