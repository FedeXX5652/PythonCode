import sqlite3

conn = sqlite3.connect('./test1.sqlite')

c = conn.cursor()
try:
    c.execute("CREATE TABLE prods(ID NUMERIC, nombre TEXT, precio NUMERIC)")
    conn.commit()
except sqlite3.OperationalError:
    print("Lista existente")

datos = (
    (1, "teclado", 500),
    (2, "mouse", 350),
    (3, "mouse pad", 100),
    (4, "monitor", 700),
    )

for ID, nombre, precio in datos:
    c.execute("INSERT INTO prods VALUES (?,?,?)", (ID, nombre, precio))
conn.commit()
conn.close()