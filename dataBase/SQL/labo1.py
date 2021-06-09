import sqlite3

plus=""

conn = sqlite3.connect('productos.db')

c = conn.cursor()
try:
    c.execute("CREATE TABLE productos(ID NUMERIC, nombre TEXT, precio NUMERIC)")
    conn.commit()
except sqlite3.OperationalError:
    print("Lista existente")

while True:
    plus = input("Add product? ")
    if plus == "n":
        break
    else:

        while True:
            try:
                ID = int(input("Enter the ID: "))
                break
            except ValueError:
                print("Enter a valid ID\n")
        while True:
            try:
                name = str(input("Enter name: "))
                break
            except ValueError:
                print("Enter a valid name\n")
        while True:
            try:
                price = int(input("Enter the price: "))
                break
            except ValueError:
                print("Enter a valid price\n")

        c.execute("INSERT INTO productos VALUES (?,?,?)", (ID, name, price))
        conn.commit()
conn.close()