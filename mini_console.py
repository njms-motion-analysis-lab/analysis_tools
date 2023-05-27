import sqlite3

from table import Table
import pdb

Table.create_tables()


from models.task import Task




tasks = Task.all()
import pdb
pdb.set_trace()
for t in tasks:
    t.plot_and_save_all()



print("Done!")