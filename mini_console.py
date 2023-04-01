import sqlite3
from models.motion import Motion
from models.patient import Patient
import pdb
conn = sqlite3.connect('motion_analysis.db')
cursor = conn.cursor()

pdb.set_trace()

print("Bye!")