# task 1
import os

folder_name = "project"
subfolder_data = "project/data"
subfolder_results = "project/results"

try:
    os.mkdir(folder_name)
    os.mkdir(subfolder_results)
    os.mkdir(subfolder_data)
except OSError:
    print("Folder already exists")

import pandas as pd

students = pd.read_csv("project/data/students.csv")
resultToJson = ''

for idx,row in students.iterrows():
    result = f'Student {row["full_name"]}, has average score: {sum(row[2:]) / len(row[2:])}\n'
    resultToJson += result

confirmation = 'n'
if os.path.exists("project/results/report.json"):
    user_confirmation = input("file exists and it will be rewritten\nDo you want to continue? [y/n]:\n")
    if user_confirmation.lower() == 'y':
        with open("project/results/report.json", 'w') as report:
            report.write(resultToJson)
else:
    with open("project/results/report.json", 'w') as report:
        report.write(resultToJson)


import zipfile as zf

with zf.ZipFile('results.zip','w',zf.ZIP_DEFLATED) as zip:
    zip.write('project/results')

import pathlib as pl
from datetime import datetime

pathToResult = pl.Path("project/results/report.json")
if pathToResult.exists():
    print("Size:", pathToResult.stat().st_size, "bytes")
    print(f"Modification date {datetime.fromtimestamp(pathToResult.stat().st_mtime):%Y-%m-%d %H:%M:%S}")