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

for idx, row in students.iterrows():
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

with zf.ZipFile('results.zip', 'w', zf.ZIP_DEFLATED) as zip:
    zip.write('project/results')

import pathlib as pl
from datetime import datetime

pathToResult = pl.Path("project/results/report.json")
if pathToResult.exists():
    print("Size:", pathToResult.stat().st_size, "bytes")
    print(f"Modification date {datetime.fromtimestamp(pathToResult.stat().st_mtime):%Y-%m-%d %H:%M:%S}")

# task 2

try:
    os.mkdir("generated_txtfiles")
except OSError:
    print("Folder_generated already exists")

texts = [
    """Маленький рассказ
На рассвете дисплей будильника мигнул и… извинился. «Ты так старался вчера, можно ещё десять минут». Я улыбнулся, укрылся сильнее и впервые поверил, что техника тоже умеет быть доброй.""",

    """Мотивационная заметка
Начни с микрошагов: один абзац, один звонок, одно отжимание. Мозгу всё равно, маленькая победа или большая — дофамин одинаковый. Собери их в цепочку, и привычка сама потянет тебя вперёд.""",

    """Описание вымышленного продукта
Кружка “ThermoWhisper” держит температуру 6 часов и тихо подсказывает цветом, когда кофе идеален. Матовая керамика не скользит, крышка закрывается магнитно. Никаких кнопок — просто налей и пей.""",

    """Научпоп-абзац
Хеш-таблица хранит пары «ключ–значение» и рассчитывает индекс через хеш-функцию. При коллизиях элементы складываются в цепочки или перераспределяются по другой схеме. Амортизированно операции поиска и вставки близки к O(1), если выбрать хорошую хеш-функцию и поддерживать коэффициент заполнения.""",

    """Лирика
Вечер сдвинул небо, как занавес,
и город стал аквариумом ламп.
Мы тихо шли: шаги — как лёгкий вес,
а ветер нёс сентябрьский шампунь фонтан."""
]

for i in range(5):
    with open("generated_txtfiles/name_" + str(i) + ".txt", "w") as text_file:
        text_file.writelines(texts[i])

searchWord = input("Enter search word: ").strip().lower()
needFiles = []
for entry in os.scandir("generated_txtfiles"):
    if entry.is_file():
        try:
            with open(entry.path, "r") as text_file:
                for line in text_file:
                    if searchWord in line.lower():
                        needFiles.append(entry.name.lower())
                        break
        except FileNotFoundError:
            print("File not found")
if len(needFiles) > 0:
    print(needFiles)
else:
    print("No file found")

