# task 1
text = 'Data Science is awesome'

wordsCnt = len(text.split())
print("Words count", wordsCnt)

upperString = text.upper()
print("UPPER STRING", upperString)

last3Chars = text[-3:]
print("Last 3 characters", last3Chars)

print('-' * 10)
# task 2
gradesOfStudents = [90, 80, 70, 60]
gpa = sum(gradesOfStudents) / len(gradesOfStudents)
print("GPA", gpa)

studentsGrades = {
    "Alex": 95,
    "Bob": 100,
    "Carolina": 90,
    "David": 60,
}
mxGrade, mxName = 0, ''
for i in studentsGrades:
    if studentsGrades[i] > mxGrade:
        mxGrade = studentsGrades[i]
        mxName = i
print("The student with maximum grade", mxName)

textToUnique = 'Lorem lorem ipsum ipsum a a a a A b B z'
uniqueWords = set(textToUnique.split())
print("Unique words", uniqueWords)

print('-' * 10)


# task 3
def maxStudentGrade(studentsList, grade):
    for student in studentsList:
        if student["grade"] > grade:
            print("the student who has more than", grade, "is", student["name"])

def avgGrade(studentsList):
    sumGrades = 0
    for student in studentsList:
        sumGrades += student["grade"]

    print("The average grade is", sumGrades / len(studentsList))

def minMaxGrade(studentsList):
    minGrade = 101
    mxGrade = 0
    for student in studentsList:
        if student["grade"] < minGrade:
            minGrade = student["grade"]
        if student["grade"] > mxGrade:
            mxGrade = student["grade"]
    print("The minimum grade is", minGrade)
    print("The maximum grade is", mxGrade)


students = [
    {"name": "Aliya", "grade": 88},
    {"name": "Dana", "grade": 92},
    {"name": "Rustem", "grade": 79},
    {"name": "Samat", "grade": 85}
]

maxStudentGrade(students, 85)
avgGrade(students)
minMaxGrade(students)