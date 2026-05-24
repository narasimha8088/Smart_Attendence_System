import cv2
import os
import sqlite3

current_dir = os.path.dirname(os.path.abspath(__file__))
classifier_path = os.path.join(current_dir, 'classifier.xml')
db_path = os.path.join(current_dir, 'face_recognition.db')
img_path = os.path.join(current_dir, 'data', 'image.111.1.jpg')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(classifier_path)

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

id, predict = recognizer.predict(img)

print(f"Predicted ID: {id}, Distance: {predict}")

conn=sqlite3.connect(db_path)
my_cursor=conn.cursor()
my_cursor.execute("select * from student_detail where student_id=?", (str(id),))
res = my_cursor.fetchall()
print(f"Database matches: {res}")
