from tkinter import*
from tkinter import ttk
from PIL import Image,ImageTk
from tkinter import messagebox
import sqlite3
import cv2
import os
from time import strftime
from datetime import datetime
import numpy as np

class Face_Recognition:
    def __init__(self,root):
        self.root=root
        self.root.geometry("1530x790+0+0")
        self.root.title("face Recognition System")

        title_lbl=Label(self.root,text="FACE RECOGNITION",font=("times new roman",35,"bold"),bg="green")
        title_lbl.place(x=0,y=0,width=1530,height=45)

        #1st image
        img_top=Image.open("Images/face_recog1.jpg")
        img_top=img_top.resize((650,700),Image.Resampling.LANCZOS)
        self.photoimg_top=ImageTk.PhotoImage(img_top)

        f_lbl=Label(self.root,image=self.photoimg_top)
        f_lbl.place(x=0,y=55,width=650,height=700)

        #2nd image
        img_bottom=Image.open("Images/face_recog2.jpg")
        img_bottom=img_bottom.resize((950,700),Image.Resampling.LANCZOS)
        self.photoimg_bottom=ImageTk.PhotoImage(img_bottom)

        f_lbl=Label(self.root,image=self.photoimg_bottom)
        f_lbl.place(x=650,y=55,width=950,height=700)

        #button
        b1_1=Button(f_lbl,text="RECOGNIZE",command=self.recognize_attendence,cursor="hand2",font=("times new roman",18,"bold"),bg="darkgreen",fg="white")
        b1_1.place(x=365,y=620,width=200,height=40)



########################################  attendance  #########################33

    def mark_attendance(self,i,n,d):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        attendance_path = os.path.join(current_dir, 'attendance.csv')
        try:
            with open(attendance_path,"r+",newline="\n") as f:
                myDataList = f.readlines()
                name_list=[]
                for line in myDataList:
                    entry=line.split((","))
                    name_list.append(entry[0])

                #to avoid repeat attendance
                if( (i not in name_list) and (n not in name_list) and (d not in name_list) ):
                    now=datetime.now()
                    d1=now.strftime("%d/%m/%y")
                    dtString=now.strftime("%H:%M:%S")
                    f.writelines(f"\n{i},{n},{d},{dtString},{d1},Present")
        except Exception as e:
            print("Error writing attendance:", e)
            






########################################  face recognition  #################3
    
    def recognize_attendence(self):
        # Use absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        classifier_path = os.path.join(current_dir, 'classifier.xml')
        cascade_path = os.path.join(current_dir, 'Cascades', 'haarcascade_frontalface_default.xml')
        db_path = os.path.join(current_dir, 'face_recognition.db')
        attendance_path = os.path.join(current_dir, 'attendance.csv')

        # Verify classifier exists
        if not os.path.exists(classifier_path):
            messagebox.showerror("Error", f"Trained model not found at: {classifier_path}\nPlease capture photos and click 'Train Data' first.", parent=self.root)
            return

        # Ensure attendance.csv exists
        if not os.path.exists(attendance_path):
            with open(attendance_path, "w", newline="\n") as f:
                f.write("ID,Name,Department,Time,Date,Status")

        recognizer = cv2.face.LBPHFaceRecognizer_create()  
        #recognizer.read("TrainingImageLabel"+os.sep+"Trainner.yml")
        recognizer.read(classifier_path)
        faceCascade = cv2.CascadeClassifier(cascade_path)
        font = cv2.FONT_HERSHEY_SIMPLEX
        

        # Fetch all student data ONCE before the frame loop to prevent SQLite locking
        student_data_dict = {}
        try:
            conn = sqlite3.connect(db_path, timeout=10)
            my_cursor = conn.cursor()
            my_cursor.execute("select student_id, name, dep, eno from student_detail")
            for row in my_cursor.fetchall():
                student_data_dict[str(row[0])] = {
                    "name": row[1] if row[1] else "Unknown",
                    "dep": row[2] if row[2] else "Unknown",
                    "eno": row[3] if row[3] else "Unknown"
                }
            conn.close()
        except Exception as e:
            print("DB pre-fetch error:", e)

        # start realtime video capture
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cam.isOpened():
            cam = cv2.VideoCapture(0) # fallback without DSHOW
        if not cam.isOpened():
            messagebox.showerror("Error", "Unable to open webcam.", parent=self.root)
            return

        cam.set(3, 640) 
        cam.set(4, 480) 
        minW = 0.1 * cam.get(3)
        minH = 0.1 * cam.get(4)

        while True:
            ret, im = cam.read()
            if not ret or im is None:
                messagebox.showerror("Error", "Lost connection to webcam.", parent=self.root)
                break
            
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(gray, 1.2, 5,
            minSize = (int(minW), int(minH)),flags = cv2.CASCADE_SCALE_IMAGE)
            for(x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x+w, y+h), (10, 159, 255), 2)
                id,predict=recognizer.predict(gray[y:y+h,x:x+w])
                confidence=int((100*(1-predict/300)))

                try:
                    str_id = str(id)
                    n = student_data_dict.get(str_id, {}).get("name", "Unknown")
                    d = student_data_dict.get(str_id, {}).get("dep", "Unknown")
                    i = student_data_dict.get(str_id, {}).get("eno", str_id)
                except Exception as e:
                    n="DB Error"
                    d="DB Error"
                    i=str(id)
                        
                if confidence>77:
                    cv2.putText(im,f"ID:{i}",(x,y-55),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
                    cv2.putText(im,f"Name:{n}",(x,y-30),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
                    cv2.putText(im,f"Dep:{d}",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)
                    self.mark_attendance(i,n,d)                
                else:
                    cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),3)
                    cv2.putText(im,"Unknown Face",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,0.8,(255,255,255),3)


            cv2.imshow("Welcome to Face Recognition",im)
    
            if (cv2.waitKey(1) == ord('q') or cv2.getWindowProperty("Welcome to Face Recognition", cv2.WND_PROP_VISIBLE) < 1):
                break
        
        cam.release()
        cv2.destroyAllWindows()













if __name__== "__main__":
    root=Tk()
    obj=Face_Recognition(root)
    root.mainloop()
