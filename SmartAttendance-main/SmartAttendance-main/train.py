from tkinter import*
from tkinter import ttk
from PIL import Image,ImageTk
from tkinter import messagebox
# import mysql.connector (unused)
import cv2
import os
import numpy as np

class Train:
    def __init__(self,root):
        self.root=root
        self.root.geometry("1530x790+0+0")
        self.root.title("face Recognition System")

        title_lbl=Label(self.root,text="TRAIN DATASET",font=("times new roman",35,"bold"),bg="blue")
        title_lbl.place(x=0,y=0,width=1530,height=60)

        #image
        img_top=Image.open("Images/training2.jpeg")
        img_top=img_top.resize((1530,725),Image.Resampling.LANCZOS)
        self.photoimg_top=ImageTk.PhotoImage(img_top)

        f_lbl=Label(self.root,image=self.photoimg_top)
        f_lbl.place(x=0,y=60,width=1530,height=725)

        #button
        b1_1=Button(self.root,text="TRAIN DATA",command=self.train_classifier,cursor="hand2",font=("times new roman",24,"bold"),bg="green",fg="white")
        b1_1.place(x=250,y=385,width=250,height=50)

        #image
        # img_bottom=Image.open("Images/training2.jpeg")
        # img_bottom=img_bottom.resize((530,325),Image.Resampling.LANCZOS)
        # self.photoimg_bottom=ImageTk.PhotoImage(img_bottom)

        # f_lbl=Label(self.root,image=self.photoimg_bottom)
        # f_lbl.place(x=0,y=435,width=530,height=325)

    def train_classifier(self):
        # Use absolute path for data directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "data")
        
        # Check if directory exists and has files
        if not os.path.exists(data_dir) or not os.listdir(data_dir):
            messagebox.showerror("Error", f"No image samples found in: {data_dir}\nPlease take photos first!", parent=self.root)
            return

        path=[os.path.join(data_dir,file) for file in os.listdir(data_dir)]

        faces=[]
        ids=[]

        for image in path:
            try:
                img=Image.open(image).convert('L')  #gray scale image
                imageNp=np.array(img,'uint8')
                # Correctly parse the ID from the filename: image.ID.SAMPLE.jpg
                id=int(os.path.split(image)[1].split('.')[1])

                faces.append(imageNp)
                ids.append(id)
                cv2.imshow("Training",imageNp)
                cv2.waitKey(1)
            except Exception as e:
                print(f"Skipping file {image}: {e}")
                continue

        if not faces:
            messagebox.showerror("Error", "No valid image data found for training.", parent=self.root)
            return

        ids=np.array(ids)

        #train classifier
        try:
            clf=cv2.face.LBPHFaceRecognizer_create()
            clf.train(faces,ids)
            clf.write("classifier.xml")
            cv2.destroyAllWindows()
            messagebox.showinfo("Result","Training datasets completed!!", parent=self.root)
        except AttributeError:
            messagebox.showerror("Error", "Missing required OpenCV modules. Please run: pip install opencv-contrib-python", parent=self.root)
        except Exception as e:
            messagebox.showerror("Error", f"An unexpected error occurred: {str(e)}", parent=self.root)

        
if __name__== "__main__":
    root=Tk()
    obj=Train(root)
    root.mainloop()