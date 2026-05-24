# Smart Attendance System

The objective of this project is to process a live video-stream of students entering their classroom and generate a list of students attending the class using facial recognition.
The system is built in Python using OpenCV, Tkinter (for GUI), and SQLite (for database).

## Project Structure

* **`main.py`** : The main script of the project. It implements the Graphical User Interface (GUI) and acts as the entry point to call other modules.
* **`Cascades/haarcascade_frontalface_default.xml`** : Contains trained configurations of the Viola-Jones Algorithm for detecting faces, provided by OpenCV.
* **`student_detail.py`** : Contains functions to collect student information, capture face images, and store details into the database.
* **`train.py`** : Processes the collected face dataset, trains the Local Binary Patterns Histograms (LBPH) face recognizer model, and stores the outcome into the `classifier.xml` file.
* **`facial_recognition.py`** : Recognizes the student through the camera feed using the trained model and logs their attendance.
* **`attendance.py`** : Manages attendance records, saving them to `attendance.csv` and allowing the user to view or export them.
* **`developer.py` / `help.py`** : Additional GUI components for developer info and help desk.

## Getting Started

### Prerequisites

The system requires Python 3.8+ installed along with the following libraries:
- OpenCV (`opencv-contrib-python`)
- Pillow
- Numpy

### Installing

1. Clone this repository to your local machine.
2. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   python main.py
   ```

## Workflow

1. **Student Registration**: Open the application, go to "Student Details", and fill in the required fields. Capture face samples using the webcam.
2. **Train Data**: Go to "Train Data" to train the LBPH Face Recognizer with the newly captured images.
3. **Take Attendance**: Open "Face Recognition" to start the webcam. The system will detect and recognize faces, marking attendance automatically in a CSV file.
4. **View Attendance**: Check the "Attendance" section to view the logged records.

## Built With

* [OpenCV](https://opencv.org/) - Facial Detection and Recognition
* [Tkinter](https://docs.python.org/3/library/tk.html) - GUI Implementation
* [Numpy](http://www.numpy.org/) - Array and Matrix Operations
* [Pillow](https://pillow.readthedocs.io/) - Image Processing
* [SQLite](https://www.sqlite.org/) - Database Management

## Author
- **Narsimha**
