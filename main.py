import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import array_to_img
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget,\
    QHBoxLayout, QMessageBox, QInputDialog
from collections import Counter
import random
import easygui


def image_reshape(image):
    # This function turns an image at image_path to array with image_size
    new_image = cv2.resize(image, (image_size, image_size))
    new_image_gray = cv2.cvtColor(new_image, cv2.COLOR_RGB2GRAY)
    # print(new_image.shape)
    return new_image_gray.reshape(-1, image_size, image_size, 1)


def predict_batch():

    capture = cv2.VideoCapture(0)
    results = []
    for i in np.arange(100):
        key = cv2.waitKey(1)
        ret, frame = capture.read()
        (x, y, z) = frame.shape
        frame = frame[0:x, int((y - x) / 2):int(y - ((y - x) / 2))]
        frame = cv2.flip(frame, 1)
        frame2learn = frame.copy()
        start_pt = int((x - sqr_dim) / 2)
        end_pt = int(x - ((x - sqr_dim) / 2))
        cv2.rectangle(frame, (start_pt, start_pt), (end_pt, end_pt), (255, 0, 0), 3)
        cv2.imshow('Teaching Mode ASL', frame)
        prediction = model.predict(image_reshape(frame2learn))
        results.append(np.argmax(prediction))
    capture.release()
    cv2.destroyAllWindows()
    c = Counter(results)
    index = list(c.most_common(3)[i][0] for i in np.arange(len(c.most_common(3))))
    percentage = list(c.most_common(3)[i][1] for i in np.arange(len(c.most_common(3))))
    labels = [categories[i] for i in index]
    return labels, percentage


def quizMode():
    capture = cv2.VideoCapture(0)
    displayed_text = random.choice(categories)
    counter = 0
    n = 0
    while n <= 500:
        key = cv2.waitKey(5)
        ret, frame = capture.read()
        (x, y, z) = frame.shape
        frame = frame[0:x, int((y - x) / 2):int(y - ((y - x) / 2))]
        start_pt = int((x - sqr_dim) / 2)
        end_pt = int(x - ((x - sqr_dim) / 2))
        frame = cv2.flip(frame, 1)
        frame2learn = frame.copy()
        cv2.rectangle(frame, (start_pt, start_pt), (end_pt, end_pt), (255, 0, 0), 3)
        cv2.putText(frame, 'Please make the sign for '+str(displayed_text), org=(0, 450), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6, color=(0, 0, 0), thickness=2)
        cv2.imshow('Quiz Mode ASL', frame)
        prediction = model.predict(image_reshape(frame2learn))
        answer = categories[np.argmax(prediction)]
        print(answer)
        if answer == displayed_text:
            displayed_text = random.choice(categories)
            counter += 1
            continue
        if counter == 10:
            break
        if key % 256 == 113:							# q
            break
        n += 1
    capture.release()
    cv2.destroyAllWindows()
    return counter


def getImage():
    capture = cv2.VideoCapture(0)
    label = easygui.enterbox('Please enter your query:')
    alpha = 0.6
    foreground = cv2.imread('Query_base/'+label+'.jpg')
    while True:
        key_escape = cv2.waitKey(1)
        ret, frame = capture.read()
        frame = cv2.flip(frame, 1)
        x, y, _ = frame.shape
        # Select the region in the background where we want to add the image and add the images using cv2.addWeighted()
        added_image = cv2.addWeighted(frame[int(x/2)-100:int(x/2)+100, int(y/2)-200:int(y/2), :], alpha, foreground[:, :, :], 1 - alpha, 0)
        frame[int(x/2)-100:int(x/2)+100, int(y/2)-200:int(y/2)] = added_image
        cv2.putText(frame, label, (int(1/2*frame.shape[0]), int(1/4*frame.shape[1])), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, 'Press Q to submit result', org=(0, 450),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6, color=(0, 0, 255), thickness=2)
        cv2.imshow('Learning Mode', frame)
        if key_escape % 256 == 113:
            ret, frame2learn = capture.read()
            break
    prediction = model.predict(image_reshape(frame2learn))
    capture.release()
    cv2.destroyAllWindows()
    return categories[np.argmax(prediction)]


def button3_clicked():

    alert = QMessageBox()
    alert.setText("Training Mode: be prepared to sign")
    alert.exec_()
    labels, percentage = predict_batch()
    text_display = 'You signed:\n'
    for i in np.arange(len(labels)):
        text_display += f'{labels[i]} with {percentage[i]}%\n'
    alert.setText(text_display)
    alert.exec_()


def button2_clicked():

    alert = QMessageBox()
    alert.setText("Quizzing Mode: please sign the letter displayed")
    alert.exec_()
    counter = quizMode()
    alert.setText("Congratulations! You got "+str(counter)+" right!")
    alert.exec_()


def button1_clicked():
    result = getImage()
    alert = QMessageBox()
    alert.setText('Query Finished.\n The computer thinks you are signing: '+result)
    alert.exec_()

    # query, done = QInputDialog.getText(title='Query Window', text='Please enter your query:\n'+str(categories))
    # if done:
    #     getImage(query)
    # alert = QMessageBox()
    # button_reply = alert.question(p_str='Continue Query', p_str_1='Would you like to continue?',
    #                               QMessageBox_StandardButton=QMessageBox.Yes,
    #                               QMessageBox_StandardButtons=QMessageBox.No)
    # alert.exec_()
    # if button_reply == QMessageBox.Yes:
    #     button1_clicked()
    # else:
    #     return


# def main_button_clicked():
#     app_main = QApplication([])
#     win_main = QMainWindow()
#     central_widget = QWidget()
#     button1 = QPushButton('Learning Mode', central_widget)
#     button2 = QPushButton('Quizzing Mode', central_widget)
#     button3 = QPushButton('Teaching Mode', central_widget)
#     layout = QHBoxLayout(central_widget)
#     layout.addWidget(button1)
#     layout.addWidget(button2)
#     layout.addWidget(button3)
#     win_main.setCentralWidget(central_widget)


def main():
    # Load NNDL model
    model = keras.models.load_model('ASL_model_914')
    # Video Stream Set up
    capture = cv2.VideoCapture(0)
    codec = cv2.VideoWriter_fourcc(*'XVID')

    while True:
        key = cv2.waitKey(1)
        k = cv2.waitKey(20) & 0xFF
        [ret, frame] = capture.read()  # taking each frame
        (x, y, z) = frame.shape
        frame = cv2.flip(frame, 1)  # Flip so that image isn't mirrored
        frame = frame[0:x, int((y-x)/2):int(y-((y-x)/2))]

        orig = (0, 465)

        cv2.putText(frame, 'Press Q to exit', orig, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.namedWindow('image')

        cv2.imshow('image', frame)
        prediction = model.predict(image_reshape(frame))
        print(categories[int(np.argmax(prediction))])

        if key%256 == 113:							# q
            break
    capture.release()
    cv2.destroyAllWindows()


image_size = 48
categories = ['A', 'B', 'C', 'D', 'del', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
sqr_dim = 300
model = keras.models.load_model('ASL_model_914')

# main()


# class Query_Window(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.title = "Query Window"
#         self.left = 10
#         self.top = 10
#         self.width = 320
#         self.height = 200
#         self.initUI()
#
#     def initUI(self):
#         self.setWindowTitle(self.title)
#         self.setGeometry(self.left, self.top, self.width, self.height)
#         query, done = QInputDialog.getText(self, 'Query Dialog', 'Enter your query:')
#         if done:
#             getImage(query)
#
#         self.show()



app = QApplication([])
win = QMainWindow()
win.setWindowTitle('Artificial Sign Learner')
win.setGeometry(10, 10, 350, 200)
central_widget = QWidget()
win.setCentralWidget(central_widget)
button1 = QPushButton('Learning Mode', central_widget)
button2 = QPushButton('Quizzing Mode', central_widget)
button3 = QPushButton('Training Mode', central_widget)
layout = QHBoxLayout(central_widget)
layout.addWidget(button1)
layout.addWidget(button3)
layout.addWidget(button2)
win.setCentralWidget(central_widget)
button3.clicked.connect(button3_clicked)
button3.show()
button2.clicked.connect(button2_clicked)
button2.show()
button1.clicked.connect(button1_clicked)
button1.show()
win.show()
app.exit(app.exec_())






