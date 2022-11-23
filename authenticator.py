import cv2
import face_recognition
import numpy as np 
import matplotlib.pyplot as plt

IMG_SIZE = 256
NUM_FRAMES_PER_VIDEO = 16

def video2frames(video_path, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    is_there_frame = True
    num_total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) 
    resampling_rate = int(num_total_frames / NUM_FRAMES_PER_VIDEO)
    idf = 0
    while is_there_frame and len(frames)<NUM_FRAMES_PER_VIDEO:
        idf = idf + 1
        is_there_frame, frame = cap.read()
        if frame is None:
            return np.array([])
        if idf % resampling_rate == 0:
            frame = cv2.resize(frame, resize)
            frames.append(frame)
    assert len(frames)==NUM_FRAMES_PER_VIDEO
    return frames

## FIRST IMAGE (FIRST VIDEO BASED ON ID)
frames_img1 = video2frames("./dataset/train_release/1/1.avi") ## FIRST VIDEO
img1 = frames_img1[NUM_FRAMES_PER_VIDEO//2 - 1]
rgb1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
#boxes1 = face_recognition.face_locations(rgb1, model="cnn")
encodings1 = face_recognition.face_encodings(rgb1)[0]

## SECOND IMAGE (PROVIDED IN LOGIN)
frames_img2 = video2frames("./dataset/train_release/2/2.avi") ## LOGIN VIDEO
img2 = frames_img2[NUM_FRAMES_PER_VIDEO//2 - 1]
rgb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
#boxes2 = face_recognition.face_locations(rgb2, model="cnn")
encodings2 = face_recognition.face_encodings(rgb2)[0]

maor = face_recognition.load_image_file("./test_authenticator/obama-240p.jpg")
maor2 = face_recognition.load_image_file("./test_authenticator/Official_portrait_of_Barack_Obama.jpg")

maor_encodings = face_recognition.face_encodings(maor)[0]
maor2_encodings = face_recognition.face_encodings(maor2)[0]

results = face_recognition.compare_faces([maor_encodings], maor2_encodings, tolerance=0.32)
results1 = face_recognition.compare_faces([encodings1], encodings2, tolerance=0.32)
print(results)
print(results1)