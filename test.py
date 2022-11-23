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

a = video2frames("./test_authenticator/maor.webm")
plt.imshow(a[0])
plt.show()