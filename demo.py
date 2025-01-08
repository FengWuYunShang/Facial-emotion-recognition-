#参考来源：https://github.com/xionghc/Facial-Expression-Recognition.git

import cv2
import torch
import os
import numpy as np
CASC_PATH = 'haarcascade_frontalface_default.xml'
cascade_classifier = cv2.CascadeClassifier(CASC_PATH)
EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

def predict_image(model, image):
    image = cv2.resize(image, (48, 48))
    image = image.astype('float32') / 255.0
    image = torch.tensor(image).unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    output = model(image)
    predicted = torch.sigmoid(output)
    predicted = predicted.squeeze()
    sum = torch.sum(predicted)
    predicted = (predicted/sum).detach().numpy()
    return predicted


def format_image(image):
  if len(image.shape) > 2 and image.shape[2] == 3:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = cascade_classifier.detectMultiScale(
    image,
    scaleFactor = 1.3,
    minNeighbors = 5
  )
  # None is no face found in image
  if not len(faces) > 0:
    return None, None
  max_are_face = faces[0]
  for face in faces:
    if face[2] * face[3] > max_are_face[2] * max_are_face[3]:
      max_are_face = face
  # face to image
  face_coor =  max_are_face
  image = image[face_coor[1]:(face_coor[1] + face_coor[2]), face_coor[0]:(face_coor[0] + face_coor[3])]
  # Resize image to network size
  try:
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_CUBIC)
  except Exception:
    print("[+} Problem during resize")
    return None, None
  return  image, face_coor


def demo(model, device, path_model_param):
    model.load_state_dict(torch.load(path_model_param, map_location=device))
 
    cap = cv2.VideoCapture('./data/video/expression.mp4')
    model.eval()
    feelings_faces = []
    
    for index, emotion in enumerate(EMOTIONS):
        feelings_faces.append(cv2.imread('./data/emojis/' + emotion + '.png', -1))
    
    emoji_face = []
    prediction = None
    while True:
        ret, frame = cap.read()
        detected_face, _ = format_image(frame)

        if cv2.waitKey(1) & 0xFF == ord(' '):
            if detected_face is not None:
                cv2.imwrite('detc_image.jpg', detected_face)
                prediction = predict_image(model, detected_face)
                print(f"predicted emotion:",EMOTIONS[np.argmax(prediction)])


        if prediction is not None:
            for index, emotion in enumerate(EMOTIONS):
                cv2.putText(frame, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
                cv2.rectangle(frame, (100, index * 20 + 10), (130 + int(prediction[index] * 100), (index + 1) * 20 + 4),
                       (255, 0, 0), -1)
            
            emoji_face = feelings_faces[np.argmax(prediction)]

            for c in range(0, 3):
                frame[200:320, 10:130, c] = emoji_face[:, :, c] * (emoji_face[:, :, 3] / 255.0) + frame[200:320, 10:130, c] * (1.0 - emoji_face[:, :, 3] / 255.0)
        cv2.imshow('face', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
