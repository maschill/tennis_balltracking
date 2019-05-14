import numpy as np 
import cv2
import tensorflow as tf 
import sys
import time

def preprocess_cv2(image):
    img = cv2.resize(image, (224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/127.5
    img = img-1.0
    return np.expand_dims(img,0)


if len(sys.argv) < 2:
    print("Usage: video_with_tf.py /path/to/file")

cap = cv2.VideoCapture(sys.argv[-1])

cv2.namedWindow("frame")
cv2.moveWindow('frame', 20, 100)

ret, frame = cap.read()

isLive_model = tf.contrib.predictor.from_saved_model('tensorflow/tf_ckpts/isLive_mobilenetv2_v01')

label_names = {"dead":0, "live":1}
label_reverse = {v:k for k,v in label_names.items()}
frame_count = 0
start = time.time()

while cap.isOpened():
    frame_count += 1
    ret, frame = cap.read()

    img = preprocess_cv2(frame)
    output = isLive_model({"mobilenetv2_1.00_224_input":img})["dense_3"]
    pred = np.exp(output)/(1+np.exp(output))
    cv2.putText(frame, label_reverse[np.argmax(pred)], (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(15) & 0xFF==ord('q'):
        break

end = time.time()

cv2.destroyAllWindows()
cap.release()
print(f"average calc time per frame: {(end-start)/frame_count:{.4}}s")


