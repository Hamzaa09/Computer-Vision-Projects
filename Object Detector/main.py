import cv2
import os

path = f"C:\\Users\\hamza\\OneDrive\\Computer Vision\\Object Detector\\"
coco = f"{path}coco.names"

className = []
with open(coco, 'rt') as file:
    className = file.read().split('\n')
    
config_path = f"{path}ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weight_path = f"{path}frozen_inference_graph.pb"

img_url = "C:\\Users\\hamza\\OneDrive\\Computer Vision\\images\\text_image1.jpg"
img = cv2.imread(img_url)

# cap = cv2.VideoCapture(0)

net = cv2.dnn_DetectionModel(weight_path, config_path)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputScale(1.0 / 127.5)
net.setInputSize(320, 320)
net.setInputSwapRB(True)

# while True:
#     success, img = cap.read()
classIds, confs, bbox = net.detect(img, confThreshold=0.5)

if len(classIds):     
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
        prediction = className[classId - 1] if classId-1 < len(className) else "None"
        cv2.putText(
            img,
            f"{prediction.upper()} {round(confidence * 100, 2)}%",
            (box[0] + 10, box[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

cv2.imshow("Image", img)  
cv2.waitKey(0)  
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     break

cv2.destroyAllWindows()
# cap.release()