import cv2
import numpy as np

frameWidth = 500
frameHeight = 400
cap = cv2.VideoCapture(0)

def empty(a):
    pass

def get_color_name(hue_min, hue_max, sat_min, val_min):
    hue_mid = (hue_min + hue_max) / 2

    # Ignore low saturation
    if sat_min < 50 or val_min < 50:
        return None

    if (hue_mid <= 10) or (hue_mid >= 170):
        return "Red"
    elif 35 <= hue_mid <= 85:
        return "Green"
    elif 100 <= hue_mid <= 130:
        return "Blue"
    elif 11 <= hue_mid <= 34:
        return "Yellow"
    elif 86 <= hue_mid <= 99:
        return "Cyan"
    elif 131 <= hue_mid <= 159:
        return "Purple"
    elif 160 <= hue_mid <= 169:
        return "Pink"
    else:
        return None

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 900, 300)
cv2.createTrackbar("Hue MIN", "HSV", 0, 179, empty)
cv2.createTrackbar("Hue MAX", "HSV", 179, 179, empty)
cv2.createTrackbar("Sat MIN", "HSV", 0, 255, empty)
cv2.createTrackbar("Sat MAX", "HSV", 255, 255, empty)
cv2.createTrackbar("Val MIN", "HSV", 0, 255, empty)
cv2.createTrackbar("Val MAX", "HSV", 255, 255, empty)

while True:
    success, img = cap.read()
    img = cv2.resize(img, (frameWidth, frameHeight))
    img = cv2.flip(img, 1)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    hue_min = cv2.getTrackbarPos("Hue MIN", "HSV")
    hue_max = cv2.getTrackbarPos("Hue MAX", "HSV")
    sat_min = cv2.getTrackbarPos("Sat MIN", "HSV")
    sat_max = cv2.getTrackbarPos("Sat MAX", "HSV")
    val_min = cv2.getTrackbarPos("Val MIN", "HSV")
    val_max = cv2.getTrackbarPos("Val MAX", "HSV")
    
    lower = np.array([hue_min, sat_min, val_min])
    upper = np.array([hue_max, sat_max, val_max])
    
    mask = cv2.inRange(imgHSV, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    colorName = get_color_name(hue_min, hue_max, sat_min, val_min)
    hueRange = hue_max - hue_min
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    imgBBox = img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(imgBBox, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(imgBBox, f"Area: {int(area)}", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                
            if hueRange <= 60:
                cv2.putText(imgBBox, f"Color: {colorName}", (x, y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
                
    
    hStack = np.hstack([img, result, imgBBox])
    cv2.imshow("Web Cam", hStack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
