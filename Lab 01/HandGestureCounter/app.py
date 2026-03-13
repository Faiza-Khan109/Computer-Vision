import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_skin = np.array([0,20,70],dtype=np.uint8)
    upper_skin = np.array([20,255,255],dtype=np.uint8)

    mask = cv2.inRange(hsv,lower_skin,upper_skin)

    contours,_ = cv2.findContours(mask,
                                  cv2.RETR_TREE,
                                  cv2.CHAIN_APPROX_SIMPLE)

    count = len(contours)

    cv2.putText(frame,"Hands: "+str(count),
                (10,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)

    cv2.imshow("Hand Counter",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()