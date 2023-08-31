import numpy as np
import cv2, math

def detect_letter(imgCanny,img):
    contours,hierarchy = cv2.findContours(imgCanny,cv2.RETR_TREE ,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        length = int(len(contours))
        area = cv2.contourArea(cnt)
        print(f"the length is :{length}")
        if area > 3000 :
            for i in range(length):
                if  hierarchy[0][i][2] == -1 and hierarchy[0][i][3] != -1 :
                    contour_position = i
                    peri = cv2.arcLength(contours[contour_position],True)
                    approx = cv2.approxPolyDP(contours[contour_position], 0.01 * peri, True)
                    objCor = len(approx)         
                    print(f"obj cor: {objCor}")

                    if objCor == 12:
                        print("FOUND IT BOIS")
                        x, y, w, h = cv2.boundingRect(contours[contour_position])
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
                        cv2.circle(img,(int((x+w/2)),int((y+h/2))),6,(255,0,255),thickness=3)
                        cv2.drawContours(img,contours[contour_position],-1,(255,0,0),2)
                        cv2.putText(img,'Successfull',(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),1,cv2.LINE_AA)
                        print(contour_position)
                    
                    else:
                        print("you have smth else as inner contour")
                        cv2.drawContours(img,contours[contour_position],-1,(255,0,0),2)
                
                else:
                    pass

kernel = np.ones((1,1),np.uint8)

img = cv2.imread('helipad1.jpg')
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
imgBlur = cv2.GaussianBlur(imgBlur, (7, 7), 1)
imgBlur = cv2.GaussianBlur(imgBlur, (7, 7), 1)

# se=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))

imgCanny = cv2.Canny(imgBlur,100,150)

# imgEx =cv2.morphologyEx(imgCanny, cv2.MORPH_DILATE, se)

imgDilated = cv2.dilate(imgCanny, kernel, iterations=1)

detect_letter(imgDilated,img)

cv2.imshow("result", img)
cv2.imshow("canny", imgDilated)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 