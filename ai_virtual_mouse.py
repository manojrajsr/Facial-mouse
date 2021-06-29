import cv2
import dlib
import numpy as np
from pynput.mouse import Controller, Button
import time
from scipy.spatial import distance
from imutils import face_utils
import wx


app = wx.App()
mouse = Controller()
#------------------------------------variables---------------------------------------------
wcam, hcam = 640, 480
ptime=0
wscreen, hscreen = wx.GetDisplaySize()
wfr = 220 #frame reduction
hfr = 160
smoothening = 2
pre_x = 650 #previous mouse locations
pre_y = 500
cuc_x = 0 #current mouse locations
cuc_y = 0

#------------------------------------------------------------------------------------------

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(r"C:\Users\manoj\OneDrive\Pictures\Camera Roll\test.mp4")


time.sleep(1.0)

cap.set(3, wcam)
cap.set(4, hcam)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #for points




def calculate_EAR(eye):                              #for calculating the ratio EAR
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio

#-------------------------------------main-----------------------------------------



while True:
    #finding points
    _, frame = cap.read()
    frame = cv2.resize(frame, (wcam,hcam), interpolation = cv2.INTER_AREA)
    
    gray = frame #cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) ----- do not use gray image, color image gives more accuracy
    faces = detector(gray)   # for detecting face

#----------------------------------frame reduction-----------------
    cv2.rectangle(frame, (wfr,hfr), (wcam-wfr, hcam-hfr), (255,255,0), 3)
        #cv2.circle(frame, (wcam-fr,hcam-fr),50,(255,255,255),-1)


#----------------for complete face
    
    for face in faces:
        facex1 = face.left()
        facey1 = face.top()
        facex2 = face.right()
        facey2 = face.bottom()
        cv2.rectangle(frame, (facex1,facey1), (facex2,facey2), (0,0,255), 3)   #detected face

        

        face_landmarks = predictor(gray, face)        # for all face points   ie, shape
        lefteye = []
        righteye = []


#-----------------------------for left eye-------------------


        for n in range(36,42):                   #for lefteye
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	lefteye.append((x,y))
        	next_point = n+1
        	if n == 41:
        		next_point = 36
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)



#---------------------------------for right eye----------------------



        for n in range(42,48):                     #for righteye
        	x = face_landmarks.part(n).x
        	y = face_landmarks.part(n).y
        	righteye.append((x,y))
        	next_point = n+1
        	if n == 47:
        		next_point = 42
        	x2 = face_landmarks.part(next_point).x
        	y2 = face_landmarks.part(next_point).y
        	cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

#----------------------------------------calculating ear---------------------
        leftear= calculate_EAR(lefteye)
        rightear = calculate_EAR(righteye)

        ratio = (leftear+rightear)/2
        ratio = round(ratio,2)

        if ratio<0.26:
                cv2.putText(frame,"CLICKED",(20,100), cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
        	#cv2.putText(frame,"Are you Sleepy?",(20,400),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
                mouse.click(Button.left, 2)
                print("clicked")

        print(ratio)

        


#----------------------------------for nose tip----------------------------------
        landmarks = predictor(gray, face)
        x=landmarks.part(30).x
        y=landmarks.part(30).y
        cv2.circle(frame, (x,y), 3, (0,0,255), -1)           #detected nose tip

        mx = np.interp(x, (wfr,wcam-hfr), (0,wscreen))
        my = np.interp(y, (wfr,hcam-hfr), (0,hscreen))



#------------------------------------smoothening--------------------------
        
        cuc_x = pre_x + (mx  - pre_x) / smoothening
        cuc_y = pre_y + (my  - pre_y) / smoothening


#-----------------------------------mouse control----------------------
        mouse.position = (cuc_x,cuc_y)
        print(mx, cuc_x)
        #mouse.position = (mx,my)
        pre_x = cuc_x
        pre_y = cuc_y
        


#--------------------------------------for clicking-----------------------------------

        """ print(left_ear, "  ", right_ear,"\n")
        if not(left_ear <0.23 and right_ear <0.23):
            #now clicked
            print("not clicked")

        else:
                print("clicked") """
        


    #frame rate
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(frame,str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,0),3)


    cv2.imshow("Image", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()









print("all good")
