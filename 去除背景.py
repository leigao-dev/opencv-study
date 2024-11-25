import cv2

cap=cv2.VideoCapture("highway.mp4")

mog=cv2.createBackgroundSubtractorMOG2()

kerne=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
min_w=100
min_h=90
line_high=1400
while True:
    ret,frame=cap.read()
    if ret==False:
        break
    gary=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(frame,(3,3),0)
    fgmask=mog.apply(blur)
    # cv2.namedWindow('erode',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('erode',640,480)
    # cv2.namedWindow('dilate',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('dilate',640,480)
    # cv2.namedWindow('close',cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('close',640,480)
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('frame',640,480)
    erode=cv2.erode(fgmask,kerne)
    dilate=cv2.dilate(erode,kerne,iterations=2)
    close=cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kerne)
    # cv2.imshow('erode',erode)
    # cv2.imshow('dilate',dilate)
    # cv2.imshow('close',close)
    contours,_=cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cv2.line(frame,(0,line_high),(2560,line_high),(0,255,0),2)
    for contour in contours:
       (x,y,w,h)= cv2.boundingRect(contour)
       is_valid=w>min_w and h>min_h
       if not  is_valid:
           continue
       cv2.rectangle(frame,(int(x),int(y)),(int(x+w),int(y+h)),(0,0,255),2)
       cv2.imshow('frame',frame)

    if cv2.waitKey(50) & 0xFF==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


