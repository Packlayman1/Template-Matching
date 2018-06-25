import cv2
import numpy as np

MIN_MATCH_COUNT=30

detector=cv2.SIFT()
FLANN_INDEX_KDITREE=0
index_params = dict(algorithm = FLANN_INDEX_KDITREE, tree=5)
flann=cv2.FlannBasedMatcher(index_params,{})

trainImg = cv2.imread('TrainingData/TrainImg.jpeg',cv2.CV_LOAD_IMAGE_GRAYSCALE)
trainKP, trainDesc = detector.detectAndCompute(trainImg,None)
cam = cv2.VideoCapture(0)
while True:
    ret, img=cam.read()
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    queryKP, queryDesc= detector.detectAndCompute(qray,None)
    matches = flann.knnMatch(queryDesc,trainDesc,k=2)

    goodKP = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            goodMatch.append(m)
            
    if(len(goodKP)>MIN_MATCH_COUNT:
        tp = [trainKP.[m.trainIdx].pt for m in goodKP]
        qp = [queryKP.[m.queryIdx].pt for m in goodKP]
        tp,qp = np.float32((tp,qp))
        H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
        h,w = trainImg.shape
        pts = np.float32([[ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]])
        dsr = cv2.perspectiveTransform(pts,H)
        cv2.polylines(img,[np.int32(dst)],True,(0,255,0),5)
    else:
        print "Not Enough match found- %d/%d"%(len(goodKP),MIN_MATCH_COUNT)
    cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
