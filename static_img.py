#Its for image testing purpose
from mtcnn import MTCNN
import cv2

detector = MTCNN()
img = cv2.imread(r"C:\Users\sahil\OneDrive\Desktop\Sahil\Projects\Image_Processing\real_mad.jpg")
#res_img = cv2.resize(img, (1200,700))

output = detector.detect_faces(img)
print(output)

for i in output:

    x,y,width,height = i['box']
    cv2.rectangle(img, pt1=(x,y), pt2=(x+width, y+height), color=(255,0,0), thickness=2)

    leye_x, leye_y = i['keypoints']['left_eye']
    reye_x, reye_y = i['keypoints']['right_eye']
    nose_x, nose_y = i['keypoints']['nose']
    lmouth_x, lmouth_y= i['keypoints']['mouth_left']
    rmouth_x, rmouth_y= i['keypoints']['mouth_right']

    cv2.circle(img, center=(leye_x,leye_y), color=(0,255,0), thickness=2, radius=5)
    cv2.circle(img, center=(reye_x,reye_y), color=(0,255,0), thickness=2, radius=5)
    cv2.circle(img, center=(nose_x,nose_y), color=(0,0,255), thickness=2, radius=3)
    cv2.circle(img, center=(lmouth_x,lmouth_y), color=(0,255,0), thickness=2, radius=3)
    cv2.circle(img, center=(rmouth_x,rmouth_y), color=(0,255,0), thickness=2, radius=3)

cv2.imshow('window',img)
cv2.waitKey(0)
cv2.destroyAllWindows()