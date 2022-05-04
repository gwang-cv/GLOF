import scipy.io
import time
import cv2
from GLOF import GLOF_filter

def draw_match(img1, img2, corr1, corr2):
    corr1 = [cv2.KeyPoint(corr1[i, 0], corr1[i, 1], 1) for i in range(corr1.shape[0])]
    corr2 = [cv2.KeyPoint(corr2[i, 0], corr2[i, 1], 1) for i in range(corr2.shape[0])]
    assert len(corr1) == len(corr2)
    draw_matches = [cv2.DMatch(i, i, 0) for i in range(len(corr1))]
    display = cv2.drawMatches(img1, corr1, img2, corr2, draw_matches, None,
                              matchColor=(0, 255, 0),
                              singlePointColor=(0, 0, 255),
                              flags=4
                              )
    return display

if __name__ == "__main__":
    data = scipy.io.loadmat('church.mat') 
    X = data['X']  # n 2
    Y = data['Y']
    I1 = data['I1']  # w h 3
    I2 = data['I2']
    start = time.clock()
    mask = GLOF_filter(X, Y)
    end = time.clock()
    print("Time cost: {} seconds".format(end-start))
    print("Correspondences before GLOF filter, please press ESC to terminate image window")
    display = draw_match(I1, I2, X, Y)
    cv2.imshow("before", display)
    # press ESC to terminate imshow
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()  
    print("Correspondences after GLOF filter, please press ESC to terminate image window")
    display2 = draw_match(I1, I2, X[mask,:], Y[mask,:])
    cv2.imshow("after", display2)
    # press ESC to terminate imshow
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows() 