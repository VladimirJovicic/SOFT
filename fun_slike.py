import cv2
import numpy as np

def ucitajSliku(putanja):
    img = cv2.imread(putanja)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((gray.shape), np.uint8)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1)
    div = np.float32(gray) / (close)
    res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
    res2 = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)

    resNovi = izolujMatricu(res, mask)

    return resNovi

def izolujMatricu(res, mask):
    thresh = cv2.adaptiveThreshold(res, 255, 0, 1, 201, 2)
    _, contours, hierarch = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = cnt

    cv2.drawContours(mask, [best_cnt], 0, 255, -1)
    cv2.drawContours(mask, [best_cnt], 0, 0, 0)
    res = cv2.bitwise_and(res, mask)
    return res

def prikaziSliku(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()