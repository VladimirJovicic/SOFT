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

    return resNovi,res2

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


def detekcijaHorizontalnihLinija(res):
    kernelx = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))

    dx = cv2.Sobel(res, cv2.CV_16S, 1, 0)
    dx = cv2.convertScaleAbs(dx)
    cv2.normalize(dx, dx, 0, 255, cv2.NORM_MINMAX)
    ret, close = cv2.threshold(dx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernelx, iterations=1)

    _, contour, hierarch = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x, y, w, h = cv2.boundingRect(cnt)
        if h / w > 5:
            cv2.drawContours(close, [cnt], 0, 255, -1)
        else:
            cv2.drawContours(close, [cnt], 0, 0, -1)
    close = cv2.morphologyEx(close, cv2.MORPH_CLOSE, None, iterations=2)
    closex = close.copy()
    return closex

def detekcijaVertikalnihLinija(res):
    kernely = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    dy = cv2.Sobel(res, cv2.CV_16S, 0, 2)
    dy = cv2.convertScaleAbs(dy)
    cv2.normalize(dy, dy, 0, 255, cv2.NORM_MINMAX)
    ret, close = cv2.threshold(dy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, kernely)

    _, contour, hierarch = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        x, y, w, h = cv2.boundingRect(cnt)
        if w / h > 5:
            cv2.drawContours(close, [cnt], 0, 255, -1)
        else:
            cv2.drawContours(close, [cnt], 0, 0, -1)

    close = cv2.morphologyEx(close, cv2.MORPH_DILATE, None, iterations=2)
    closey = close.copy()
    return closey

def dodajKoordinatePreseka(res,img):
    _, contour, hierarch = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contour:
        mom = cv2.moments(cnt)
        (x, y) = int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00'])
        # ovo sluzi samo za preoveru
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        centroids.append((x, y))
    return centroids

def setuj_i_sortiraj(centroids):
    centroids = np.array(centroids, dtype=np.float32)
    c = centroids.reshape((100, 2))
    c2 = c[np.argsort(c[:, 1])]

    b = np.vstack([c2[i * 10:(i + 1) * 10][np.argsort(c2[i * 10:(i + 1) * 10, 0])] for i in xrange(10)])
    bm = b.reshape((10, 10, 2))
    return bm,b

def kreirajMatricu(b,bm,res2):
    output = np.zeros((450, 450, 3), np.uint8)
    for i, j in enumerate(b):
        ri = i / 10
        ci = i % 10
        if ci != 9 and ri != 9:
            src = bm[ri:ri + 2, ci:ci + 2, :].reshape((4, 2))
            dst = np.array([[ci * 50, ri * 50], [(ci + 1) * 50 - 1, ri * 50], [ci * 50, (ri + 1) * 50 - 1],
                            [(ci + 1) * 50 - 1, (ri + 1) * 50 - 1]], np.float32)
            retval = cv2.getPerspectiveTransform(src, dst)
            warp = cv2.warpPerspective(res2, retval, (450, 450))
            output[ri * 50:(ri + 1) * 50 - 1, ci * 50:(ci + 1) * 50 - 1] = warp[ri * 50:(ri + 1) * 50 - 1,
                                                                           ci * 50:(ci + 1) * 50 - 1].copy()
            #prikaziSliku(output)
    return output


def prikaziSliku(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

