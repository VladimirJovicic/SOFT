import cv2
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw



def prikaziSliku(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def ucitajSliku(putanja):
    img = cv2.imread(putanja)   #ucita sliku sa prosledjene putanje
    # bluruje sliku - bude onako smooth mutna
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # menja RGB u BGR i postaje crno-bela slika
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # kreira masku - niz koji je dimenzija crno bele slike
    mask = np.zeros((gray.shape), np.uint8)
    # pretstavlja elipse na osnovu kojih se detektuju oblasti
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # detektuje one "izrazenije" oblasti koje su se odredile na osnovu kernela
    # i uklanja sumove
    close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel1)
    # slika postaje crno bela bez nijansi sive
    div = np.float32(gray) / (close)
    res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
    # treba jos obraditi sliku pa vracamo i res i res2, zato vracamo res2 a res jos obradjujemo
    res2 = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    resNovi = izolujMatricu(res, mask)

    return resNovi,res2

def izolujMatricu(res, mask):
    # konvertuje sve u crno belo za lakse izolovanje
    # brojevi 0,1,201 i 2 su eksperimentalno dobijeni
    thresh = cv2.adaptiveThreshold(res, 255, 0, 1, 201, 2)
    # pronalazi sve konture u obliku cetvorougla
    # smesta u pomenljivu contours, a ove prazne promenljive
    # su tu zato sto funkcija fintContours mora da ima 3 izlaza
    # i smesta u srednju promenljivu
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # pretpostavka : najveci kvadrat na slici pretstavlja sudoku matricu
    max_area = 0
    best_cnt = None
    for cnt in contours:
        # pretrazuje i gleda svaku konturu posebno
        area = cv2.contourArea(cnt)
        # pretpostavka da ce matrica da zauzima veci deo slike
        if area > 1000:
            if area > max_area:
                max_area = area
                best_cnt = cnt

    # crta konture : na masku(crnu pozadinu) stavlja najveci kvadrat koji je preuzet
    # prvi parametar je izvorna slika, drugi parametar pretstavlja konture koje se iscrtavaju
    # treci je index konture (posto imamo samo jednu, prosledjuje se 0 (da zelimo iscrtati sve konture
    #   stavi se -1)
    # ovo ostalo treba da su za boje i to, ne radi kad se unese nesto drugo od 255 i -1
    cv2.drawContours(mask, [best_cnt], 0, 255, -1)

    # u masci se nalazi beli cetvorougao koji je dimenzija sudoku matrice
    # potrebno je samo jos sa AND operacijom da se puste res i mask i kao rezultat
    #   se izbacuje izdvojena sudoku matrica sa slike na crnoj pozadini
    res = cv2.bitwise_and(res, mask)
    return  res


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
    i = 0
    _, contour, hierarch = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    for cnt in contour:
        mom = cv2.moments(cnt)
        (x, y) = int(mom['m10'] / mom['m00']), int(mom['m01'] / mom['m00'])
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        centroids.append((x, y))
    return centroids


def setuj_i_sortiraj(centroids):
    centroids = np.array(centroids, dtype=np.float32)
    centri = centroids.reshape((100, 2))
    sortirani_centri = centri[np.argsort(centri[:, 1])]

    stekovani = np.vstack([sortirani_centri[i * 10:(i + 1) * 10][np.argsort(sortirani_centri[i * 10:(i + 1) * 10, 0])] for i in range(10)])
    stekovani_r = stekovani.reshape((10, 10, 2))
    return stekovani_r,stekovani

def kreirajMatricu(b,bm,res2):
    niz = []
    output = np.zeros((450, 450, 3), np.uint8)
    for i, j in enumerate(b):
       # pred_int("j = ",j)
        red_i = i // 10
        kolona_i = i % 10
        #pred_int('int(red_i) = ',int(red_i))
        #pred_int(kolona_i)
        if kolona_i != 9 and red_i != 9:
            src = bm[red_i:red_i + 2, kolona_i:kolona_i + 2].reshape((4, 2))
            #pred_int(i, '\n', src)
            dst = np.array([[kolona_i * 50, red_i * 50], [(kolona_i + 1) * 50 - 1, red_i * 50], [kolona_i * 50, (red_i + 1) * 50 - 1],
                            [(kolona_i + 1) * 50 - 1, (red_i + 1) * 50 - 1]], np.float32)
            retval = cv2.getPerspectiveTransform(src, dst)
            warp = cv2.warpPerspective(res2, retval, (450, 450))
            output[int(red_i) * 50:(int(red_i) + 1) * 50 - 1, int(kolona_i) * 50:(int(kolona_i) + 1) * 50 - 1] = warp[int(red_i) * 50:(int(red_i) + 1) * 50 - 1,
                                                                           int(kolona_i) * 50:(int(kolona_i) + 1) * 50 - 1].copy()
            niz.append(warp[int(red_i) * 50:(int(red_i) + 1) * 50 - 1,int(kolona_i) * 50:(int(kolona_i) + 1) * 50 - 1].copy())
            #pred_ikaziSliku(output)
    return output,niz


def razbiSlikuNaKvadrate(img):
    # Creates a list containing 9 lists, each of 9 items, all set to 0
    w, h = 9, 9;
    Matrix = [[0 for x in range(w)] for y in range(h)]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for i in range(0, 9):
        for j in range(0, 9):
            img_crop = img[i*50:(i+1)*50, j*50:(j+1)*50]
            #img_crop = resize_region(img_crop)
            Matrix[i][j] = img_crop

    return Matrix

def iscrtajBrojeveNaSliku(ulazna_matrica, resena_matrica, slika):
    image = Image.fromarray(slika, 'RGB')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('arial.ttf', 40)
    for i in range(0, 9):
        for j in range(0, 9):
            # draw.text((i*50 + 10, j*50), "1", (i*30, i*j, j*20),font = font)
            if ulazna_matrica[j][i] == 0:
                #print(ulazna_matrica[i][j])
                draw.text((i * 50 + 10, j * 50), str(resena_matrica[j][i]), (0, 255, 0), font=font)

    konacna_slika = np.array(image)
    # kovertovanje RGB i BGR
    konacna_slika = konacna_slika[:, :, ::-1].copy()
    prikaziSliku(konacna_slika)
