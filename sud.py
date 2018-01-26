import cv2
import numpy as np
import fun_mat as mat_fun
import fun_slike as slike_fun



putanja = 'slike\proba.png'
img = cv2.imread(putanja)
res,res2 = slike_fun.ucitajSliku(putanja)

#slike_fun.prikaziSliku(res)

closex = slike_fun.detekcijaHorizontalnihLinija(res)
slike_fun.prikaziSliku(closex)

closey = slike_fun.detekcijaVertikalnihLinija(res)
slike_fun.prikaziSliku(closey)

#pravi presek
res = cv2.bitwise_and(closex,closey)
slike_fun.prikaziSliku(res)


centroids = slike_fun.dodajKoordinatePreseka(res,img)
slike_fun.prikaziSliku(img)

bm,b = slike_fun.setuj_i_sortiraj(centroids)

output = slike_fun.kreirajMatricu(b,bm,res2)
slike_fun.prikaziSliku(output)