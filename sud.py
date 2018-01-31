import cv2
import fun_slike as slike_fun
import sudoku

putanja = 'slike\\test_1.jpg'
img = cv2.imread(putanja)
res,res2 = slike_fun.ucitajSliku(putanja)
#slike_fun.prikaziSliku(res)

closex = slike_fun.detekcijaHorizontalnihLinija(res)
#slike_fun.prikaziSliku(closex)

closey = slike_fun.detekcijaVertikalnihLinija(res)
#slike_fun.prikaziSliku(closey)

#pravi presek
res = cv2.bitwise_and(closex,closey)
#slike_fun.prikaziSliku(res)


centroids = slike_fun.dodajKoordinatePreseka(res,img)
slike_fun.prikaziSliku(img)

bm,b = slike_fun.setuj_i_sortiraj(centroids)

output,niz = slike_fun.kreirajMatricu(b,bm,res2)
slike_fun.prikaziSliku(output)

'''
puzzle = [  (5,3,0,0,7,0,0,0,0),
            (6,0,0,1,9,5,0,0,0),
            (0,9,8,0,0,0,0,6,0),
            (8,0,0,0,6,0,0,0,3),
            (4,0,0,8,0,3,0,0,1),
            (7,0,0,0,2,0,0,0,6),
            (0,6,0,0,0,0,2,8,0),
            (0,0,0,4,1,9,0,0,5),
            (0,0,0,0,8,0,0,7,9),
          ]

puzzle = [  (0,7,5,0,9,0,0,0,6),
            (0,2,3,0,8,0,0,4,0),
            (8,0,0,0,0,3,0,0,1),
            (5,0,0,7,0,2,0,0,0),
            (0,4,0,8,0,6,0,2,0),
            (0,0,0,9,0,1,0,0,3),
            (9,0,0,4,0,0,0,0,7),
            (0,6,0,0,7,0,5,8,0),
            (7,0,0,0,1,0,3,9,0)
          ]
puzzle = np.array(puzzle)
if sudoku.solve(0,0,puzzle) == True:
   sudoku.print_sudoku(puzzle)
else:
    print('Nesto nije u redu sa maticom')

'''