import cv2
import fun_slike
import sudoku
from keras.models import model_from_json
import numpy as np
import fun_NM


putanja = 'slike\\test_3.jpg'
img = cv2.imread(putanja)
res,res2 = fun_slike.ucitajSliku(putanja)
#fun_slike.prikaziSliku(res)

closex = fun_slike.detekcijaHorizontalnihLinija(res)
#slike_fun.prikaziSliku(closex)

closey = fun_slike.detekcijaVertikalnihLinija(res)
#slike_fun.prikaziSliku(closey)

#pravi presek
res = cv2.bitwise_and(closex,closey)
#slike_fun.prikaziSliku(res)


centroids = fun_slike.dodajKoordinatePreseka(res,img)
#slike_fun.prikaziSliku(img)

stekovani_r,stekovani = fun_slike.setuj_i_sortiraj(centroids)

output,niz = fun_slike.kreirajMatricu(stekovani,stekovani_r,res2)
fun_slike.prikaziSliku(output)

########################################################
deloviSlike = fun_slike.razbiSlikuNaKvadrate(output)
#for i in range(0, 9):
   # for j in range(0, 9):
        #fun_slike.prikaziSliku(deloviSlike[i][j])

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ann = model_from_json(loaded_model_json)
ann.load_weights("model.h5")

alphabet = [0,1,2,3,4,5,6,7,8,9]
sudokuMatrica = []
for i in range(0, 9):
    inputs = fun_NM.prepare_for_ann(deloviSlike[i])     #pripremanje za neuronsku mrezu matrice red po red
    results = ann.predict(np.array(inputs, np.float32))

    redMatrice = []

    #print(result)
    for result in results:
        winner = max(enumerate(result), key = lambda x: x[1])[0]
        redMatrice.append(alphabet[winner])

    sudokuMatrica.append(redMatrice)

sudoku.print_sudoku(sudokuMatrica)
ulaznaMatrica = sudokuMatrica
if sudoku.solve(0,0,sudokuMatrica) == True:
   sudoku.print_sudoku(sudokuMatrica)
   fun_slike.iscrtajBrojeveNaSliku(ulaznaMatrica,sudokuMatrica,output)

else:
    print('Nesto nije u redu sa maticom')

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