import numpy as np

def print_sudoku(puzzle):
    for i in range(0, 9):
        red = []
        for j in range(0, 9):
            red.append(puzzle[i][j])
        print(red)

def validno(i,j,val,puzzle):
    #ako se broj moze staviti na dato polje funkcija vraca True
    #ako se ne moze staviti na dato polje, funkcija vraca False

    #proverava da li se broj nalazi u istom redu
    for k in range(0,8):
        if val == puzzle[k][j]:
            return False

    #proverava da li se broj nalazi u istoj koloni
    for k in range(0,8):
        if val == puzzle[i][k]:
            return False

    #proverava da li je broj u istom kvadratu 3x3
    kvadratRed    = (i // 3) * 3
    kvadratKolona = (j // 3) * 3
    for k in range(0,3):
        for m in range(0,3):
            if val == puzzle[kvadratRed + k][kvadratKolona + m]:
                return False

    return True

def solve(i,j,puzzle):
    if i==9:
        i = 0
        j = j + 1
        if j==9:
            return True

    if puzzle[i][j] != 0:
        return solve(i+1,j,puzzle)

    for val in range(1,10):
        #print(val)
        if validno(i,j,val,puzzle) == True:
            puzzle[i][j] = val
            #print(puzzle)
            #print('\n')
            if solve(i+1,j,puzzle):
                return True

    puzzle[i][j] = 0
    return False
