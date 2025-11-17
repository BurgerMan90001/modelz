
import numpy as np


# example matrix
a = np.matrix('1 2; 3 4; 4 4')
class TicTacToe:
    # empty matrix
    board = np.matrix([[0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    ])
    def __init__(self, player_first: bool):
        
        pass
    
    def run(self):
        
        self.player_turn()
        
    def is_entry_empty(self, row_number: int, column_number:int) -> bool:
        return self.board[row_number, column_number] == 0
    
    def display_board(self):
        for row in self.board:
            print(row)
    def minimax(self): # recursive
        empty_entries = 0

        #for i in self.board:
            #for j in i:

        if (self.is_entry_empty(i, j)):
            empty_entries += 1
            pass
                #print(self.board[1 ,1])

        #possible_moves = 
        #self.minimax()
    # the cost of a move
    """""
    2 1 2
    1 3 1
    2 1 2
    """
    def cost(self):
        pass
    def player_turn(self):
        try:
            row_number:int = int(input("row number: "))
            column_number:int = int(input("column number: "))

            # if the entry is empty
            if (self.is_entry_empty(row_number, column_number)):
                # assign a player piece
                self.board[row_number-1, column_number-1] = 1
                self.display_board()
                self.minimax()

            else:
                print("The entry is not empty, choose another")
                self.player_turn()
                
        except ValueError:
            print("The value given was not an int")
            self.player_turn()
        except IndexError:
            print("Choice is out of range")
            self.player_turn()
            
def add_matrices():
    pass
    
#print("Via string input : \n", a, "\n\n")

tic_tac_toe = TicTacToe(True)

tic_tac_toe.run()

