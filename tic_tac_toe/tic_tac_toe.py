# from ..utils import BASE_DIR
import numpy as np
from tqdm import trange
from time import sleep

class Player:
    """
    This class defines the player
    """
    def __init__(self) -> None:
        self.player_id = None
        self.player_val = None
        self.state_values = None
        self.moves = []
    
    def move(self, states, greedy_rate):      
        row = -1
        col = -1

        # Uniform distribution in [0, 1]
        if np.random.rand(1)[0] > greedy_rate:
          available_states = np.argwhere(states == 0)
          rand_index = np.random.randint(0, available_states.shape[0])

          row = available_states[rand_index, 0]
          col = available_states[rand_index, 1]
        
        # By max probability of state values
        else:
            # BAD SOLUTION
            # Zero trick
            zeros = np.zeros_like(self.state_values)
            joint_array = np.where(states == 0, self.state_values, zeros)
            max_idx = np.argmax(joint_array)
            row, col = np.unravel_index(max_idx, shape=joint_array.shape)

        # Change state according to player movement
        states[row, col] = self.player_val

        # Add new state location to moves
        self.moves.append((row, col))
      
    def backward(self, learning_rate: float = 1e-2):
        row, col = self.moves[-1]
        self.state_values[row, col] += learning_rate

        next_row, next_col = row, col
        for i in range(len(self.moves) - 2, -1, -1):
            row, col = self.moves[i]
            self.state_values[row, col] += learning_rate * \
              (self.state_values[next_row, next_col] - self.state_values[row, col])
            
            next_row, next_col = row, col

    def print_state(self):
        for i in range(self.state_values.shape[0]):
          print('-'*22)
          print('|', end='')
          for j in range(self.state_values.shape[1]):
              print(' {:.2f} |'.format(self.state_values[i, j]), end='')
          print()
        print('-'*22)
        print()

class TicTacToeBoard:
    """
    Implementation of game "Tic-Tac-Toe"
    """
    def __init__(self,
                 n_rows: int = 3,
                 n_cols: int = 3,
                 greedy_rate: float = 0.3) -> None:
        """
        n_rows: int -- the number of rows in game
        n_cols: int -- the number of columns in game
        """

        self.player1 = Player()
        self.player2 = Player()

        self.player1.player_id = 1
        self.player2.player_id = 1

        self.player1.player_val = 1
        self.player2.player_val = -1

        self.n_rows = n_rows
        self.n_cols = n_cols
        self.greedy_rate = greedy_rate

        # reset game
        self.reset_game(player_state=True)

    def reset_game(self,
                   player_state: bool = False) -> None:
        # board
        # We assumse 1, -1 and 0 indicate the players for 1st, 2nd and empty place
        self.states = np.zeros((self.n_rows, self.n_cols), dtype=int)
        self.player1.moves.clear()
        self.player2.moves.clear()

        if player_state:
            # state values
            self.player1.state_values = np.zeros((self.n_rows, self.n_cols)) + 0.5
            self.player2.state_values = np.zeros((self.n_rows, self.n_cols)) + 0.5

    @property
    def is_end(self) -> bool:
        return np.all(self.states != 0)
    
    @property
    def who_won(self) -> Player:
        
        # By row winning
        row_sums = np.sum(self.states, axis=1)
        # print('row_sums: ', row_sums)
        if self.n_rows in row_sums:
            return self.player1
        if -self.n_rows in row_sums:
            return self.player2
        
        # By column winning
        col_sums = np.sum(self.states, axis=0)
        # print('col_sums: ', col_sums)
        if self.n_cols in col_sums:
            return self.player1
        if -self.n_cols in col_sums:
            return self.player2
        
        # By diagonal winning
        # it is square board
        if self.n_cols == self.n_rows:
            # By primary daigonal
            s1 = self.states.trace()
            s2 = self.states[::-1].trace()
            # print('diagonal: ', s1, s2, self.n_cols)

            if s1 == self.n_cols or s2 == self.n_cols:
                return self.player1
            elif s1 == -self.n_cols or s2 == -self.n_cols:
                return self.player2

            
        return None

    def print_board(self):
        for i in range(self.n_rows):
          print('-'*13)
          print('|', end='')
          for j in range(self.n_rows):
              s = ' '
              if self.states[i, j] == 1:
                  s = 'X'
              elif self.states[i, j] == -1:
                  s = 'O'
              
              print(f' {s} |', end='')
          print()
        print('-'*13)
        print()

    def train_game(self,
                 num_episodes: int = 100):
        # The first player always start moving firstly then the second player
        for episode in range(num_episodes):
            print(f'Episode: {episode}')
        # for episode in trange(num_episodes, desc="Episode: "):
            self.print_board()
            # Forward
            while not self.is_end:
                
                # The first player's turn
                # new state locations
                self.player1.move(states=self.states, 
                                  greedy_rate=self.greedy_rate)
                # print(self.states)
                self.print_board()

                # Backward
                winner = self.who_won
                if winner:
                    # print('ishladi')
                    winner.backward(learning_rate=0.1)
                    self.reset_game()
                    break
                
                if not self.is_end:
                    self.player2.move(states=self.states, 
                                  greedy_rate=self.greedy_rate)
                    self.print_board()
                    # print(self.states)
            
                # Backward
                winner = self.who_won
                if winner:
                    # print('ishladi')
                    winner.backward(learning_rate=0.1)
                    self.reset_game()
                    break
                
            self.player1.print_state()
            self.player2.print_state()

    def play(self):
        self.reset_game()          
        print('Now, let\'s play')
        turn = ''
        answers = ['yes', 'no']
        while turn.lower() not in answers:
            self.print_board()
            turn = input("Would you be the first or second player. (Yes or No): ")

        if turn.lower() == answers[0]:
            player = self.player1
            ai = self.player2
            winner = self.who_won
            while winner is None:
                print("Your movement")
                row = int(input('Next row: ')) - 1
                col = int(input('Next col: ')) - 1
                while self.states[row, col] != 0:
                    print('Please select empty cell...')
                    row = int(input('Next row: ')) - 1
                    col = int(input('Next col: ')) - 1

                # Player movement, just make state as X
                self.states[row, col] = player.player_val
                self.print_board()
                winner = self.who_won
                if winner is not None and winner == player:
                    print("You won, congratulations.....")
                    break
                
                # AI movement
                print("AI movement")
                ai.move(states=self.states, greedy_rate=1.1)
                self.print_board()
                winner = self.who_won
                if winner is not None and winner == ai:
                    print("AI won")
                    break
                
        else:
            player = self.player2
            ai = self.player1
            winner = self.who_won
            while winner is None:
                # AI movement
                print("AI movement")
                ai.move(states=self.states, greedy_rate=1.1)
                self.print_board()
                winner = self.who_won
                if winner is not None and winner == ai:
                    print("AI won")
                    break

                print("Your movement")
                row = int(input('Next row: ')) - 1
                col = int(input('Next col: ')) - 1
                while self.states[row, col] != 0:
                    print('Please select empty cell...')
                    row = int(input('Next row: ')) - 1
                    col = int(input('Next col: ')) - 1

                # Player movement, just make state as X
                self.states[row, col] = player.player_val
                self.print_board()
                winner = self.who_won
                if winner is not None and winner == player:
                    print("You won, congratulations.....")
                    break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eps-greedy', type=float, default=0.3)
    parser.add_argument('--num-episodes', type=int, default=300)
    args = parser.parse_args()

    game = TicTacToeBoard(greedy_rate=args.eps_greedy)
    game.train_game(num_episodes=args.num_episodes)
    game.play()