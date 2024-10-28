import numpy as np

class GoGame:
    def __init__(self, size=9):
        self.size = size
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.turn = 1  # 1 for black, 2 for white
        self.history = []  # To track past board states (for Ko rules)

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.turn = 1
        self.history = []
        return self.board

    def get_legal_actions(self):
        legal_moves = []
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0:
                    legal_moves.append((i, j))
        legal_moves.append("pass")  # Include the option to pass
        return legal_moves

    def step(self, action):
        if action == "pass":
            self.turn = 3 - self.turn  # Switch turns
            return self.board, 0, False  # No reward for passing, game not over
        
        x, y = action
        if self.board[x, y] != 0:
            raise ValueError("Illegal move. Cell already occupied.")
        
        self.board[x, y] = self.turn
        # Capture logic would go here
        
        self.history.append(self.board.copy())
        self.turn = 3 - self.turn  # Switch turns
        return self.board, 0, self.is_game_over()  # Default reward of 0

    def is_game_over(self):
        # Implement a simple check for consecutive passes or custom logic
        return False

    def render(self):
        # Display the board in ASCII (for simplicity)
        for row in self.board:
            print(' '.join(['.' if cell == 0 else ('B' if cell == 1 else 'W') for cell in row]))

# Example Usage
game = GoGame(size=9)
game.reset()
game.render()
print(game.get_legal_actions())
game.step((4, 4))  # Place a stone in the center
game.render()
