import numpy as np
import govars

class GoGame:
    def __init__(self, size: int = 9, board: np.ndarray = None):
        self.size = size
        self.shape = (self.size, self.size)
        self.shape = (self.size, self.size)
        # Empty = 0, black = 1, white = 2

        if board:
            self.board = board 
        else:
            self.board = np.zeros(self.shape, dtype=int)
        self.turn = 1  # 1 for black, 2 for white
        
        """"/home/larsmiloni/Skole/NTNU_5.semester/anvendt_maskinlæring/prosjekt/SigmaGo/goEnv.py", line 91
        State consists of the channels:
        
        0: Black pieces
        1: White pieces
        2: Turn (0 or 1)
        3: Invalid moves (0 or 1)
        4: Previous move was a pass
        5: Game over (0 or 1)
        6: Board (0, 1, 2)

        All of the channels are 9x9 arrays
        """
        self.state = np.zeros((govars.NUM_CHNLS, self.size, self.size))
        
        self.history = []  # To track past board states (for Ko rules)
        self.consecutivePasses = 0

        self.winner = 0
        
    def reset(self):
        self.board = np.zeros((self.shape), dtype=int)
        self.state = np.zeros((govars.NUM_CHNLS, self.size, self.size))
        self.turn = 1
        self.consecutivePasses = 0
        self.history = []
        return self.board

    def step(self, action):
        if action not in self.get_legal_actions():
            raise ValueError("Illegal move.")

        # Used for updating state
        previous_move_was_pass = self.consecutivePasses

        if action == "resign":
            print("resignation from:", self.turn)
            self.winner = 3 - self.turn
            self.update_state(previous_move_was_pass)
            self.state[govars.DONE_CHNL] = np.ones((self.shape))
            return self.board, 0, True
         
        # Handle pass action
        if action == "pass":
            print("pass")
            self.consecutivePasses += 1
            self.turn = 3 - self.turn  # Switch turns

            # End the game if both players pass consecutively
            if self.consecutivePasses >= 2:
                print("Game over due to consecutive passes.")
                self.isGameOver = True
                self.update_state(previous_move_was_pass)
                self.state[govars.DONE_CHNL] = np.ones((9, 9))
                return self.board, 0, True

            return self.board, 0, False  # No reward for passing, game not over

        self.consecutivePasses = 0

        x, y = action
        self.board[y, x] = self.turn
        captured_stones = self.check_captures(y, x)  # Capture logic

        self.history.append(self.board.copy())
        self.turn = 3 - self.turn  # Switch turns

        self.update_state(previous_move_was_pass)

        # Reward is number of captured stones
        return self.board, len(captured_stones), False
    
    def update_state(self, previous_move_was_pass: int):
        # Update black and white channel
        black_channel = (self.board == 1).astype(int)
        self.state[govars.BLACK_CHNL] = black_channel

        white_channel = (self.board == 2).astype(int)
        self.state[govars.WHITE_CHNL] = white_channel

        # Unsure if turn should be set before or after switching turns
        if self.turn == 1:
            self.state[govars.TURN_CHNL] = np.zeros(self.shape)
        else:
            self.state[govars.TURN_CHNL] = np.ones(self.shape)

        if previous_move_was_pass:
            self.state[govars.PASS_CHNL] = np.ones(self.shape)
        else:
            self.state[govars.PASS_CHNL] = np.zeros(self.shape)
        
        # Get invalid moves
        valid_moves = self.get_legal_actions()
        valid_moves.remove('pass')
        valid_moves.remove('resign')
        invalid_moves = np.ones(self.shape)
        for move in valid_moves:
            i, j = move
            invalid_moves[i][j] = 0
        self.state[govars.INVD_CHNL] = invalid_moves

        self.state[govars.DONE_CHNL] = np.zeros(self.shape)

        self.state[govars.BOARD_CHNL] = self.state[govars.BLACK_CHNL] + (self.state[govars.WHITE_CHNL] * 2)
        
    """Get all legal moves on the board. Pass is always a legal move."""

    def get_legal_actions(self):
        legal_moves = []
        size = self.size

        for i in range(size):
            for j in range(size):
                if self.board[i, j] == 0:
                    # Temporarily place a stone to check if it results in a legal move
                    self.board[i, j] = self.turn
                    captured = self.check_captures(i, j)

                    # Check if placing this stone results in a self-capture or captures opponent stones and that the move
                    if (not self.is_self_capture((i, j)) or captured) and not self.is_ko():
                        legal_moves.append((j, i))

                    # Reset the board after checking
                    self.board[i, j] = 0
                    for cx, cy in captured:
                        self.board[cx, cy] = 3 - self.turn

        # Add the pass action as a legal move
        legal_moves.append("pass")
        legal_moves.append("resign")
        return legal_moves

    """Check that move does not repeat a previous board instance."""

    def is_ko(self):
        for board in self.history:
            if np.array_equal(board, self.board):
                return True
        return False

    """Check if placing a stone at the given move results in self-capture."""

    def is_self_capture(self, move):
        x, y = move

        group = self.get_group((x, y))

        # If the placed stone or its group has liberties, the move is not a self-capture
        if any(self.hasLiberties(stone) for stone in group):
            return False

        # Check if placing this stone captures any opponent stones
        # If it captures at least one stone, it's not a self-capture
        opponent = 3 - self.turn
        for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == opponent:
                opponent_group = self.get_group((nx, ny))
                if not any(self.hasLiberties(stone) for stone in opponent_group):
                    return False

        # If no liberties and no opponent captures, it's a self-capture
        return True

    """Get all stones connected to the given intersection (same color)."""

    def get_group(self, intersection):
        x, y = intersection
        color = self.board[x, y]
        visited = set()
        group = []

        def dfs(px, py):
            if (px, py) in visited:
                return
            if not (0 <= px < len(self.board) and 0 <= py < len(self.board)):
                return
            if self.board[px, py] != color:
                return

            visited.add((px, py))
            group.append((px, py))

            # Explore all four directions
            for nx, ny in [(px-1, py), (px+1, py), (px, py-1), (px, py+1)]:
                dfs(nx, ny)

        dfs(x, y)
        return group

    """Check if an intersection has any empty intersections adjacent to it (has liberties)."""

    def hasLiberties(self, intersection):
        x, y = intersection
        size = len(self.board)

        for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if 0 <= nx < size and 0 <= ny < size and self.board[nx][ny] == 0:
                return True

        return False

    """Check and capture any opponent groups that have no liberties."""

    def check_captures(self, x, y):
        opponent = 3 - self.turn
        captured_stones = []

        # Check all four adjacent positions for opponent stones
        for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if 0 <= nx < len(self.board) and 0 <= ny < len(self.board) and self.board[nx, ny] == opponent:
                group = self.get_group((nx, ny))
                if not any(self.hasLiberties(stone) for stone in group):
                    captured_stones.extend(group)
                    for gx, gy in group:
                        self.board[gx, gy] = 0  # Remove captured stones

        return captured_stones

    def count_stones(self):
        black_stones = sum(1 for i in range(self.size)
                           for j in range(self.size) if self.board[i, j] == 1)
        white_stones = sum(1 for i in range(self.size)
                           for j in range(self.size) if self.board[i, j] == 2)
        return black_stones, white_stones
    




    def render_in_terminal(self):
        print("  0 1 2 3 4 5 6 7 8")
        for i, row in enumerate(self.board):
            row_print = str(i) + " "
            row_print += '─'.join(['┼' if cell == 0
                                   else ('○' if cell == 1 else '●') for cell in row])
            print(row_print)



"""
#Example Usage
game = GoGame(size=7)
game.reset()
print(game.get_legal_actions())
game.step((0, 2))
game.step((0, 1))
game.step((0, 3))
game.step((1, 2))
game.step((4, 2))
game.step((1, 3))
game.step((5, 3))
game.step((0, 4))
game.step((0, 2))
game.step((3, 3))
game.step((5, 5))
game.render_in_terminal()

# Reset game when two consecutive passes
_, _, game_over = game.step("pass")
if game_over:
    game.reset()

_, _, game_over = game.step(("pass"))
if game_over:
    game.reset()
game.render_in_terminal()
"""


"""
game = GoGame(size=9)
game.reset()

# Example SGF moves string
sgf_moves = ";B[fe];W[de];B[dd];W[cd];B[dc];W[ee];B[fd];W[ff];B[gf];W[ed];B[ec];W[ef];B[ce];W[cc];B[cb];W[bb];B[db];W[be];B[cf];W[gg];B[bf];W[bc];B[ae];W[ad];B[bd];W[fc];B[gc];W[fb];B[gb];W[cg];B[ac];W[hf];B[ge];W[he];B[gd];W[dg];B[hd];W[bg];B[ba];W[ag];B[ie];W[hg];B[if];W[ig];B[id];W[af];B[ab];W[df]"

# Convert SGF to list of moves
moves = game.sgf_to_coordinates(sgf_moves)

# Print the moves in (player, (row, col)) format
for move in moves:
    game.step(move)
    print(move)

game.render_in_terminal()
winner = game.determine_winner()
print("Winner: ", winner)

"""

game = GoGame(size=3)
game.reset()

game.step((1, 1))
game.step((0, 0))
game.step((2, 1))
game.step((1, 2))

game.render_in_terminal()

print(game.get_legal_actions())


"""
game.step((5, 0))

game.step((1, 1))
game.step((5, 1))

game.step((1, 2))
game.step((5, 2))

game.step((1, 3))
game.step((5, 3))

game.step((1, 4))
game.step((5, 4))

game.step((1, 5))
game.step((5, 5))

game.step((1, 6))
game.step((6, 5))

game.step((0, 2))
game.step((0, 6))

game.step((6, 0))
game.step((0, 5))
"""
"""
_, _, game_over = game.step("pass")
_, _, game_over = game.step("pass")

_, _, game_over = game.step("resign")

print("game over: ", game_over)

game.render_in_terminal()
#winner = game.determine_winner()
#print("Winner: ", winner)
"""