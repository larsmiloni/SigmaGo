import numpy as np
import govars
import subprocess


class GoGame:
    def __init__(self, size: int = 9):
        self.size = size
        self.shape = (self.size, self.size)

        """
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
        self.state = np.zeros((govars.NUM_LAYERS, self.size, self.size))
        self.history = []  # To track past board states (for Ko rules)
        self.sgf_moves = ""
        self.num_moves = 0

    def reset(self):
        self.state = np.zeros((govars.NUM_LAYERS, self.size, self.size))
        self.history = []
        self.sgf_moves = ""
        self.num_moves = 0

    def step(self, action):
        if self.get_turn() == 1:
            color = "B"
        else:
            color = "W"


        self.get_invalid_moves()
        if action not in self.get_legal_actions():
            raise ValueError("Illegal move.")

        self.num_moves += 1

        # Used for updating state
        previous_move_was_pass = np.max(self.state[govars.PASS] == 1) == 1

        # Handle pass action
        if action == "pass":
            self.state[govars.PASS] = 1
            print("pass")
            self.update_state()  # Switch turns
            
            # Add pass move to the SGF moves string
            self.sgf_moves += f";{color}[]"

            # End the game if both players pass consecutively
            if previous_move_was_pass:
                print("Game over due to consecutive passes.")
                self.state[govars.DONE] = 1
                return self.get_board(), 0, True

            return self.get_board(), 0, False  # No reward for passing, game not over

        self.state[govars.PASS] = 0
        x, y = action

        # Add the move to the SGF moves string
        self.sgf_moves += f";{color}[{chr(x + 97)}{chr(y + 97)}]"

        self.get_board()[y, x] = self.get_turn()
        captured_stones = self.check_captures(y, x)  # Capture logic
        self.history.append(self.get_board().copy())
        self.update_state()

        # Reward is number of captured stones
        return self.get_board(), len(captured_stones), False


    def get_invalid_moves(self):
        # Get invalid moves
        valid_moves = self.get_legal_actions()
        valid_moves.remove("pass")
        invalid_moves = np.ones(self.shape)
        for move in valid_moves:
            i, j = move
            invalid_moves[j][i] = 0
        self.state[govars.INVD] = invalid_moves

    def get_turn(self):
        # Add 1 to keep range between 1 = black and 2 = white for old methods
        return np.max(self.state[govars.TURN]) + 1


    """Updates the layer of the color whose turn it is. Then updates/switches turn."""
    def update_state(self):
        if self.get_turn() == 1:
            black_channel = (self.get_board() == govars.BLACK_PCS).astype(int)
            self.state[govars.BLACK] = black_channel
        else:
            white_channel = (self.get_board() == govars.WHITE_PCS).astype(int)
            self.state[govars.WHITE] = white_channel

        self.state[govars.TURN] = 1 - self.state[govars.TURN]

    def get_board(self):
        return self.state[govars.BOARD]


    """Get all legal moves on the board. Pass is always a legal move."""
    def get_legal_actions(self):
        legal_moves = []
        size = self.size

        for i in range(size):
            for j in range(size):
                if self.get_board()[i, j] == 0:
                    # Temporarily place a stone to check if it results in a legal move
                    self.get_board()[i, j] = self.get_turn()
                    captured = self.check_captures(i, j)

                    # Check if placing this stone results in a self-capture or captures opponent stones and that the move
                    if (not self.is_self_capture((i, j)) or captured) and not self.is_ko():
                        legal_moves.append((j, i))

                    # Reset the board after checking
                    self.get_board()[i, j] = 0
                    for cx, cy in captured:
                        self.get_board()[cx, cy] = 3 - self.get_turn()

        # If no legal moves are available except pass
        if not legal_moves:
            return ["pass"]

        # Add the pass action as a legal move
        legal_moves.append("pass")
        return legal_moves


    """Check that move does not repeat a previous board instance."""
    def is_ko(self):
        for board in self.history:
            if np.array_equal(board, self.get_board()):
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
        opponent = 3 - self.get_turn()
        for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if 0 <= nx < self.size and 0 <= ny < self.size and self.get_board()[nx, ny] == opponent:
                opponent_group = self.get_group((nx, ny))
                if not any(self.hasLiberties(stone) for stone in opponent_group):
                    return False

        # If no liberties and no opponent captures, it's a self-capture
        return True


    """Get all stones connected to the given intersection (same color)."""
    def get_group(self, intersection):
        x, y = intersection
        color = self.get_board()[x, y]
        visited = set()
        group = []

        def dfs(px, py):
            if (px, py) in visited:
                return
            if not (0 <= px < len(self.get_board()) and 0 <= py < len(self.get_board())):
                return
            if self.get_board()[px, py] != color:
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
        size = len(self.get_board())

        for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if 0 <= nx < size and 0 <= ny < size and self.get_board()[nx][ny] == 0:
                return True

        return False


    """Check and capture any opponent groups that have no liberties."""
    def check_captures(self, x, y):
        opponent = 3 - self.get_turn()
        captured_stones = []

        # Check all four adjacent positions for opponent stones
        for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if 0 <= nx < len(self.get_board()) and 0 <= ny < len(self.get_board()) and self.get_board()[nx, ny] == opponent:
                group = self.get_group((nx, ny))
                if not any(self.hasLiberties(stone) for stone in group):
                    captured_stones.extend(group)
                    for gx, gy in group:
                        self.get_board()[gx, gy] = 0  # Remove captured stones

        return captured_stones

    def render_in_terminal(self):
        print("  0 1 2 3 4 5 6 7 8")
        for i, row in enumerate(self.get_board()):
            row_print = str(i) + " "
            row_print += '─'.join(['┼' if cell == 0
                                   else ('○' if cell == 1 else '●') for cell in row])
            print(row_print)


    """Convert SGF move notation to a list of (row, col) coordinates."""
    def sgf_to_coordinates(self, sgf_moves):
        moves = []
        for entry in sgf_moves.split(';'):
            if not entry:
                continue
            # Determine player and move
            player = entry[0]  # 'B' for Black or 'W' for White
            move = entry[2:4]  # Get the two-letter move, e.g., 'fe'

            # Convert letters to board coordinates
            row = ord(move[0]) - ord('a')  # Vertical position (y-coordinate)
            col = ord(move[1]) - ord('a')  # Horizontal position (x-coordinate)

            # Append to moves list as (player, (row, col))
            moves.append((row, col))

        return moves
    
    def write_to_sgf(self, komi):
        sgf_string = f"(;GM[1]SZ[9]KM[{komi-0.5}]RU[Chinese]\nPB[player1 (1)]\nPW[player2 (1)]\n{self.sgf_moves})"

        f = open("game.sgf", "w")
        f.write(sgf_string)
        f.close()


    """Returns 0 if black wins, and  if white wins"""
    def determine_winner(self, komi):
        self.write_to_sgf(komi)
        winner_str = subprocess.run(
            f"gnugo --score estimate --quiet -L {self.num_moves} -l game.sgf", shell=True, capture_output=True, text=True)

        print(winner_str.stdout)

        if winner_str.stderr:
            raise EnvironmentError

        if "Black" in winner_str.stdout:
            return 0
        else:
            return 1


game = GoGame(size=7)
game.reset()

game.step((0, 1))
game.step((0, 2))

game.step((1, 1))
game.step((1, 2))

game.step((2, 1))
game.step((2, 2))

game.step((3, 1))
game.step((3, 2))

game.step((4, 1))
game.step((3, 3))

game.step((4, 2))
game.step((4, 3))

game.step((5, 2))
game.step((5, 3))

game.step((6, 2))
game.step((6, 3))

game.step((0, 5))
game.step((5, 4))

game.step((1, 5))
game.step((5, 5))

game.step((2, 5))
game.step((5, 6))

game.step((3, 5))
game.step((6, 5))

game.step((3, 6))
game.step((0, 3))

game.step((1, 6))
game.step((3, 0))

game.step((1, 0))
game.step((0, 4))

game.step((5, 0))
game.step((1, 4))

game.step((5, 1))
game.step((2, 4))

game.step((6, 0))
game.step((3, 4))

game.step(("pass"))
game.step((4, 4))

game.step(("pass"))
game.step((4, 5))

game.step(("pass"))

#print(game.sgf_moves)

#game.write_to_sgf(7.5)

game.render_in_terminal()
print(game.determine_winner(7.5))

#game.render_in_terminal()
#game.end_game()
#print(game.get_legal_actions())
#print("GROOO:", game.get_group((1, 3)))