import numpy as np
import src.utils.govars as govars
import subprocess


class GoGame:
    """
    Class for the Go game environment.
    """
    def __init__(self, size: int = 9):
        """
        Initializes a new Go game with the given board size.

        Args:
            size (int): The size of the board.
        """
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
        self.isGameOver = False

    def reset(self):
        """
        Resets the game state to the initial state.
        """
        self.state = np.zeros((govars.NUM_LAYERS, self.size, self.size))
        self.history = []
        self.sgf_moves = ""

    def step(self, action: tuple):
        """
        Takes a step in the game by applying the given action.

        Args:
            action (tuple): The action to apply.
        """
        if self.get_turn() == 1:
            color = "B"
        else:
            color = "W"


        self.get_invalid_moves()
        if action not in self.get_legal_actions():
            raise ValueError("Illegal move.")
        
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
                self.isGameOver = True
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
        """
        Updates the invalid moves channel in the state.
        """
        # Get invalid moves
        valid_moves = self.get_legal_actions()
        valid_moves.remove("pass")
        invalid_moves = np.ones(self.shape)
        for move in valid_moves:
            i, j = move
            invalid_moves[j][i] = 0
        self.state[govars.INVD] = invalid_moves

    def get_turn(self):
        """
        Returns the current player's turn.
        """
        # Add 1 to keep range between 1 = black and 2 = white for old methods
        return np.max(self.state[govars.TURN]) + 1


    
    def update_state(self):
        """Updates the layer of the color whose turn it is. Then updates/switches turn."""
        if self.get_turn() == 1:
            black_channel = (self.get_board() == govars.BLACK_PCS).astype(int)
            self.state[govars.BLACK] = black_channel
        else:
            white_channel = (self.get_board() == govars.WHITE_PCS).astype(int)
            self.state[govars.WHITE] = white_channel

        self.state[govars.TURN] = 1 - self.state[govars.TURN]

    def get_board(self) -> np.ndarray:
        """
        Returns the current board state.

        Returns:
            np.ndarray: The current board state.
        """
        return self.state[govars.BOARD]


    
    def get_legal_actions(self):
        """Get all legal moves on the board. Pass is always a legal move."""
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
        """
        Check if the current board state is a Ko situation.
        """
        for board in self.history:
            if np.array_equal(board, self.get_board()):
                return True
        return False


    
    def is_self_capture(self, move):
        """Check if placing a stone at the given move results in self-capture."""
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


    
    def get_group(self, intersection):
        """Get all stones connected to the given intersection (same color)."""
        x, y = intersection
        color = self.get_board()[x, y]
        visited = set()
        group = []

        def dfs(px, py):
            """
            Recursive depth-first search to find all connected stones of the same color.

            Args:
                px (int): The x-coordinate of the current position.
                py (int): The y-coordinate of the current position.
            """
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


    
    def hasLiberties(self, intersection):
        """
        Check if an intersection has any empty intersections adjacent to it (has liberties).

        Args:
            intersection (tuple): The intersection to check.
        """

        x, y = intersection
        size = len(self.get_board())

        for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
            if 0 <= nx < size and 0 <= ny < size and self.get_board()[nx][ny] == 0:
                return True

        return False


    
    def check_captures(self, x, y):
        """
        Check and capture any opponent groups that have no liberties.

        Args:
            x (int): The x-coordinate of the move.
            y (int): The y-coordinate of the move.
        """
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
        """
        Renders the current board state in the terminal.
        """
        print("  0 1 2 3 4 5 6 7 8")
        for i, row in enumerate(self.get_board()):
            row_print = str(i) + " "
            row_print += '─'.join(['┼' if cell == 0
                                   else ('○' if cell == 1 else '●') for cell in row])
            print(row_print)


    
    def sgf_to_coordinates(self, sgf_moves):
        """
        Convert SGF move notation to a list of (row, col) coordinates.
        """
        moves = []
        for entry in sgf_moves.split(';'):
            if not entry:
                continue
            # Determine player and move
            player = entry[0]  # 'B' for Black or 'W' for White
            if entry[1:3] == "[]":
                move = "pass"
            else:
                move = entry[2:4]  # Get the two-letter move, e.g., 'fe'

                # Convert letters to board coordinates
                row = ord(move[0]) - ord('a')  # Vertical position (y-coordinate)
                col = ord(move[1]) - ord('a')  # Horizontal position (x-coordinate)

                move = (row, col)

            moves.append(move)

        return moves
    
    def write_to_sgf(self, komi):
        sgf_string = f"(;GM[1]SZ[9]KM[{komi}]RU[Chinese]\nPB[player1 (1)]\nPW[player2 (1)]\n{self.sgf_moves})"

        f = open("tempGame.sgf", "w")
        f.write(sgf_string)
        f.close()

    
    def determine_winner(self, komi = govars.KOMI):
        """Returns 0 if black wins, and  if white wins"""
        self.write_to_sgf(komi)
        winner_str = subprocess.run(
            f"gnugo --chinese-rules --score aftermath --quiet -l tempGame.sgf", shell=True, capture_output=True, text=True)

        print(winner_str.stderr)
        print(winner_str.stdout)

        if "error" in winner_str.stderr.lower():
            print("Error occurred while running gnugo:", winner_str.stderr)
            raise EnvironmentError("GnuGo encountered an error: " + winner_str.stderr)

        if "Black" in winner_str.stdout:
            return 0
        else:
            return 1

game = GoGame(size=9)
game.reset()

moves = game.sgf_to_coordinates(";B[fe];W[cf];B[de];W[ce];B[df];W[cc];B[dd];W[cg];B[dc];W[cb];B[cd];W[bd];B[be];W[bc];B[dg];W[ch];B[dh];W[he];B[hc];W[hd];B[gc];W[gh];B[bf];W[bg];B[ci];W[ae];B[hg];W[gg];B[hh];W[hf];B[id];W[ie];B[fi];W[ih];B[ig];W[hi];B[ii];W[fh];B[ei];W[ih];B[bh];W[af];B[ii];W[gd];B[gi];W[fc];B[fb];W[eb];B[ec];W[fd];B[fa];W[db];B[ha];W[ff];B[ee];W[eg];B[ge];W[gf];B[eh];W[ef];B[ic]")

for move in moves:
    game.step(move)

game.render_in_terminal()
w = game.determine_winner(7)
print(w)

