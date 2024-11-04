import numpy as np


class GoGame:
    def __init__(self, size=9):
        self.size = size
        # Empty = 0, black = 1, white = 2
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.turn = 1  # 1 for black, 2 for white

        self.history = []  # To track past board states (for Ko rules)
        self.consecutivePasses = 0
        self.isGameOver = False

        self.winner = 0
        
    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.turn = 1
        self.consecutivePasses = 0
        self.history = []
        return self.board

    def step(self, action):
        if action not in self.get_legal_actions():
            raise ValueError("Illegal move.")

        if action == "resign":
            print("resignation from:", self.turn)
            self.winner = 3 - self.turn
            self.isGameOver = True
            return self.board, 0, True

        # Handle pass action
        if action == "pass":
            print(f"Player {self.turn} passes.")
            self.consecutivePasses += 1
            self.turn = 3 - self.turn  # Switch turns

            # End the game if both players pass consecutively
            if self.consecutivePasses >= 2:
                print("Game over due to consecutive passes.")
                self.isGameOver = True
                return self.board, 0, True

            return self.board, 0, False  # No reward for passing, game not over

        # Reset consecutive passes if a stone is placed and the last move was not a pass
        self.consecutivePasses = 0

        x, y = action
        self.board[x, y] = self.turn
        captured_stones = self.check_captures(x, y)  # Capture logic

        self.history.append(self.board.copy())
        self.turn = 3 - self.turn  # Switch turns
        # Reward is number of captured stones
        return self.board, len(captured_stones), False

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
                        legal_moves.append((i, j))

                    # Reset the board after checking
                    self.board[i, j] = 0
                    for cx, cy in captured:
                        self.board[cx, cy] = 3 - self.turn

                    # If no legal moves are available except pass/resign
                    if not legal_moves:
                        return ["pass", "resign"]

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
    

    def find_dead_stones(self):
        """Identify groups with no liberties, marking them as dead."""
        black_dead, white_dead = set(), set()

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] in (1, 2):  # Only check stones
                    group = self.get_group((i, j))
                    if not any(self.hasLiberties(stone) for stone in group):
                        if self.board[i, j] == 1:
                            black_dead.update(group)
                        else:
                            white_dead.update(group)

        return black_dead, white_dead

    def determine_winner(self):
        """Determine winner considering dead stones."""
        black_stones, white_stones = self.count_stones()
        
        # Identify dead stones
        black_dead, white_dead = self.find_dead_stones()
        
        # Adjust counts to exclude dead stones
        black_stones -= len(black_dead)
        white_stones -= len(white_dead)

        # Calculate territory for black and white
        black_territory, white_territory = self.count_territories()

        # Total score calculation
        black_score = black_stones + black_territory
        white_score = white_stones + white_territory


        if black_score > white_score:
            self.winner = 2
            return "Black wins", black_score, white_score
        elif white_score > black_score:
            self.winner = 1
            return "White wins", black_score, white_score
        else:
            self.winner = 0
            return "Draw", black_score, white_score

    def count_territories(self):
        visited = set()
        black_territory = 0
        white_territory = 0

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0 and (i, j) not in visited:
                    empty_group, border_colors = self.get_empty_group_and_borders((i, j))
                    visited.update(empty_group)

                    # If the empty area is bordered by only one color, count it as territory
                    if len(border_colors) == 1:
                        if 1 in border_colors:
                            black_territory += len(empty_group)
                        elif 2 in border_colors:
                            white_territory += len(empty_group)

        return black_territory, white_territory

    def get_empty_group_and_borders(self, intersection):
        x, y = intersection
        visited = set()
        empty_group = []
        border_colors = set()

        def dfs(px, py):
            if (px, py) in visited:
                return
            if not (0 <= px < self.size and 0 <= py < self.size):
                return

            visited.add((px, py))

            if self.board[px, py] == 0:
                empty_group.append((px, py))

                # Explore all four directions
                for nx, ny in [(px-1, py), (px+1, py), (px, py-1), (px, py+1)]:
                    dfs(nx, ny)
            else:
                # Record border color if we reach a stone
                border_colors.add(self.board[px, py])

        dfs(x, y)
        return empty_group, border_colors
    

    def render_in_terminal(self):
        print("  0 1 2 3 4 5 6")
        for i, row in enumerate(self.board):
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
            move = entry[2:4]  # Get the two-letter move, e.g., 'fe'
            
            # Convert letters to board coordinates
            row = ord(move[0]) - ord('a')  # Vertical position (y-coordinate)
            col = ord(move[1]) - ord('a')  # Horizontal position (x-coordinate)
            
            # Append to moves list as (player, (row, col))
            moves.append((row, col))
        
        return moves



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

game = GoGame(size=9)
game.reset()

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
_, _, game_over = game.step("pass")
_, _, game_over = game.step("pass")
"""
_, _, game_over = game.step("resign")

print("game over: ", game_over)

game.render_in_terminal()
winner = game.determine_winner()
print("Winner: ", winner)
