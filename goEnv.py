import numpy as np
from scipy.ndimage import label, binary_dilation


class GoGame:
    def __init__(self, size=9):
        self.size = size
        # Empty = 0, black = 1, white = 2
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.turn = 1  # 1 for black, 2 for white

        self.history = []  # To track past board states (for Ko rules)

        # negative if white captures more, positive if black captures more
        self.capture_balance = 0

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.turn = 1
        self.consecutivePasses = 0
        self.history = []
        return self.board

    def step(self, action):
        if action not in self.get_legal_actions():
            raise ValueError("Illegal move.")

        if action == "pass":
            print("pass")
            self.consecutivePasses += 1
            self.turn = 3 - self.turn  # Switch turns

            if self.consecutivePasses == 2:
                print("Game is over")
                self.get_areas()
                print(self.capture_balance)
                return self.board, 0, True

            return self.board, 0, False  # No reward for passing, game not over

        self.consecutivePasses = 0

        x, y = action
        self.board[x, y] = self.turn
        captured_stones = self.check_captures(x, y)  # Capture logic

        self.history.append(self.board.copy())
        self.turn = 3 - self.turn  # Switch turns

        # Reward is number of captured stones
        return self.board, len(captured_stones), self.is_game_over()

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

        # Add the pass action as a legal move
        legal_moves.append("pass")
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
                        # print("remove ", opponent)
                        self.board[gx, gy] = 0  # Remove captured stones

                        # Update capture balance
                        if opponent == 1:
                            self.capture_balance -= 1
                        else:
                            self.capture_balance += 1

        return captured_stones

    def count_stones(self):
        black_stones = sum(1 for i in range(self.size)
                           for j in range(self.size) if self.board[i, j] == 1)
        white_stones = sum(1 for i in range(self.size)
                           for j in range(self.size) if self.board[i, j] == 2)
        return black_stones, white_stones

    """ Gets the value of territory for both black and white areas """

    def get_areas(self):

        # TODO: Implement check for prisoners

        black_area, white_area = 0, 0

        # Matrix that shows all empty intersections
        empty_matrix = np.array([[1 if self.board[x, y] == 0 else 0 for y in range(
            len(self.board[0]))] for x in range(len(self.board))])

        # Clusters intersections with labels 1, 2, 3 etc.
        labeled_empty_matrix, num_empty_areas = label(
            empty_matrix)

        for area in range(1, num_empty_areas + 1):
            # Isolate specific empty-cluster and converting to boolean array
            empty_area = labeled_empty_matrix == area
            neighbors = binary_dilation(empty_area)

            black_claim = False
            white_claim = False
            # Extract all white and black pieces into separate boards
            black_board = np.where(self.board == 1, 1, 0)
            white_board = np.where(self.board == 2, 1, 0)

            # Elementwise multiplication between black/white board and neighbors. If any product > 0, the area has a neighbor of that color
            if (black_board * neighbors > 0).any():
                black_claim = True
            if (white_board * neighbors > 0).any():
                white_claim = True
            if black_claim and not white_claim:
                black_area += np.sum(empty_area)
            elif white_claim and not black_claim:
                white_area += np.sum(empty_area)

        print("black area: ", black_area, " white area: ", white_area)
        return black_area, white_area

    def is_game_over(self):
        return False

    def render_in_terminal(self):
        print("  0 1 2 3 4 5 6")
        for i, row in enumerate(self.board):
            row_print = str(i) + " "
            row_print += '─'.join(['┼' if cell == 0
                                   else ('○' if cell == 1 else '●') for cell in row])
            print(row_print)


game = GoGame(size=5)
game.reset()
print(game.get_legal_actions())
game.step((0, 2))
game.step((0, 1))
game.step((0, 3))
game.step((1, 2))
game.step((4, 2))
game.step((1, 3))
game.step((4, 3))
game.step((0, 4))
game.step((0, 2))
game.step((3, 3))
game.step((4, 4))
game.step((2, 3))
game.step((4, 1))
game.step((3, 4))
game.render_in_terminal()

# Reset game when two consecutive passes
_, _, game_over = game.step("pass")
if game_over:
    game.reset()

_, _, game_over = game.step(("pass"))
if game_over:
    game.reset()

"""
# Example Usage
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
"""
