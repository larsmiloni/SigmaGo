import numpy as np


class GoGame:
    def __init__(self, size=9):
        self.size = size
        # Empty = 0, black = 1, white = 2
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.turn = 1  # 1 for black, 2 for white

        self.history = []  # To track past board states (for Ko rules)
        self.consecutivePasses = 0
        
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
                        self.board[gx, gy] = 0  # Remove captured stones

        return captured_stones

    def count_stones(self):
        black_stones = sum(1 for i in range(self.size)
                           for j in range(self.size) if self.board[i, j] == 1)
        white_stones = sum(1 for i in range(self.size)
                           for j in range(self.size) if self.board[i, j] == 2)
        return black_stones, white_stones

    def is_game_over(self):
        return False
    

    def determine_winner(self):
        # Count stones on the board for each player
        black_stones, white_stones = self.count_stones()

        # Track dead stones for each color
        dead_black_stones = set()
        dead_white_stones = set()

        # Identify dead stones for each player
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 1 and self.is_dead((i, j)):
                    dead_black_stones.update(self.get_group((i, j)))
                elif self.board[i, j] == 2 and self.is_dead((i, j)):
                    dead_white_stones.update(self.get_group((i, j)))

        # Adjust stone count by excluding dead stones
        black_stones -= len(dead_black_stones)
        white_stones -= len(dead_white_stones)

        # Initialize territory counts
        black_territory = 0
        white_territory = 0
        visited = set()

        def bfs_territory(start):
            queue = [start]
            territory = set()
            boundary_colors = set()

            while queue:
                x, y = queue.pop(0)
                if (x, y) in visited:
                    continue
                visited.add((x, y))
                territory.add((x, y))

                for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        if self.board[nx, ny] == 0 and (nx, ny) not in visited:
                            queue.append((nx, ny))
                        elif self.board[nx, ny] in (1, 2):
                            boundary_colors.add(self.board[nx, ny])

            # Determine ownership if surrounded by one color
            if len(boundary_colors) == 1:
                owner = boundary_colors.pop()
                if owner == 1:
                    return len(territory), "black"
                elif owner == 2:
                    return len(territory), "white"
            return 0, "neutral"

        # Traverse all empty points to calculate territory ownership
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i, j] == 0 and (i, j) not in visited:
                    territory_count, owner = bfs_territory((i, j))
                    if owner == "black":
                        black_territory += territory_count
                    elif owner == "white":
                        white_territory += territory_count

        # Add dead stones to the opponent's territory count
        black_territory += len(dead_white_stones)
        white_territory += len(dead_black_stones)

        # Total score for determining winner
        black_total = black_stones + black_territory
        white_total = white_stones + white_territory + 7.5

        # Determine the winner based on higher score
        if black_total > white_total:
            return "black"
        elif white_total > black_total:
            return "white"

    def is_dead(self, point):
        """Determine if a group of stones at the given point is dead (cannot form two eyes)."""
        group = self.get_group(point)
        liberties = set()
        
        # Find all liberties of the group
        for x, y in group:
            for nx, ny in [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]:
                if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == 0:
                    liberties.add((nx, ny))
        
        # A group with fewer than two liberties is almost certainly dead
        if len(liberties) < 2:
            return True

        # Check if there are at least two separate eye-like regions among liberties
        potential_eyes = []
        visited = set()

        def is_eye(liberty):
            """Determine if a given liberty point can be an eye."""
            lx, ly = liberty
            eye_border_color = self.board[point[0], point[1]]
            
            for nx, ny in [(lx-1, ly), (lx+1, ly), (lx, ly-1), (lx, ly+1)]:
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if self.board[nx, ny] != eye_border_color and self.board[nx, ny] != 0:
                        return False
            return True

        # Group the liberties into potential eyes
        for liberty in liberties:
            if liberty not in visited and is_eye(liberty):
                # Find the connected region for this liberty to consider it as a potential eye
                eye_region = set()
                stack = [liberty]

                while stack:
                    lx, ly = stack.pop()
                    if (lx, ly) in visited:
                        continue
                    visited.add((lx, ly))
                    eye_region.add((lx, ly))

                    # Explore adjacent liberties to expand the eye region
                    for nx, ny in [(lx-1, ly), (lx+1, ly), (lx, ly-1), (lx, ly+1)]:
                        if (nx, ny) in liberties and (nx, ny) not in visited and is_eye((nx, ny)):
                            stack.append((nx, ny))

                # Add the identified eye region if it's distinct
                if eye_region:
                    potential_eyes.append(eye_region)

        # A group is alive if it has at least two distinct eyes
        return len(potential_eyes) < 2
    

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
game.step((1, 0))
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

game.render_in_terminal()
winner = game.determine_winner()
print("Winner: ", winner)
"""