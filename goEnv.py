import numpy as np
import govars

class GoGame:
    def __init__(self, size: int = 9):
        self.size = size
        self.shape = (self.size, self.size)
        self.state = np.zeros((govars.NUM_LAYERS, self.size, self.size))
        self.history = []
        self.prisoners = {1: 0, 2: 0}

    def reset(self):
        self.state = np.zeros((govars.NUM_LAYERS, self.size, self.size))
        self.history = []

    def step(self, action):
        self.get_invalid_moves()
        if action not in self.get_legal_actions():
            raise ValueError("Illegal move.")

        previous_move_was_pass = np.max(self.state[govars.PASS] == 1) == 1

        if action == "pass":
            self.state[govars.PASS] = 1
            print("pass")
            self.update_state()

            if previous_move_was_pass:
                print("Game over due to consecutive passes.")
                self.isGameOver = True
                self.state[govars.DONE] = 1
                return self.get_board(), 0, True

            return self.get_board(), 0, False

        self.state[govars.PASS] = 0

        x, y = action
        # Place stone at (x,y) on the board
        self.get_board()[y][x] = self.get_turn()
        captured_stones = self.check_captures(x, y)

        self.history.append(self.get_board().copy())
        self.update_state()

        return self.get_board(), len(captured_stones), False

    def get_invalid_moves(self):
        valid_moves = self.get_legal_actions()
        valid_moves.remove("pass")
        invalid_moves = np.ones(self.shape)
        for x, y in valid_moves:
            invalid_moves[y][x] = 0
        self.state[govars.INVD] = invalid_moves

    def get_turn(self):
        return np.max(self.state[govars.TURN]) + 1

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

    def get_legal_actions(self):
        legal_moves = []

        for x in range(self.size):
            for y in range(self.size):
                if self.get_board()[y][x] == 0:
                    self.get_board()[y][x] = self.get_turn()
                    captured = self.check_captures(x, y)

                    if (not self.is_self_capture((x, y)) or captured) and not self.is_ko():
                        if not self.is_in_two_eye_group((x, y)):
                            legal_moves.append((x, y))

                    self.get_board()[y][x] = 0
                    for cx, cy in captured:
                        self.get_board()[cy][cx] = 3 - self.get_turn()

        if not legal_moves:
            return ["pass"]

        legal_moves.append("pass")
        return legal_moves

    def is_eye(self, position, color):
        """Check if a position is an eye for the given color.
        An eye is an empty point surrounded by stones of the same color,
        where the diagonal points are either friendly stones or the point is on the edge."""
        x, y = position
        board = self.get_board()
        
        # The position itself must be empty
        if board[y][x] != 0:
            return False
            
        # Check orthogonal positions (adjacent)
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                if board[ny][nx] != color:
                    return False
            
        # Count diagonal positions that need to be checked
        required_diagonal_stones = 0
        diagonal_stones = 0
        
        # Check diagonal positions
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                required_diagonal_stones += 1
                if board[ny][nx] == color:
                    diagonal_stones += 1
        
        # For corner positions
        if required_diagonal_stones <= 2:
            return True
        # For edge positions
        elif required_diagonal_stones == 3:
            return diagonal_stones >= 2
        # For center positions
        else:
            return diagonal_stones >= 3
        


    def is_ko(self):
        for board in self.history:
            if np.array_equal(board, self.get_board()):
                return True
        return False

    def is_self_capture(self, move):
        x, y = move
        group = self.get_group((x, y))

        if any(self.hasLiberties(stone) for stone in group):
            return False

        opponent = 3 - self.get_turn()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and self.get_board()[ny][nx] == opponent:
                opponent_group = self.get_group((nx, ny))
                if not any(self.hasLiberties(stone) for stone in opponent_group):
                    return False

        return True

    def get_group(self, intersection):
        x, y = intersection
        color = self.get_board()[y][x]
        visited = set()
        group = []

        def dfs(px, py):
            if (px, py) in visited:
                return
            if not (0 <= px < self.size and 0 <= py < self.size):
                return
            if self.get_board()[py][px] != color:
                return

            visited.add((px, py))
            group.append((px, py))

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                dfs(px + dx, py + dy)

        dfs(x, y)
        return group

    def hasLiberties(self, intersection):
        x, y = intersection
        
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and self.get_board()[ny][nx] == 0:
                return True

        return False

    def check_captures(self, x, y):
        opponent = 3 - self.get_turn()
        captured_stones = []

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and self.get_board()[ny][nx] == opponent:
                group = self.get_group((nx, ny))
                if not any(self.hasLiberties(stone) for stone in group):
                    captured_stones.extend(group)
                    for gx, gy in group:
                        self.get_board()[gy][gx] = 0

        return captured_stones

    def count_stones(self):
        black_stones = sum(1 for y in range(self.size)
                         for x in range(self.size) if self.get_board()[y][x] == 1)
        white_stones = sum(1 for y in range(self.size)
                         for x in range(self.size) if self.get_board()[y][x] == 2)
        return black_stones, white_stones

    def render_in_terminal(self):
        colNums = "  " + " ".join(str(i) for i in range(self.size))
        print(colNums)
        for i, row in enumerate(self.get_board()):
            row_print = str(i) + " "
            row_print += '─'.join(['┼' if cell == 0
                                else ('○' if cell == 1 else '●') for cell in row])
            print(row_print)

    def sgf_to_coordinates(self, sgf_moves):
        moves = []
        for entry in sgf_moves.split(';'):
            if not entry:
                continue
            player = entry[0]
            move = entry[2:4]
            
            x = ord(move[0]) - ord('a')
            y = ord(move[1]) - ord('a')
            
            moves.append((x, y))

        return moves
    

    def count_territory(self):
        """Count territory using Chinese rules"""
        territory = {1: 0, 2: 0}  # Black and White territory
        visited = set()
        
        def flood_fill(x, y):
            """Returns (territory_color, size) or (0, 0) if neutral"""
            if (x, y) in visited or not (0 <= x < self.size and 0 <= y < self.size):
                return 0, 0
                
            visited.add((x, y))
            current = self.get_board()[y][x]
            
            if current != 0:  # If it's a stone, count it for that player
                return current, 1
                
            surrounding_colors = set()
            region_size = 1
            
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    color, size = flood_fill(nx, ny)
                    if color != 0:
                        surrounding_colors.add(color)
                    region_size += size
            
            # If region is surrounded by only one color, it's their territory
            if len(surrounding_colors) == 1:
                return surrounding_colors.pop(), region_size
            return 0, 0  # Neutral territory
            
        # Count all regions
        for x in range(self.size):
            for y in range(self.size):
                if (x, y) not in visited:
                    color, size = flood_fill(x, y)
                    if color in territory:
                        territory[color] += size
                        
        return territory

    def determine_winner(self):
        """Determine winner using Chinese rules"""
        territory = self.count_territory()
        black_score = territory[1]  # Black's territory and stones
        white_score = territory[2] #+ 7.5  # White's territory and stones plus komi
        
        print(f"Black's score: {black_score}")
        print(f"White's score: {white_score}")
        
        if black_score > white_score:
            return "Black wins by " + str(black_score - white_score) + " points"
        elif white_score > black_score:
            return "White wins by " + str(white_score - black_score) + " points"
        else:
            return "The game is a draw"

    def is_in_two_eye_group(self, move):
        """Check if filling this eye would remove a vital eye from a group with exactly two eyes"""
        x, y = move
        group = self.get_group((x, y))
        
        # First, simulate placing the stone
        self.get_board()[y][x] = self.get_turn()
        eyes = []
        
        # Find all eyes of the group
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) not in group and self.is_eye((i, j), self.get_turn()):
                    connected_to_group = False
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = i + dx, j + dy
                        if (nx, ny) in group:
                            connected_to_group = True
                            break
                    if connected_to_group:
                        eyes.append((i, j))
        
        # Remove the simulated stone
        self.get_board()[y][x] = 0
        
        # If the group has exactly two eyes and we're trying to fill one
        return len(eyes) == 2 and (x, y) in eyes

    def end_game(self):
        """Handle end game under Chinese rules"""
        # Remove dead groups
        dead_stones = self.identify_dead_groups()
        for stone in dead_stones:
            x, y = stone
            self.get_board()[y][x] = 0
        
        # Determine and print winner
        result = self.determine_winner()
        print(result)
        return result

    def identify_dead_groups(self):
        """Identify groups that are completely surrounded (dead)"""
        dead_stones = set()
        
        for y in range(self.size):
            for x in range(self.size):
                if self.get_board()[y][x] != 0:
                    group = self.get_group((x, y))
                    if not self.has_living_space(group):
                        dead_stones.update(group)
                        
        return dead_stones

    def has_living_space(self, group):
        """Check if a group has potential to live (has or can make eyes)"""
        eye_spaces = 0
        potential_eyes = set()
        
        # Check all empty points adjacent to the group
        for x, y in group:
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.size and 0 <= ny < self.size and 
                    self.get_board()[ny][nx] == 0 and 
                    self.is_eye((nx, ny), self.get_board()[y][x])):
                    potential_eyes.add((nx, ny))
        
        return len(potential_eyes) >= 2

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

game.render_in_terminal()
game.end_game()
game.render_in_terminal()
#print(game.get_legal_actions())
#print("GROOO:", game.get_group((1, 3)))