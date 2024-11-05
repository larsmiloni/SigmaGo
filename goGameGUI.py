import pygame
import random
from goEnv import GoGame


class GoGameGUI:
    def __init__(self, game):
        self.game = game

        self.square_size = 60
        # Number of squares * size of square.
        self.board_size = self.game.size * self.square_size
        # Make the screen the size of the board.
        self.screen = pygame.display.set_mode(
            (self.board_size, self.board_size + 30))

        pygame.display.set_caption("Go Game")
        self.running = True

        # Timer setup for automatic moves
        self.move_interval = 300
        self.last_move_time = 0
        self.label = 0
        self.myfont = 0

    def draw_board(self):
        self.screen.fill((212, 149, 60))  # Background color

        for i in range(self.game.size):
            for j in range(self.game.size):
                # Draw grid
                if i < self.game.size - 1 and j < self.game.size - 1:
                    pygame.draw.rect(self.screen, (0, 0, 0),
                                     (j * self.square_size + 30, i * self.square_size + 30,
                                     self.square_size, self.square_size), 1)

                # Draw gray stones for legal moves
                if (j, i) in self.game.get_legal_actions():
                    pygame.draw.circle(self.screen, (211, 211, 211),
                                       (j * self.square_size + self.square_size // 2,
                                        i * self.square_size + self.square_size // 2),
                                       (self.square_size / 1.5) // 2 - 5)

                # Draw black stones
                if self.game.board[i, j] == 1:
                    pygame.draw.circle(self.screen, (0, 0, 0),
                                       (j * self.square_size + self.square_size // 2,
                                        i * self.square_size + self.square_size // 2),
                                       self.square_size // 2 - 5)

                # Draw white stones
                elif self.game.board[i, j] == 2:
                    pygame.draw.circle(self.screen, (255, 255, 255),
                                       (j * self.square_size + self.square_size // 2,
                                        i * self.square_size + self.square_size // 2),
                                       self.square_size // 2 - 5)

        self.myfont = pygame.font.SysFont("monospace", 15)
        #black_score, white_score = self.game.get_score()

        #winner = self.game.determine_winner()

        #score_text = f"Winner: {winner}"
        self.label = self.myfont.render("score_text", 1, (0, 0, 0))
        self.screen.blit(self.label, (80, 300))

    def make_random_move(self):
        legal_moves = self.game.get_legal_actions()

        if legal_moves:
            # Choose a random legal move and make it
            move = random.choice(legal_moves)
            _, _, game_over = self.game.step(move)
            if game_over:
                print("Game over by random move")
                self.running = False

    def run(self):
        while self.running:
            current_time = pygame.time.get_ticks()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        x, y = event.pos
                        row = y // self.square_size
                        col = x // self.square_size

                        # Check if a legal board position was clicked
                        if (col, row) in self.game.get_legal_actions():
                            self.game.step((col, row))
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        _, _, game_over = self.game.step(("pass"))
                        if game_over:
                            print("Game over")
                            # self.running = False

            """
            # Check if it's time to make a random move
            if current_time - self.last_move_time >= self.move_interval:
                self.make_random_move()
                self.last_move_time = current_time  # Update last move time
            """

            self.draw_board()
            pygame.display.flip()


if __name__ == "__main__":
    pygame.init()
    game = GoGame(size=9)
    gui = GoGameGUI(game)
    gui.run()
    pygame.quit()
