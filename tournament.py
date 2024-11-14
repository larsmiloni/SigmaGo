from policy_network import PolicyNetwork
import copy
from goEnv import GoGame
import govars

from policy_network import PolicyNetwork
import copy
from goEnv import GoGame
import govars
from collections import Counter

class Tournament:
    def __init__(self):
        self.players = []
        self.results = []

    def add_player(self, player):
        self.players.append(player)

    def get_players(self):
        return self.players

    def get_results(self):
        return self.results

    def play_round(self):
        for i in range(len(self.players)):
            for j in range(i+1, len(self.players)):
                player1 = self.players[i]
                player2 = self.players[j]
                game_result = self.play_match(player1, player2)
                self.results.append(game_result)

    def play_match(self, player1, player2):
        game = GoGame(size=9)
        current_player = 'black'
        while not game.state[govars.DONE].any():
            if current_player == 'black':
                move = player1.get_player_move(game)
            else:
                move = player2.get_player_move(game)

            if move != "pass":
                game.step(move)
                current_player = 'white' if current_player == 'black' else 'black'
            else:
                game.step("pass")
                current_player = 'white' if current_player == 'black' else 'black'

        winner = game.determine_winner()
        if winner == 0:
            return (player1.get_name(), player2.get_name())
        else:
            return (player2.get_name(), player1.get_name())

    def get_tournament_results(self):
        # Count the number of wins for each player
        win_counts = Counter([result[0] for result in self.results])

        # Sort the players by their win counts
        sorted_players = sorted(win_counts.items(), key=lambda x: x[1], reverse=True)

        # Determine the number of players with non-zero wins
        num_players_with_wins = len([player for player, wins in sorted_players if wins > 0])

        tournament_results = {}

        # Assign places based on the number of players with non-zero wins
        if num_players_with_wins >= 3:
            tournament_results = {
                "1st Place": sorted_players[0][0],
                "2nd Place": sorted_players[1][0],
                "3rd Place": sorted_players[2][0]
            }
        elif num_players_with_wins == 2:
            tournament_results = {
                "1st Place": sorted_players[0][0],
                "2nd Place": sorted_players[1][0]
            }
        elif num_players_with_wins == 1:
            tournament_results = {
                "1st Place": sorted_players[0][0]
            }
        else:
            tournament_results = {
                "1st Place": "No Winner"
            }

        return tournament_results

class Player:
    def __init__(self, network):
        self.network = network
        self.player_policy = self.load_player()
        self.name = network

    def get_name(self):
        return self.name

    def load_player(self) -> PolicyNetwork:
        player_policy = PolicyNetwork(self.network)
        return player_policy

    def get_player_move(self, game):
        move = self.player_policy.select_move(game)
        print(move)
        return move

# Example usage
tournament = Tournament()



player1 = Player("./models/PN_R3_C64.pt")
player2 = Player("./models/PN_R3_C64_IMPROVED_MODEL.pt")
player3 = Player("./models/PN_R3_C64_IMPROVED_MODEL_2.pt")

tournament.add_player(player1)
tournament.add_player(player2)
tournament.add_player(player3)

tournament.play_round()
results = tournament.get_tournament_results()
print(results)