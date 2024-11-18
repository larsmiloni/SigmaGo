import random
from policy_network import PolicyNetwork
from goEnv import GoGame
import govars
import glob
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import networkx as nx

class Tournament:
    def __init__(self):
        self.players = []
        self.results = []
        self.stages = ["Quarterfinals", "Semifinals", "Final"]
        self.placement_matches = {
            "5th-8th Semifinals": [],
            "5th-6th Place": [],
            "7th-8th Place": [],
            "3rd Place": []
        }

    def add_player(self, player):
        self.players.append(player)

    def get_players(self):
        return self.players

    
    def reset_results(self):
        self.results = []
        for key in self.placement_matches:
            self.placement_matches[key] = []

    def play_tournament(self):
        if len(self.players) < 2:
            print("Not enough players to start the tournament.")
            return None  # No tournament run

        current_stage_players = self.players.copy()
        stage_index = 0
        placements = {}  # player_name: placement

        # Quarterfinals
        print(f"\n----- ({self.stages[stage_index]}) -----")
        random.shuffle(current_stage_players)
        winners_qf = []
        losers_qf = []

        for i in range(0, len(current_stage_players), 2):
            if i + 1 < len(current_stage_players):
                player1 = current_stage_players[i]
                player2 = current_stage_players[i + 1]
                winner, loser = self.play_match(player1, player2)
                self.results.append((self.stages[stage_index], player1.get_name(), player2.get_name(), winner.get_name()))
                winners_qf.append(winner)
                losers_qf.append(loser)
            else:
                # If odd number of players, the last player gets a bye
                player = current_stage_players[i]
                print(f"{player.get_name()} gets a bye to the next round.")
                winners_qf.append(player)

        stage_index += 1

        # 5th-8th Semifinals
        print(f"\n----- (5th-8th Semifinals) -----")
        random.shuffle(losers_qf)
        winners_5_8_sf = []
        losers_5_8_sf = []

        for i in range(0, len(losers_qf), 2):
            if i + 1 < len(losers_qf):
                player1 = losers_qf[i]
                player2 = losers_qf[i + 1]
                winner, loser = self.play_match(player1, player2)
                self.results.append(("5th-8th Semifinals", player1.get_name(), player2.get_name(), winner.get_name()))
                self.placement_matches["5th-8th Semifinals"].append((player1.get_name(), player2.get_name(), winner.get_name()))
                winners_5_8_sf.append(winner)
                losers_5_8_sf.append(loser)
            else:
                # If odd, assign bye
                player = losers_qf[i]
                print(f"{player.get_name()} gets a bye in 5th-8th Semifinals.")
                winners_5_8_sf.append(player)

        # Semifinals
        print(f"\n----- ({self.stages[stage_index]}) -----")
        random.shuffle(winners_qf)
        winners_sf = []
        losers_sf = []

        for i in range(0, len(winners_qf), 2):
            if i + 1 < len(winners_qf):
                player1 = winners_qf[i]
                player2 = winners_qf[i + 1]
                winner, loser = self.play_match(player1, player2)
                self.results.append((self.stages[stage_index], player1.get_name(), player2.get_name(), winner.get_name()))
                winners_sf.append(winner)
                losers_sf.append(loser)
            else:
                player = winners_qf[i]
                print(f"{player.get_name()} gets a bye to the next round.")
                winners_sf.append(player)

        stage_index += 1

        # 5th-6th Place Match
        if len(winners_5_8_sf) >= 2:
            print(f"\n----- (5th-6th Place Match) -----")
            player1, player2 = winners_5_8_sf[:2]
            winner, loser = self.play_match(player1, player2)
            self.results.append(("5th-6th Place", player1.get_name(), player2.get_name(), winner.get_name()))
            self.placement_matches["5th-6th Place"].append((player1.get_name(), player2.get_name(), winner.get_name()))
            placements[winner.get_name()] = 5
            placements[loser.get_name()] = 6
        elif len(winners_5_8_sf) == 1:
            player = winners_5_8_sf[0]
            placements[player.get_name()] = 5
            print(f"{player.get_name()} is assigned 5th place by default.")

        # 7th-8th Place Match
        if len(losers_5_8_sf) >= 2:
            print(f"\n----- (7th-8th Place Match) -----")
            player1, player2 = losers_5_8_sf[:2]
            winner, loser = self.play_match(player1, player2)
            self.results.append(("7th-8th Place", player1.get_name(), player2.get_name(), winner.get_name()))
            self.placement_matches["7th-8th Place"].append((player1.get_name(), player2.get_name(), winner.get_name()))
            placements[winner.get_name()] = 7
            placements[loser.get_name()] = 8
        elif len(losers_5_8_sf) == 1:
            player = losers_5_8_sf[0]
            placements[player.get_name()] = 7
            print(f"{player.get_name()} is assigned 7th place by default.")

        # Semifinal Losers Play for 3rd Place
        if len(losers_sf) >= 2:
            print(f"\n----- (3rd Place Match) -----")
            player1, player2 = losers_sf[:2]
            winner, loser = self.play_match(player1, player2)
            self.results.append(("3rd Place", player1.get_name(), player2.get_name(), winner.get_name()))
            self.placement_matches["3rd Place"].append((player1.get_name(), player2.get_name(), winner.get_name()))
            placements[winner.get_name()] = 3
            placements[loser.get_name()] = 4
        elif len(losers_sf) == 1:
            player = losers_sf[0]
            placements[player.get_name()] = 3
            print(f"{player.get_name()} is assigned 3rd place by default.")

        # Final Match
        if len(winners_sf) >= 2:
            print(f"\n----- (Final) -----")
            player1, player2 = winners_sf[:2]
            winner, loser = self.play_match(player1, player2)
            self.results.append(("Final", player1.get_name(), player2.get_name(), winner.get_name()))
            placements[winner.get_name()] = 1
            placements[loser.get_name()] = 2
            print(f"Runner-up: {loser.get_name()}")
            print(f"Champion: {winner.get_name()}")
        elif len(winners_sf) == 1:
            player = winners_sf[0]
            placements[player.get_name()] = 1
            print(f"{player.get_name()} is assigned Champion by default.")
        else:
            print("No final match could be determined.")

        # Assign placements for any remaining players (if any)
        all_player_names = [player.get_name() for player in self.players]
        for player in all_player_names:
            if player not in placements:
                # Assign a placement based on elimination round or default to 8
                placements[player] = 8
                print(f"{player} is assigned 8th place by default.")

        return placements

    def play_match(self, player1, player2):
        print(f"Match: {player1.get_name()} vs {player2.get_name()}")
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

        # Determine the winner and loser
        winner = game.determine_winner()
        if winner == 0:
            print(f"Winner: {player1.get_name()}\n")
            return player1, player2
        else:
            print(f"Winner: {player2.get_name()}\n")
            return player2, player1

    def get_tournament_results(self):
        return self.results


def calculate_mean_placements(all_placements, players):
    placement_sums = defaultdict(float)
    placement_counts = defaultdict(int)

    for run in all_placements:
        for player, placement in run.items():
            placement_sums[player] += placement
            placement_counts[player] += 1

    mean_placements = {}
    for player in players:
        if placement_counts[player] > 0:
            mean = placement_sums[player] / placement_counts[player]
            mean_placements[player] = mean
        else:
            mean_placements[player] = float('inf')  # If player was never placed

    return mean_placements



class Player:
    def __init__(self, network):
        self.network = network
        self.player_policy = self.load_player()
        self.name = network.model_path

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
def main():
    tournament_runs = 2  # Number of tournament runs
    all_placements = []  # List to store placements from each run

    # List all model files in the models folder
    model_files = glob.glob("./models/*.pt")

    if len(model_files) < 2:
        print("Need at least 2 models to start the tournament.")
        return

    # Load models
    models = []
    for model_file in model_files:
        model = PolicyNetwork(model_path=model_file)
        models.append(model)

    # Create players
    players = []
    for idx, model in enumerate(models):
        player = Player(network=model)
        players.append(player)
        print(f"Player {idx + 1} added with model {model.model_path}")

    # Ensure there are exactly 8 players (or adjust accordingly)
    desired_player_count = 8
    while len(players) < desired_player_count:
        # If not enough unique players, cycle through existing players to fill the bracket
        players.append(players[len(players) % len(models)])
    players = players[:desired_player_count]  # Limit to desired count

    # Extract player names
    player_names = [player.get_name() for player in players]

    # Run the tournament multiple times
    for run in range(1, tournament_runs + 1):
        print(f"\n===== Tournament Run {run} =====")
        tournament = Tournament()

        # Add players to the tournament
        for player in players:
            tournament.add_player(player)

        # Play the tournament
        placements = tournament.play_tournament()

        if placements:
            all_placements.append(placements)

        # Reset the tournament results for the next run
        tournament.reset_results()

    # Calculate mean placements
    mean_placements = calculate_mean_placements(all_placements, player_names)

    # Sort players by mean placement (lower is better)
    sorted_mean = sorted(mean_placements.items(), key=lambda x: x[1])

    # Print mean placements
    print("\n===== Mean Placements After {} Runs =====".format(tournament_runs))
    for player, mean in sorted_mean:
        print(f"{player}: Mean Placement = {mean:.2f}")

    # Optional: Visualize Mean Placements
    try:
        players_sorted = [player for player, _ in sorted_mean]
        means = [mean_placements[player] for player in players_sorted]

        plt.figure(figsize=(12, 8))
        bars = plt.bar(players_sorted, means, color='skyblue')
        plt.xlabel('Players')
        plt.ylabel('Mean Placement')
        plt.title(f'Mean Placement of Players Over {tournament_runs} Tournament Runs')
        plt.xticks(rotation=45, ha='right')

        # Annotate bars with the mean values
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            plt.annotate(f'{mean:.2f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig("mean_placements.png")
    except Exception as e:
        print(f"Visualization failed: {e}")


if __name__ == "__main__":
    main()