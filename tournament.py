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

    def add_player(self, player):
        self.players.append(player)

    def get_players(self):
        return self.players

    def get_results(self):
        return self.results
    
    def reset_results(self):
        self.results = []

    def play_tournament(self):
        if len(self.players) < 2:
            print("Not enough players to start the tournament.")
            return None  # No tournament run

        current_stage_players = self.players.copy()
        stage_index = 0
        placements = {}  # player_name: placement

        while len(current_stage_players) > 1 and stage_index < len(self.stages):
            stage = self.stages[stage_index]
            print(f"\n----- ({stage}) -----")

            # Shuffle players to randomize matchups
            random.shuffle(current_stage_players)

            next_round_players = []
            for i in range(0, len(current_stage_players), 2):
                if i + 1 < len(current_stage_players):
                    player1 = current_stage_players[i]
                    player2 = current_stage_players[i + 1]
                    winner, loser = self.play_match(player1, player2)
                    self.results.append((stage, player1.get_name(), player2.get_name(), winner.get_name()))
                    next_round_players.append(winner)
                else:
                    # If odd number of players, the last player gets a bye
                    player = current_stage_players[i]
                    print(f"{player.get_name()} gets a bye to the next round.")
                    next_round_players.append(player)

            current_stage_players = next_round_players
            stage_index += 1

        if current_stage_players:
            champion = current_stage_players[0]
            print(f"\n----- Tournament Champion -----")
            print(f"The champion is: {champion.get_name()}")
            placements[champion.get_name()] = 1  # 1st place

            # Assign 2nd place (runner-up)
            if len(current_stage_players) > 1:
                runner_up = None
                # Find the last match to get the runner-up
                last_stage_results = [res for res in self.results if res[0] == self.stages[stage_index - 1]]
                if last_stage_results:
                    _, p1, p2, winner = last_stage_results[-1]
                    runner_up = p2 if winner == p1 else p1
                    placements[runner_up] = 2
                    print(f"Runner-up: {runner_up}")

            # Assign 3rd place
            # This implementation does not include a match for 3rd place.
            # You can add this feature if needed.
        else:
            print("No champion could be determined.")

        # Assign placements for players who did not reach the final stages
        # This is a simplistic approach and can be enhanced
        all_player_names = [player.get_name() for player in self.players]
        for player in all_player_names:
            if player not in placements:
                # Assign a placement based on elimination round
                # For simplicity, assign the next available placement
                placements[player] = len(placements) + 1

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
    tournament_runs = 5  # Number of tournament runs
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

        plt.figure(figsize=(10, 6))
        plt.bar(players_sorted, means, color='skyblue')
        plt.xlabel('Players')
        plt.ylabel('Mean Placement')
        plt.title('Mean Placement of Players Over {} Tournament Runs'.format(tournament_runs))
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("mean_placements.png")
    except Exception as e:
        print(f"Visualization failed: {e}")

if __name__ == "__main__":
    main()