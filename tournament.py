from policy_network import PolicyNetwork
from goEnv import GoGame
import govars
import glob
from collections import Counter
import matplotlib.pyplot as plt
import networkx as nx

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
        print("Playing round...")
        for i in range(len(self.players)):
            for j in range(i+1, len(self.players)):
                player1 = self.players[i]
                player2 = self.players[j]
                game_result = self.play_match(player1, player2)
                self.results.append(game_result)

    def play_match(self, player1, player2):
        print('Match started')
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
            return (player1.get_name(), player2.get_name())  # Player1 is the winner
        else:
            return (player2.get_name(), player1.get_name())  # Player2 is the winner


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
tournament = Tournament()

# List all model files in the models folder
model_files = glob.glob("./models/*.pt")
# Load models
models = []
for model_file in model_files:
    model = PolicyNetwork(model_path=model_file)
    models.append(model)



Players = []
Player_count = 0
for model in models:
    # Added player
    Players.append(Player(network=model))
    print(f"Player {Player_count} added with model {model.model_path}")

for player in Players:
    while Player_count !=16:
        tournament.add_player(player)
        Player_count += 1
    break


def create_tournament_bracket(players, results, output_file="tournament_bracket.png"):
    # Create a directed graph
    G = nx.DiGraph()

    # Create a mapping for subsets (round levels)
    subset_mapping = {player: 0 for player in players}  # Start all players in the first round

    # Add edges for matches and update subsets
    for match in results:
        winner, loser = match
        G.add_edge(loser, winner)
        subset_mapping[winner] = max(subset_mapping.get(winner, 0), subset_mapping[loser] + 1)

    # Assign subset attributes to nodes
    for player in players:
        G.nodes[player]["subset"] = subset_mapping[player]

    # Generate positions for the bracket using graphviz layout
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    node_shapes = []
    for node in G.nodes():
        if G.out_degree(node) == 0 and G.in_degree(node) > 0:
            node_shapes.append('o')  # Winner
        else:
            node_shapes.append('s')  # Other players

    # Draw the graph
    plt.figure(figsize=(12, 8))

    # Draw nodes with different colors based on their round
    node_colors = [subset_mapping[node] for node in G.nodes()]
    cmap = plt.cm.cool  # Choose a color map

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=2000,
        node_color=node_colors,
        cmap=cmap,
        node_shape='o',
        alpha=0.9,
    )

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        arrowstyle='->',
        arrowsize=15,
        edge_color='gray',
        width=2,
    )

    # Draw labels
    nx.draw_networkx_labels(
        G,
        pos,
        font_size=10,
        font_weight="bold",
    )

    # Draw edge labels if you have match scores
    # edge_labels = { (loser, winner): f"Score: {score}" for (winner, loser, score) in results }
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    # Save the output
    plt.title("Tournament Bracket", fontsize=16, fontweight="bold")
    plt.axis('off')  # Hide axes for better visualization
    plt.tight_layout()
    plt.savefig(output_file)
    plt.show()


# Generate the bracket



tournament.play_round()

# Ensure the results are formatted as (winner, loser)
results = tournament.results  # Each entry in results is now a tuple (winner, loser)

# Generate players' names
players = [player.get_name() for player in tournament.get_players()]

# Generate the bracket visualization
create_tournament_bracket(players, results)