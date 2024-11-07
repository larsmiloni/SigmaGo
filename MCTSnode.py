import copy
import math
import numpy as np
from scipy.special import softmax

class MCTSNode:
    """
    Represents a node in the Monte Carlo Tree Search (MCTS) process, tracking the current state,
    possible moves, visit count, value, and the parent-child relationships.

    Attributes:
        game_state (GoGame): Current game state for this node.
        parent (MCTSNode): Parent node in the MCTS tree.
        move (tuple): Move that led to this node.
        prior (float): Prior probability from the policy network.
    """



    def __init__(self, game_state, parent=None, move=None, prior=0.0):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value = 0
        self.prior = prior
        self.is_expanded = False
        self.rave_value = 0  # Cumulative RAVE value for this node
        self.rave_visits = 0  # RAVE visit count for this node


    
    def select_child(self, c_puct=math.sqrt(2), rave_weight=0.5, top_n=5):
        """
        Selects a child node using softmax sampling over the top-N highest UCB scores.
        Includes comprehensive error handling and special move handling for Go.
        
        Args:
            c_puct (float): Constant for exploration term scaling.
            rave_weight (float): Weight for the RAVE bonus term.
            top_n (int): Number of top children to consider for softmax sampling.
        
        Returns:
            Tuple: The selected move and child node.
        """
        # First check if this is a terminal state
        if self.game_state.isGameOver:
            return "pass", None
            
        # Get legal actions directly from game state
        legal_actions = self.game_state.get_legal_actions()
        
        # Handle special cases first
        if not legal_actions:
            return "pass", None
        if set(legal_actions) == {"pass"}:
            return "pass", None
            
        # If no children exist but we have legal moves, expand first
        if not self.children and legal_actions:
            try:
                policy, _ = self.game_state.network.predict(self.game_state.state)
                for move in legal_actions:
                    if move not in ("pass"):
                        move_idx = move[0] * 9 + move[1]
                        new_state = copy.deepcopy(self.game_state)
                        new_state.step(move)
                        self.children[move] = MCTSNode(
                            new_state,
                            parent=self,
                            move=move,
                            prior=policy[move_idx]
                        )
            except AttributeError:
                # If no network available, use uniform prior
                prior = 1.0 / len(legal_actions)
                for move in legal_actions:
                    if move not in ("pass"):
                        new_state = copy.deepcopy(self.game_state)
                        new_state.step(move)
                        self.children[move] = MCTSNode(
                            new_state,
                            parent=self,
                            move=move,
                            prior=prior
                        )
        
        # Now calculate scores for existing children
        scores = []
        moves = []
        children = list(self.children.items())
        
        if not children:
            # If still no children after expansion attempt, fall back to pass
            return "pass", None
            
        for move, child in children:
            # Safe score calculation with error handling
            try:
                q_value = child.value / max(child.visits, 1)  # Avoid division by zero
                u_value = c_puct * child.prior * math.sqrt(max(self.visits, 1)) / (1 + child.visits)
                rave_bonus = (rave_weight * (child.rave_value / max(child.rave_visits, 1)) 
                            if child.rave_visits > 0 else 0)
                
                # Add small noise for exploration and to break ties
                noise = np.random.normal(0, 0.01)
                score = q_value + u_value + rave_bonus + noise
                
                scores.append(score)
                moves.append((move, child))
            except Exception as e:
                print(f"Warning: Error calculating score for move {move}: {e}")
                continue
        
        # Handle case where all score calculations failed
        if not scores:
            return "pass", None
        
        # Select top-N moves safely
        top_n = min(top_n, len(scores))  # Adjust top_n if we have fewer moves
        indices = np.argsort(scores)[-top_n:]
        top_moves = [moves[i] for i in indices]
        top_scores = [scores[i] for i in indices]
        
        try:
            # Apply temperature scaling for exploration control
            temperature = max(0.1, min(1.0, 20.0 / (self.visits + 1)))
            scaled_scores = [s / temperature for s in top_scores]
            probs = softmax(scaled_scores)
            
            # Verify probabilities sum to 1 and are valid
            if not np.isclose(sum(probs), 1.0) or np.any(np.isnan(probs)):
                raise ValueError("Invalid probability distribution")
                
            sampled_index = np.random.choice(len(top_moves), p=probs)
            selected_move, selected_child = top_moves[sampled_index]
            
        except Exception as e:
            print(f"Warning: Error in move selection: {e}")
            # Fall back to highest scoring move
            selected_move, selected_child = top_moves[-1]
        
        return selected_move, selected_child


    def backup(self, reward):
        """
        Backup the reward through the tree, updating both standard and RAVE values.
        """
        node = self
        while node is not None:
            node.visits += 1
            node.value += reward

            # Update RAVE only if the move matches (or is considered equivalent)
            if node.parent and node.parent.move == self.move:  # Adjust this condition as needed
                node.rave_visits += 1
                node.rave_value += reward

            node = node.parent