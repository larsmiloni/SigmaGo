from goMCTS import MCTS

def ai_vs_ai():
    print('Slect AI mode:')
    print('1. Ai with MCTS vs Ai with MCTS')
    print('2. Ai with MCTS vs Ai with CNN policy')
    print('3. Ai with CNN policy vs Ai with CNN policy')
    mode = input()
    if mode == '1':

        ai_vs_ai_mcts()
    # elif mode == '2':
    #     ai_vs_ai_mcts_cnn()
    # elif mode == '3':
    #     from ai_vs_ai_cnn import ai_vs_ai_cnn
    #     ai_vs_ai_cnn()

def ai_vs_ai_mcts():
    mcts = MCTS() 

if __name__ == '__main__':
    print('Slect mode:')
    print('1. Play human vs AI')
    print('2. Play AI vs AI')
    mode = input()
    if mode == '1':
        print('Select player:')
        print('1. Human first')
        print('2. AI first')
        player = input()
        if player == '1':
            human_vs_ai()
        else:
            ai_vs_human()
    else:
        ai_vs_ai()



def create_ai_match(network_player1: Type[tf.keras.Model], network_player2:Type[tf.keras.Model]) -> Type[tf.keras.Model]:
    """
    Simulates a game between two Ai players.
    
    Args:
        network_player1 (tf.keras.Model): Pre-trained neural network model for player 1.
        network_player12 (tf.keras.Model): Pre-trained neural network model for player 2.
    
    Returns:
        Type[tf.keras.Model]: AI player model.
    """
    mcts_simulations = 20
    mcts_player1 = MCTS(network_player1, num_simulations=mcts_simulations/5)
    mcts_player2 = MCTS(network_player2, num_simulations=mcts_simulations)
    game = GoGame(size=9)
    game_states = []
    mcts_policies = []
    move_count = 0
    current_player = 'black'  # Track current player
    
    while not game.state[govars.DONE].any():
        print(f"\nMove {move_count + 1} - {current_player}'s turn")
        current_state = copy.deepcopy(game.state)
        
        # Store current state before move
        game_states.append(current_state)
        
        # Get move from MCTS
        if current_player == 'black':
            move = mcts_player1.search(game)
        else:
            move = mcts_player2.search(game)

        
        # Make the move and update game state
        if move != "pass":
            
            #Make the move
            game.step(move)  
            # Create policy vector from the move
            policy = np.zeros(83)
            move_idx = move[0] * 9 + move[1]
            policy[move_idx] = 1
            mcts_policies.append(policy)
            
            # Print updated board
            print("\nBoard after move:")
            print_board_state(game)
            print(f"Move: {chr(65 + move[1] if move[1] < 8 else 66 + move[1])}{9 - move[0]}")
            
            # Switch players
            current_player = 'white' if current_player == 'black' else 'black'
            move_count += 1
            
            print('Move count:', move_count)

        
        else:
            print(f"{current_player} passes")
            move = game.step("pass")
            current_player = 'white' if current_player == 'black' else 'black'
    
    # Game over - evaluate result
    winner = game.determine_winner()
    print(f"\nGame finished!")
    print(f"Winner: {'white' if winner == 1 else 'black'}") # Current player is the winner
    return MCTS(network)