from goMCTS import create_ai_match
from policy_network import PolicyNetwork

def ai_vs_ai():
    print('Slect AI mode:')
    print('1. Ai👾 with MCTS vs Ai👾 with MCTS')
    print('2. Ai👾 with MCTS vs Ai👾 with CNN policy')
    print('3. Ai👾 with CNN policy vs Ai👾 with CNN policy')
    mode = input()
    if mode == '1':

        ai_vs_ai_mcts()
    elif mode == '2':
        ai_vs_ai_mcts_cnn()
    elif mode == '3':
        ai_vs_ai_cnn()

def ai_vs_ai_mcts():
    network_player1 = PolicyNetwork("./models/PN_R3_C64_IMPROVED_MODEL.pt")
    network_player2 = PolicyNetwork("./models/PN_R3_C64_IMPROVED_MODEL_2.pt") 
    create_ai_match(network_player1, network_player2, isBothMCTS=True)

def ai_vs_ai_mcts_cnn():
    network_player1 = PolicyNetwork("./models/PN_R3_C64.pt")
    network_player2 = PolicyNetwork("./models/PN_R3_C64_IMPROVED_MODEL.pt") 
    create_ai_match(network_player1, network_player2, isCNNvsMCTS=True)

def human_vs_ai():
    network = PolicyNetwork("./models/PN_R3_C64.pt")
    create_ai_match(network, None, isHumanvsAi=True)

def ai_vs_human():
    network = PolicyNetwork("./models/PN_R3_C64.pt")
    create_ai_match(network, None, isAivsHuman=True)

def ai_vs_ai_cnn():
    network_player1 = PolicyNetwork("./models/PN_R3_C64.pt")
    network_player2 = PolicyNetwork("./models/PN_R3_C64_IMPROVED_MODEL.pt") 
    create_ai_match(network_player1, network_player2, isBothCNN=True)

if __name__ == '__main__':
    print('Slect mode:')
    print('1. Play human👨 vs AI👾')
    print('2. Play AI👾 vs AI👾')
    mode = input()
    if mode == '1':
        print('Select player:')
        print('1. Human👨 first')
        print('2. AI👾 first')
        player = input()
        if player == '1':
            human_vs_ai()
        else:
            ai_vs_human()
        
    else:
        ai_vs_ai()



