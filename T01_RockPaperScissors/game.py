import numpy as np

# initial states and probability vector
moves = ['Rock', 'Paper', 'Scissors']
p_start = ['0.33', '0.33', '0.34']

final = ['win', 'defeat']


def play():
    player_move = np.random.choice(moves, replace=True, p=p_start)
    # player_move = input('Write "r", "p" or "s"')
    ai_move = np.random.choice(moves, replace=True, p=p_start)

    if player_move == 'Rock':
        if ai_move == 'Rock':
            return 0
        elif ai_move == 'Paper':
            return -1
        elif ai_move == 'Scissors':
            return 1

    if player_move == 'Paper':
        if ai_move == 'Rock':
            return 1
        elif ai_move == 'Paper':
            return 0
        elif ai_move == 'Scissors':
            return -1

    if player_move == 'Scissors':
        if ai_move == 'Rock':
            return -1
        elif ai_move == 'Paper':
            return 1
        elif ai_move == 'Scissors':
            return 0


won_games = 0
draws = 0
games = 100000
for i in range(games):
    res = play()
    if res == 1:
        # print('You won')
        won_games += 1
    if res == 0:
        # print('draw')
        draws += 1
    # if res == -1:
        # print('You lost')
print(f'You won {won_games} times, there was a draw {draws} times, lost {games-won_games-draws} times')
