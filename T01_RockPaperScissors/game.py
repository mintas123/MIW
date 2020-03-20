import numpy as np


class SingleResult:
    def __init__(self, move_1, move_2, score):
        self.move_1 = move_1
        self.move_2 = move_2
        self.score = score


moves = ['r', 'p', 's']
# equal chances
p_start = ['0.333', '0.333', '0.334']
# calculated % of users choice, when game starts its 1/3,1/3,1/3
p_calculated = p_start


def interpret_results(results):
    total_sum = 0
    r_sum = 0
    r_to_p = 0
    r_to_r = 0
    r_to_s = 0
    p_sum = 0
    p_to_p = 0
    p_to_r = 0
    p_to_s = 0
    s_sum = 0
    s_to_p = 0
    s_to_r = 0
    s_to_s = 0
    for r in results:
        if r.move_1 == 'r':
            r_sum += 1
            if r.move_2 == 'r':
                r_to_r += 1
            if r.move_2 == 'p':
                r_to_p += 1
            if r.move_2 == 's':
                r_to_s += 1

        if r.move_1 == 'p':
            p_sum += 1
            if r.move_2 == 'r':
                p_to_r += 1
            if r.move_2 == 'p':
                p_to_p += 1
            if r.move_2 == 's':
                p_to_s += 1
        if r.move_1 == 's':
            s_sum += 1
            if r.move_2 == 'r':
                s_to_r += 1
            if r.move_2 == 'p':
                s_to_p += 1
            if r.move_2 == 's':
                s_to_s += 1
        total_sum += 1
    return [p_sum / total_sum, r_sum / total_sum, s_sum / total_sum,  # its a mess but first 3 are user preferences
            r_to_r / r_sum, r_to_p / r_sum, r_to_s / r_sum,
            p_to_r / p_sum, p_to_p / p_sum, p_to_s / p_sum,  # chance of moving from x to y
            s_to_r / s_sum, s_to_p / s_sum, s_to_s / s_sum]


def game_move(mode):

    if mode.casefold() == 'y':
        player_move = input('Write "r", "p" or "s"')
    else:
        player_move = np.random.choice(moves, replace=True, p=p_calculated)

    ai_move = np.random.choice(moves, replace=True, p=p_start)  # opponent pc has no skew
    score = -999

    if player_move == ai_move:
        score = 0
    elif player_move == 'r':
        if ai_move == 'p':
            score = -1
        elif ai_move == 's':
            score = 1
    elif player_move == 'p':
        if ai_move == 'r':
            score = 1
        elif ai_move == 's':
            score = -1
    elif player_move == 's':
        if ai_move == 'r':
            score = -1
        elif ai_move == 'p':
            score = 1

    return SingleResult(player_move, ai_move, score)


file = open("results.txt", "w")  # in file primary_results there are my data I base diagram and matrix on
flag = True
while flag:
    won_games = 0
    draws = 0
    games = 30
    result = []
    choice = input('Do you want to play by yourself? (y/n)')

    for i in range(games):
        res = game_move(choice)
        if res.score == 1:
            won_games += 1
        if res.score == 0:
            draws += 1
        if res.score == -999:
            i -= 1

        if choice.casefold() == 'y':
            print(f'Opponent: {res.move_2}, result:  {res.score} \n')

        result.append(res)
        file.write(f'{res.move_1} VS  {res.move_2}  =>  {res.score} \n')

    file.write(f'You won {won_games} times, there was a draw {draws} times, lost {games - won_games - draws} times\n')
    print(f'You won {won_games} times, there was a draw {draws} times, lost {games - won_games - draws} times\n')

    prob = interpret_results(result)
    p_calculated = prob[0:3]  # setting the probability of the Ai to mimic player

    file.write("ANALYSIS: \n\n")
    file.write(f"User probabilities of choosing  R, P, S : {prob[0:3]} \n")
    file.write(f"Choices after Rock - R, P, S: {prob[3:6]} \n")
    file.write(f"Choices after Paper - R, P, S: {prob[6:9]} \n")
    file.write(f"Choices after Scissors - R, P, S: {prob[9:12]} \n\n")
    file.write("_______________ \n\n")

    play_again = input("Again? (y/n)")

    if play_again.casefold() == "n":
        file.close()
        print("check result file.")
        flag = False
