import numpy as np

states = [
    'seek_enemy',  # 1
    'seek_medkit',  # 2
    'fight',  # 3
    'flee',  # 4

    'victory',  # 5
    'defeat']  # 6

final = ['win', 'defeat']  # accepting states

# transitions
transitions = [
    ['found_enemy', 'apply_medkit', 'low_on_hp'],  # 1
    ['noticed_enemy', 'found_medkit'],  # 2
    ['died', 'killed_enemy', 'low_on_hp'],  # 3
    ['low_on_hp']]  # 4

# transition probabilities for each transition according to the transition list
transition_probabilities = [
    [0.5, 0.35, 0.15],
    [0.5, 0.5],
    [0.35, 0.35, 0.3],
    [1.0],
    [1.0],
    [1.0]]


def play():
    # start from, q0
    current_state = 'seek_enemy'
    # in this case we always start at 'seek_enemy', so the initial probability vector (for all command) is:
    # [1, 0, 0, 0, 0, 0]

    # store all steps taken
    steps = [current_state]

    # Accumulate probabilities for victory or defeat
    prob = 1

    while True:

        # state 1
        if current_state == 'seek_enemy':

            # randomly choose a transition
            change = np.random.choice(transitions[0], replace=True, p=transition_probabilities[0])  # from seek_enemy

            # every transition has certain probability
            if change == 'nothing':
                prob *= 0.2  # prop = prob * 0.2
                current_state = current_state

            elif change == 'found_enemy':
                prob *= 0.5
                current_state = 'fight'

            elif change == 'apply_medkit':
                prob *= 0.15
                current_state = current_state

            elif change == 'low_on_hp':
                prob *= 0.15
                current_state = 'seek_medkit'

            # append the new state to the history of steps
            steps.append(current_state)

        # state 2
        elif current_state == 'seek_medkit':

            change = np.random.choice(transitions[1], replace=True, p=transition_probabilities[1])  # from seek_medkit

            if change == 'noticed_enemy':
                prob = prob * 0.5
                current_state = 'flee'

            elif change == 'found_medkit':
                prob = prob * 0.5
                current_state = 'seek_enemy'

            steps.append(current_state)
        # state 3
        elif current_state == 'fight':

            change = np.random.choice(transitions[2], replace=True, p=transition_probabilities[2])  # from fight

            if change == 'died':
                prob = prob * 0.35
                current_state = 'defeat'

            elif change == 'killed_enemy':
                prob = prob * 0.35
                current_state = 'victory'

            elif change == 'low_on_hp':
                prob = prob * 0.3
                current_state = 'flee'

            steps.append(current_state)
        # state 4
        elif current_state == 'flee':

            change = np.random.choice(transitions[3], replace=True, p=transition_probabilities[3])  # from flee

            if change == 'low_on_hp':
                prob = prob * 1.0
                current_state = 'seek_medkit'

            steps.append(current_state)
        # state 5 - accepting
        elif current_state == 'victory':
            return 'Victory', prob, steps

        # state 6 - accepting
        elif current_state == 'defeat':
            return 'Defeat', prob, steps


results = []

for i in range(30):
    res = play()
    results.append(res)

file = open("results.txt", "w")

i = 1
for game in results:
    file.write(f'Game #{i} \n')
    file.write(f'Victory or Defeat?: {game[0]} \n')
    file.write(f'Probability: {game[1] * 100}% \n')
    file.write(f'Steps: {game[2]} \n')
    file.write('\n')

    i += 1

file.close()
