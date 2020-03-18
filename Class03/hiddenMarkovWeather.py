import numpy as np
# initial states and initial probability vector
start = ['Rainy', 'Sunny']
p_start = [0.2, 0.8]


# hidden states and transition probability matrix
t = ['Rainy', 'Sunny']
tm = [
    # rainy # sunny
    [0.4, 0.6], # rainy
    [0.3, 0.7] # sunny
]

# observations (emissions) and output (emissions) probability matrix
e = ['Walk', 'Shop', 'Clean', 'visit friends']
ep = [
    # walk # shop # clean  # friends
    [0.1, 0.3, 0.2, 0.4],  # rainy
    [0.6, 0.2, 0.1, 0.1]   # sunny
]

# chose an initial state based on the probabilities given to them
initial = np.random.choice(start, replace=True, p=p_start)
state = initial

#
days = 10

for day in range(1, days+1):
    print(f'Day {day}')
    if state == 'Rainy':
        activity = np.random.choice(e, replace=True, p=ep[1])
        state = np.random.choice(t, replace=True, p=tm[0])

    elif state == 'Sunny':
        activity = np.random.choice(e, replace=True, p=ep[1])
        state = np.random.choice(t, replace=True, p=tm[1])

    print(f'It is:\t {state}')
    print(f'I will:\t {activity}')
    print('\n')