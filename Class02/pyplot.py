import numpy as np
from matplotlib import pyplot as plt

speed = [1, 2, 3, 4, 5]  # Y-axis
time = np.arange(start=0, stop=2.5, step=0.5)  # X-axis, from 0(inclusive) over to 2.5(exclusive) with the step 0.5

plt.plot(time, speed, label='Speed change over time') # plot X and Y, give the line a label which will show up as legend

plt.xlabel('Time [s]') # laleb X-axis
plt.ylabel('Speed [km/h]') # label Y-axis

plt.legend() # enable legend

plt.show() # show the result

# scatter plot

num_elements = 50

# a dictionary containing data describing our scatter plot
data = {
    'x': np.arange(num_elements), # a numpy array of an integer sequence from 0(inclusive) to 50(exclusive)
    'y': np.random.randn(num_elements), # generate 50 random float numbers
    'colors': np.random.randint(0, 70, num_elements), # a numpy array of an integers from 0(inclusive) to 50(exclusive) os the size 50
    'diameters': np.random.randn(num_elements) * 300 # the same as for 'y' but scaled by 300
}

plt.scatter('x', 'y', c='colors', s='diameters', data=data) # passing the dictionary as data and giving the keys as X, Y, colors and dot sizes

plt.show()

# 3 subplots in one figure

groups = ['Pepsi', 'Fanta', 'Sprite', 'Coca-Cola Zero']
amount = [1, 1, 5, 7]

plt.figure(figsize=(18, 6)) # modify the size of the figure

# 3-digit numbers represent rows"columns|indeces
plt.subplot(131)
plt.bar(groups, amount)

plt.subplot(132)
plt.scatter(groups, amount)

plt.subplot(133)
plt.plot(groups, amount)

plt.show()