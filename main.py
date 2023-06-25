# -*- coding: utf-8 -*-

### Import libraries
import numpy as np

"""### Env Setup"""

# env variables
environment = [
    [-10, 1, 0],
    [0, -10, 10]
]
grid_rows = len(environment)
grid_cols = len(environment[0])

num_actions = 4

# Define the reward matrix
rewards = np.full((grid_rows, grid_cols), environment)

# Define the Q-Table
q_values = np.zeros((grid_rows, grid_cols, num_actions))

# Set hyper-parameters
alpha = 0.1  # Learning Rate
gamma = 0.9  # Discount Factor
epsilon = 0.1  # Exploration Rate

# Actions
actions = ['up', 'right', 'down', 'left']

print(rewards)

"""### Functions Definitions"""

def isTerminal(current_row, current_col):
    if rewards[current_row, current_col] == 10:
        return False
    else:
        return True

def startLocation():
    current_row = np.random.randint(grid_rows)
    current_col = np.random.randint(grid_cols)

    while isTerminal(current_row, current_col):
        current_row = np.random.randint(grid_rows)
        current_col = np.random.randint(grid_cols)

    return current_row, current_col

def getNextAction(current_row, current_col, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row, current_col])
    else:
        return np.random.randint(4)

def getNextLocation(current_row, current_col, action):
    new_row = current_row
    new_col = current_col

    if actions[action] == 'up' and current_row > 0:
        new_row -= 1
    elif actions[action] == 'right' and  current_col < grid_cols - 1:
        new_col += 1
    elif actions[action] == 'down' and current_row < grid_rows - 1:
        new_row += 1
    elif actions[action] == 'left' and current_col > 0:
        new_col -= 1

    return new_row, new_col

def getShortestPath(start_row, start_col):
    if isTerminal(start_row, start_col):
        return []
    else:
        current_row, current_col = start_row, start_col
        shortest_path = []
        shortest_path.append([current_row, current_col])
        while not isTerminal(current_row, current_col):
            action = getNextAction(current_row, current_col, 1.)

            current_row, current_col = getNextLocation(
                current_row,
                current_col,
                action
            )
            shortest_path.append([current_row, current_col])
        return shortest_path

"""### Main function"""

for episode in range(20000):
    row_index, col_index = startLocation()

    # Initialize moves on each iteration
    moves = 0

    while True:
        action_index = getNextAction(row_index, col_index, epsilon)

        old_row_index, old_col_index = row_index, col_index
        row_index, col_index = getNextLocation(row_index, col_index, action_index)

        reward = rewards[row_index, col_index]
        old_q_value = q_values[row_index, col_index, action_index]

        td = reward + (gamma * np.max(q_values[row_index, col_index])) - old_q_value

        new_q_value = old_q_value + (alpha * td)
        q_values[old_row_index, old_col_index, action_index] = new_q_value

        # Increase moves
        moves+=1
        # If reward is equal 10 or moves are greater than or equal to 4
        if reward == 10 or moves>=4:
          break

print('Q Values from Q Table.')
print(q_values)