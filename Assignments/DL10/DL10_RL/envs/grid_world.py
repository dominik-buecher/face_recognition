import numpy as np


class Grid:  # Environment
    def __init__(self, width, height, step_reward=-0.1, obstacles=2):
        # List of rewards for each possible state
        self.rewards = None

        # Set field size and starting position
        self.width = width
        self.height = height
        self.starting_position = (np.random.randint(0, self.height), np.random.randint(0, self.width))
        self.current_position = list(self.starting_position)

        # Create a random grid
        self.rewards, self.terminal_states, self.obstacle_fields = self.create_random_grid(obstacles, step_reward)

    def create_random_grid(self, obstacles, step_reward):
        # Create random target and avoid positions, make sure they are not equal to each other or the starting position
        while True:
            target_position = np.array([np.random.randint(0, self.height), np.random.randint(0, self.width)])
            avoid_position = np.array([np.random.randint(0, self.height), np.random.randint(0, self.width)])
            if (target_position != avoid_position).any() and (target_position != self.starting_position).any() and \
                (avoid_position != self.starting_position).any():
                break
        # Create obstacle fields that can not be entered, make sure they do not equal other special fields
        obstacle_fields = []
        for i in range(obstacles):
            while True:
                obstacle_position = np.array([np.random.randint(0, self.height), np.random.randint(0, self.width)])
                if (obstacle_position != target_position).any() and \
                        (obstacle_position != self.starting_position).any() and \
                        (obstacle_position != avoid_position).any():
                    obstacle_fields.append(tuple(obstacle_position))
                    break

        # Create a dictionary of fields with their respective rewards
        reward_dict = {}
        for row in range(self.height):
            for col in range(self.width):
                reward_dict[(row, col)] = step_reward
        reward_dict[tuple(target_position)] = 1
        reward_dict[tuple(avoid_position)] = -1
        for o in obstacle_fields:
            reward_dict[tuple(o)] = 0

        # Create a list of terminal states
        terminal_states = [tuple(target_position), tuple(avoid_position)]

        return reward_dict, terminal_states, obstacle_fields

    def get_current_state(self):
        return tuple(self.current_position)

    def is_state_terminal(self):
        return tuple(self.current_position) in self.terminal_states

    def move(self, action):
        """Moves the player into one of four possible directions (Up, Down, Left, Right).

            Parameters:
                action (str): "U", "D", "L" or "R"

            Returns:
                reward (float): The resulting reward for the chosen action
        """
        # Store the old position to revert to in case of an invalid move
        old_position = self.current_position.copy()

        # Execute the action
        if action == 'U':
            self.current_position[0] -= 1
        elif action == 'D':
            self.current_position[0] += 1
        elif action == 'R':
            self.current_position[1] += 1
        elif action == 'L':
            self.current_position[1] -= 1

        # Check if the resulting field is valid
        if tuple(self.current_position) in self.obstacle_fields or self.current_position[0] < 0 or \
                self.current_position[0] >= self.height or self.current_position[1] < 0 or \
                self.current_position[1] >= self.width:
            self.current_position = old_position

        # return a reward
        return self.rewards[tuple(self.current_position)]

    def reset(self):
        self.current_position = list(self.starting_position)

    def print_grid_rewards(self):
        for row in range(self.height):
            print("------"*self.width)
            for col in range(self.width):
                r = self.rewards[(row, col)]
                if r >= 0:
                    print(" {:.2f}|".format(r), end="")
                else:
                    print("{:.2f}|".format(r), end="")
            print("")
        print("")

    def print_grid(self):
        for row in range(self.height):
            print("------"*self.width)
            for col in range(self.width):
                r = self.rewards[(row, col)]
                if (np.array([row, col]) == np.array(self.starting_position)).all():
                    print("  s  |", end="")
                elif r == 0:
                    print("  x  |", end="")
                elif r == 1:
                    print("  t  |", end="")
                elif r == -1:
                    print("  a  |", end="")
                else:
                    print("     |", end="")
            print("")
        print("------"*self.width)
        print("")

    def print_state_values(self, value_dict):
        for row in range(self.height):
            print("------"*self.width)
            for col in range(self.width):
                if (row, col) in self.terminal_states or (row, col) in self.obstacle_fields:
                    print("     |", end="")

                else:
                    r = value_dict[(row, col)]
                    if r >= 0:
                        print(" {:.2f}|".format(r), end="")
                    else:
                        print("{:.2f}|".format(r), end="")
            print("")
        print("")

    def print_policy(self, action_value_dict):
        for row in range(self.height):
            print("------"*self.width)
            for col in range(self.width):
                if (row, col) in self.terminal_states or (row, col) in self.obstacle_fields:
                    print("     |", end="")
                else:
                    r = action_value_dict[(row, col)]
                    print("  {}  |".format(r), end="")
            print("")
        print("")

    def print_state_visits(self, state_visit_dict):
        state_visit_sum = 0
        for key, val in state_visit_dict.items():
            state_visit_sum += val
        print("Total states visited: ", state_visit_sum)

        for row in range(self.height):
            print("------"*self.width)
            for col in range(self.width):
                if (row, col) in self.terminal_states or (row, col) in self.obstacle_fields:
                    print("     |", end="")
                else:
                    print(" {:.2f}|".format(state_visit_dict[(row, col)]/state_visit_sum), end="")
            print("")
        print("")


if __name__ == '__main__':
    g = Grid(4, 4, obstacles=4)
    g.print_grid()
    g.print_grid_rewards()
