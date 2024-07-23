import numpy as np
from nashpy import Game

# Step 1: Read input matrices
n, m = map(int, input().split())

profits_abolf = [list(map(int, input().split())) for _ in range(n)]
profits_behzad = [list(map(int, input().split())) for _ in range(n)]

# Step 2: Create the game
game = Game(np.array(profits_abolf), np.array(profits_behzad))

# Step 3: Compute Nash equilibrium using Lemke-Howson algorithm
eqs = game.lemke_howson(initial_dropped_label=0)

# Step 4: Print strategies
abolf_strategy = list(map(lambda x: round(x, 6), eqs[0]))
behzad_strategy = list(map(lambda x: round(x, 6), eqs[1]))

print(*abolf_strategy)
print(*behzad_strategy)
