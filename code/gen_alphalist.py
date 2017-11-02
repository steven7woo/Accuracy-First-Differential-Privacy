import numpy as np
import math

step = 0.005
largest = 0.2
largest = int(largest/step) * step
num_steps = int(largest/step)
alphas = np.linspace(step, largest, num_steps)

print("\n".join(list(map(str, alphas))))
