import numpy as np
import matplotlib.pyplot as plt
import random

# 1. Environment & Setup
grid = np.zeros((10, 10))
grid[1, 1:4] = 1; grid[3, 0:8] = 1; grid[5, 2:10] = 1; grid[7, 0:7] = 1
robot_pos = (0, 0)
victims = [(0, 9), (2, 5), (4, 1), (6, 8), (9, 0)]
all_points = [robot_pos] + victims 

# 2. Search Task: BFS Distance Matrix
def get_bfs_distance(start, end, grid):
    queue = [(start, 0)]
    visited = {start}
    while queue:
        (curr_x, curr_y), dist = queue.pop(0)
        if (curr_x, curr_y) == end: return dist
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = curr_x + dx, curr_y + dy
            if 0 <= nx < 10 and 0 <= ny < 10 and grid[nx, ny] == 0 and (nx, ny) not in visited:
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
    return float('inf')

dist_matrix = np.zeros((6, 6))
for i in range(6):
    for j in range(6):
        dist_matrix[i, j] = get_bfs_distance(all_points[i], all_points[j], grid)

def calculate_fitness(chromosome):
    total_dist = dist_matrix[0, chromosome[0]]
    for i in range(len(chromosome)-1):
        total_dist += dist_matrix[chromosome[i], chromosome[i+1]]
    return total_dist

# --- GA Operators ---

def ordered_crossover(p1, p2):
    # Special crossover for permutations to avoid duplicates
    size = len(p1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[start:end] = p1[start:end]
    p2_remaining = [item for item in p2 if item not in child]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = p2_remaining[idx]
            idx += 1
    return child

def swap_mutation(chromosome):
    # Optimization Technique: Inorder Mutation
    idx1, idx2 = random.sample(range(len(chromosome)), 2)
    chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
    return chromosome

def evolve_ga():
    population = [random.sample([1, 2, 3, 4, 5], 5) for _ in range(20)]
    for generation in range(100):
        population = sorted(population, key=lambda x: calculate_fitness(x))
        new_gen = population[:2] # Elitism: keep best 2
        while len(new_gen) < 20:
            # Selection
            parent1, parent2 = random.sample(population[:10], 2)
            # Crossover
            if random.random() < 0.8: # Crossover Rate 80%
                child = ordered_crossover(parent1, parent2)
            else:
                child = parent1[:]
            # Mutation
            if random.random() < 0.2: # Mutation Rate 20%
                child = swap_mutation(child)
            new_gen.append(child)
        population = new_gen
    return population[0]

# 4. Results & Visualization
best_order = evolve_ga()
print(f"Optimal Order: {best_order}")
print(f"Total Steps: {calculate_fitness(best_order)}")

plt.imshow(grid, cmap='binary')
plt.scatter([p[1] for p in victims], [p[0] for p in victims], color='red', label='Victims')
plt.scatter(robot_pos[1], robot_pos[0], color='blue', label='Robot')
path_coords = [robot_pos] + [victims[i-1] for i in best_order]
for i in range(len(path_coords)-1):
    p1, p2 = path_coords[i], path_coords[i+1]
    plt.plot([p1[1], p2[1]], [p1[0], p2[0]], 'g--')
plt.legend(); plt.title("Rescue Mission: Full GA Pipeline"); plt.show()