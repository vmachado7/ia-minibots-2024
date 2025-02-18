import random
import operator
import numpy as np
from deap import base, creator, tools, gp, algorithms

# Definimos el espacio (10x10) y la posición de los ingenieros
dim = 10
engineers = [tuple(e) for e in [(random.randint(0, dim-1), random.randint(0, dim-1)) for _ in range(5)]]

# Funciones primitivas
def move_up(pos):
    print(f"Moving up from {pos}")
    return (pos[0], max(0, pos[1]-1))

def move_down(pos):
    print(f"Moving down from {pos}")
    return (pos[0], min(dim-1, pos[1]+1))

def move_left(pos):
    print(f"Moving left from {pos}")
    return (max(0, pos[0]-1), pos[1])

def move_right(pos):
    print(f"Moving right from {pos}")
    return (min(dim-1, pos[0]+1), pos[1])

def distance_to_nearest(pos):
    distance = min(np.linalg.norm(np.array(pos) - np.array(e), ord=1) for e in engineers)
    print(f"Distance from {pos} to nearest engineer: {distance}")
    return distance

def deliver(pos, score):
    pos = tuple(pos)  # Ensure position is a tuple
    if pos in engineers:
        print(f"Delivering at {pos}, new score: {score + 1}")
        return score + 1
    print(f"No engineer at {pos}, score remains: {score}")
    return score

# Configuración de GP
pset = gp.PrimitiveSet("MAIN", 1)  # 1 argumento (posición del robot)
pset.addPrimitive(move_up, 1)
pset.addPrimitive(move_down, 1)
pset.addPrimitive(move_left, 1)
pset.addPrimitive(move_right, 1)
pset.addPrimitive(distance_to_nearest, 1)
pset.addPrimitive(deliver, 2)
pset.addTerminal((random.randint(0, dim-1), random.randint(0, dim-1)))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def eval_robot(individual):
    func = gp.compile(individual, pset)
    pos = (random.randint(0, dim-1), random.randint(0, dim-1))
    print(f"Starting position: {pos}")
    score = 0
    for _ in range(20):  # 20 movimientos
        pos = func(pos)
        if not isinstance(pos, tuple) or len(pos) != 2 or not all(isinstance(i, int) for i in pos):
            print(f"Invalid position generated: {pos}, penalizing.")
            return 0,  # Penalización si la función genera valores inválidos
        score = deliver(pos, score)
    print(f"Final score: {score}")
    return score,

toolbox.register("evaluate", eval_robot)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("mutate", gp.mutUniform, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    print("Starting evolutionary algorithm...")
    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=True)
    print("Best solution found:", hof[0])

if __name__ == "__main__":
    main()
