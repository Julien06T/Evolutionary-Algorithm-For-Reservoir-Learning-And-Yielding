#imports
import pickle
from sklearn.linear_model import RidgeCV
from joblib import Parallel, delayed
import glob
import pandas as pd
import os
import gymnasium as gym
from sklearn.metrics import mean_absolute_error, mean_squared_error
import reservoirpy as rpy
from reservoirpy.nodes import *
from reservoirpy.model import Model
from reservoirpy.observables import rmse
import numpy as np
import graphviz
import stream_benchmark as sb
#import stream_benchmark as sb
from ray.util.joblib import register_ray
from scipy.stats import loguniform

# init joblib's ray backend (more robust to high memory usage than loky backend)
register_ray()

# Set reservoirpy verbosity to 0 (deprecated in 0.4)
rpy.verbosity(0)

# Pretty print individuals
np.set_printoptions(precision=3, suppress=True, linewidth=120)

#number of time a model is evaluated (n_instances different seeds)
n_instances = 5

# auxillary function to count amount of reservoirs in an individual
def count_reservoirs(pop):
    return [ind.shape[0]-2 for ind in pop]

# auxillary function to count the amount of connections (links) between reservoirs in an individual
def count_connections(pop):

    con = []
    fb_conn = []
    direct_conn = []
    for ind in pop:
        divider = np.where(np.all(ind == -1, axis=0))[0][0]
        con_matrix = ind[:, :divider]
        direct_counter = 0
        fb_counter = 0
        for i in range(len(con_matrix)):
            for j in range(len(con_matrix[i])):
                if con_matrix[i][j] == 1:
                    direct_counter += 1
                elif con_matrix[i][j] == 2:
                    fb_counter += 1
        fb_conn.append(fb_counter)
        direct_conn.append(direct_counter)
        con.append(fb_counter+direct_counter)

    return con, direct_conn, fb_conn

# check the amount of units (overall neurons) within an individual
def check_units(pop, best):

    units= []

    if best:
        ind = pop[0]

        divider = np.where(np.all(ind == -1, axis=0))[0][0]


        HP = ind[:, divider+1:]

        for row in range(len(HP)):
            for col in range(len(HP[row])):
                #print(HP[row][col])
                if col == 0:
                    units.append(HP[row][col])





        units = units[1:-1]
        min_unit, max_unit = min(units), max(units)
        median = np.median(units)
        mean = np.mean(units)
        return units, min_unit, max_unit, median, mean

    else:
        for i in range(10):
            units = []
            ind = pop[90+i]

            divider = np.where(np.all(ind == -1, axis=0))[0][0]
            #print(ind)

            HP = ind[:, divider + 1:]

            for row in range(len(HP)):
                for col in range(len(HP[row])):
                    # print(HP[row][col])
                    if col == 0:
                        units.append(HP[row][col])

            units = units[1:-1]
            min_unit, max_unit = min(units), max(units)
            median = np.median(units)
            mean = np.mean(units)
            print(units)

def logspace_mutation(x, sigma, rng=None):
    rng = rng or np.random.default_rng()
    return np.exp(np.log(x) + sigma * rng.normal())

# initialise the population
def init_pop(max_res: int):
    return _is_valid_test_RO(has_cycle_RO(generate_random_individual_RO(max_res)), max_res)

# function to evaluate a child
def eval_childs(o, pop_size, population, fitness, hp_alpha, mut_rate, hp_rate, max_res, task_name,
                task_function, gen):

    # select parents for crossover
    parents = parent_selection(pop_size, population, fitness)

    # define alpha (% of p1 and 100-% of p2) for vertical slice
    alpha = np.random.uniform(1, 99)

    # perform cross over
    new_ind = cross_over_RO(parents[0], parents[1], alpha, hp_alpha)

    # sometimes it may happen that individual has only 1 row (invalid shape)
    # re-do the crossover
    while new_ind.shape[0] == 1 :
        print(new_ind)
        parents = parent_selection(pop_size, population, fitness)
        alpha = np.random.uniform(1, 99)
        new_ind = cross_over_RO(parents[0], parents[1], alpha, hp_alpha)
           
    # perform mutation - for now no adaptative rate
    new_ind = mutation(new_ind, mut_rate, hp_rate)

    # check validity
    np.fill_diagonal(new_ind, 0)
    # turn cycles into feedback connection
    new_ind = has_cycle_RO(new_ind)
    # return valid individual
    new_ind = _is_valid_test_RO(new_ind, max_res)
    np.fill_diagonal(new_ind, 0)

    # Evaluate new individual
    f_new = task_function(new_ind, task_name)
    print("New fitness: {}".format(f_new))

    return f_new, new_ind

# Main function
def evolving_res(max_res : int, pop_size : int, task_name : str or list ,generations : int, frequence_update : int,
                 hp_alpha : float, hp_rate : float, mut_rate : float, type_of_task : str, exploitation = False,
                 stagnation = False):

    # create directories for logs
    task_name_doc = task_name+"100n_job2"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, type_of_task, "FINAL_RUNS",task_name_doc)
    os.makedirs(log_dir, exist_ok=True)

    # add 1 to max_res (input/output doesnt count)
    max_res += 1

    # create dictionary for mapping the correct function to the task type
    task_functions = {
        'covid_task': covid_task_ridge,
        'RL': None,
        'supervised_tasks': evaluation,
        'supervised_multi': evaluate_multi_stream_task,
        'dummy task easy': fake_task_easy,
        'dummy task impossible': fake_task_impossible,
        'dummy task impossible_2' : fake_task_impossible_random_f,
        'rl': eval_task_RL,
        #'sb_eval': evaluate_stream_task_2
    }
    task_function = task_functions.get(type_of_task)

    # HPs
    fitness = []
    best_f = []
    avg_f = []
    var_f = []
    pop_saving = []
    fitness_saving = []
    best_f_tracker = 0
    ratio_fitness_stop = 0.02 # 2% of initial fitness variance
    ratio_counter = 0 # counter to continue the EA for a few gen after the ratio is too low
    max_gen_after_low_ratio = 10 # continue for 10 gen
    new_best = False


    # init population
    pop_init = Parallel(n_jobs=-1, backend="ray")(delayed(init_pop)(max_res) for _ in range(pop_size))

    population = list(pop_init)


    # evaluate the population
    print('Evaluating initial population...')
    print("Population size: {}".format(pop_size))

    def evaluate_individual(ind):
        score = task_function(ind, task_name)
        return score

    #evaluate the individuals
    results = Parallel(n_jobs=-1, backend="ray")(delayed(evaluate_individual)(i) for i in population)

    # get the initial fitness and std
    fitness = list(results)
    initial_fitness_std = np.std(fitness)


    #print gen 0 infos
    print('Generation 0')
    print("Average fitness: ", np.mean(fitness))
    print("Best fitness: ", np.min(fitness))
    best_f.append(np.min(fitness))
    min_f = np.min(fitness)
    avg_f.append(np.mean(fitness))
    var_f.append(np.std(fitness))
    print('Best individual: ')
    print(population[np.argmin(fitness)])

    # begin evolution
    for i in range(generations):

        # various info about current state
        print("Stagnation {}".format(stagnation))
        print("Exploitation {}".format(exploitation))
        print('Best F tracker {}'.format(best_f_tracker))

        # get the 50 offsprings and their fitness
        new_gen = Parallel(n_jobs=64, backend="ray")(delayed(eval_childs)(o, pop_size, population, fitness, hp_alpha, mut_rate,
                                                           hp_rate, max_res, task_name,
                                                           task_function, i) for o in range(frequence_update))

        # get the results from Parallel joblib
        f_new, new_inds = zip(*new_gen)
        fitness_child = list(f_new)
        pop_child = list(new_inds)

        # add up the old + new fitnesses and population
        fitness = fitness + fitness_child
        population = population + pop_child


        # sort both list based on increasing fitness
        sort = sorted(zip(fitness, population), key=lambda p: p[0])

        # (fit, pop) best 100 out of 150
        fitness, population = zip(*sort[:pop_size])

        # Set the fitness and pop for next gen
        fitness = list(fitness)
        population = list(population)

        # Check if new best f has been obtained
        if np.min(fitness) < min_f:
            min_f = np.min(fitness)
            best_f_tracker = 0


        # increment best f tracker
        best_f_tracker += 1

        # append logs
        best_f.append(np.min(fitness))
        avg_f.append(np.mean(fitness))
        var_f.append(np.var(fitness))
        pop_saving.append(population)
        fitness_saving.append(fitness)

        # saving - checkpoints
        np.savetxt(log_dir+"/best_f", best_f, delimiter=",")
        np.savetxt(log_dir+"/AverageF", avg_f, delimiter=",")
        np.savetxt(log_dir + "/varF", var_f, delimiter=",")


        #if (i+1) % 10 == 0: #save every 10 gens
        p_f_path = log_dir+"/pop_fit/pop_fit_gen_"+str(i+1)
        os.makedirs(p_f_path, exist_ok=True)
        population_path = os.path.join(p_f_path, 'population.pkl')
        fitness_path = os.path.join(p_f_path, 'fitness.pkl')
        with open(population_path, 'wb') as f:
            pickle.dump(population, f)
        with open(fitness_path, 'wb') as f:
            pickle.dump(fitness, f)

        # Print statement when no new best f since 100 generations
        if best_f_tracker == 100:
            print('-------------- no best_f since 100 gen')
           # return population[np.argmin(fitness)], np.min(fitness), np.mean(
            #    fitness), population, fitness, avg_f, best_f, pop_saving, fitness_saving

            # Check new fitness variance
        new_fitness_variance = np.std(fitness)
        fitness_ratio = new_fitness_variance / initial_fitness_std
        if fitness_ratio < ratio_fitness_stop:
            ratio_counter += 1

        # log info
        print('#' * 10)
        print("Generation " + str(i + 1))
        print("Average fitness: ", np.mean(fitness))
        print("Best fitness: ", np.min(fitness))
        print("Variance Fitness: ", var_f[-1])
        print("Ratio Fitness: ", fitness_ratio)
        print('Best Individual: ')
        print(population[np.argmin(fitness)])



        #if ratio_counter == max_gen_after_low_ratio:
         #   return population[np.argmin(fitness)], np.min(fitness), np.mean(
          #      fitness), population, fitness, avg_f, best_f, pop_saving, fitness_saving

    return population[np.argmin(fitness)], np.min(fitness), np.mean(fitness), population, fitness, avg_f, best_f, pop_saving, fitness_saving

#parent selection function
def parent_selection(pop_size, population, fitness):

    # perform 2 parents tournament selection
    parents = []
    # Variable to check whether parent 1 != parent 2
    p1_id = -1

    # repeat process 2 times
    for _ in range(2):

        # randomly picks two parents in the pop
        parent_1_index = np.random.randint(0, pop_size - 1)
        parent_2_index = np.random.randint(0, pop_size - 1)


        # if id 1 == id 2, pick a new second until different
        while parent_1_index == parent_2_index == p1_id:
            parent_2_index = np.random.randint(0, len(population) - 1)

        #parent_1_index, parent_2_index = np.random.choice(range(len(population)), size=2, replace=False)

        # select best between the two
        best_parent = np.argmin([fitness[parent_1_index], fitness[parent_2_index]])

        if best_parent == 0:
            parents.append(population[parent_1_index])
            p1_id = parent_1_index

        else:
            parents.append(population[parent_2_index])
            p1_id = parent_2_index

    return parents

#adjacency matrixto graph
def adjacency_matrix_to_graph_2(matriz):

    # only keep the connections | discard the HPs for the graph
    divider = np.where(np.all(matriz == -1, axis=0))[0][0]
    matrix = matriz[:, :divider]
    dot = graphviz.Digraph(format="png")
    HP = matriz[:, divider + 1:]

    num_nodes = len(matrix[0])  # Number of columns determines nodes

    node_names = ["Res" + str(i) for i in range(1, num_nodes - 1)]  # Exclude last two columns
    input_node = "Input"
    output_node = "Readout"


        # Add nodes
    dot.node(input_node)  # Input node (first row)
    for name in range(len(node_names)):
        dot.node(node_names[name])#, label=node_names[name] + '\nUnits: ' + str(HP[name+1,0]) +
          #' Lr: ' + str(HP[name+1,1]) +
          #' SR: ' + str(HP[name+1,2]) +
         # ' IS: ' + str(HP[name+1,3]),
    #fontsize="10" )
    dot.node(output_node, label=output_node )  # Readout node (last column)

        # Add edges
    for i in range(len(matrix)):  # Iterate over rows (starting nodes)
        for j in range(len(matrix[i])):  # Iterate over columns (ending nodes)
            if matrix[i][j] != 0:
                if i == 0:  # First row -> Input node
                    start_node = input_node
                elif i < len(node_names) + 1:  # Regular reservoir nodes
                    start_node = node_names[i - 1]
                else:  # Last row should never be a starting node
                    start_node = output_node

                        # Determine the ending node
                if j == len(matrix[i]) - 1:  # Last column -> Readout
                    end_node = output_node
                elif j > 0 and j < len(node_names) + 1:  # Regular reservoir nodes
                    end_node = node_names[j - 1]
                elif j == 0:
                    end_node = output_node
                else:
                    print('ignore')# Ignore invalid cases
                    continue

                if matrix[i][j] == 1 and i != matrix.shape[0]-1:
                    color = 'green'
                else:
                    color = 'red'


                dot.edge(start_node, end_node, color=color)

    return dot

# generate random individual with their corresponding HPs
def generate_random_individual_RO(max_res):

    # first generate amount of reservoirs
    reservoirs = (np.random.randint(3, max_res+1))

    # generate connections randomly
    res_con = np.random.randint(0,2, (reservoirs+1, reservoirs+1))

    # generate HP
    res_con = np.hstack((res_con, -1*np.ones((res_con.shape[0], 1))))

    HP = np.zeros((len(res_con), 8))
    #Iterate over reservoirs
    for i in range(1, len(res_con)-1):

        # iterates over HPs
        for j in range(8):

        # create units
            if j == 0:
                HP[i][j] = int(loguniform(25, 100).rvs())

        # create lr
            elif j == 1:
                HP[i][j] = float(loguniform(1e-4, 1).rvs())

        # create input scaling
            elif j == 2:
                HP[i][j] = float(loguniform(1e-4, 1e1).rvs())

        # create sr
            elif j == 3:
                HP[i][j] = float(loguniform(1e-4, 1e1).rvs())

        # create fb scaling:
            elif j == 4:
                HP[i][j] = float(loguniform(1e-4, 1e1).rvs())

        # create input_conn
            elif j == 5:
                HP[i][j] = float(loguniform(1e-2, 1).rvs())

        # create res conn
            elif j == 6:
                HP[i][j] = float(loguniform(1e-1, 1).rvs())

        # create feedback_conn
            elif j == 7:
                HP[i][j] = float(loguniform(1e-2, 1).rvs())



    return np.hstack((res_con, HP))

# cross_over function
def cross_over_RO(parent_1, parent_2, alpha, hp_alpha):

    # divide connections and HPs
    divider1 = np.where(np.all(parent_1 == -1, axis=0))[0][0]
    divider2 = np.where(np.all(parent_2 == -1, axis=0))[0][0]

    # connections
    parent1 = parent_1[:-1, :divider1]
    parent2 = parent_2[:-1, :divider2]

    # readout_fb
    ro_1 = parent1[-1, :divider1]
    ro_2 = parent2[-1, :divider2]


    # hp
    HP1 = parent_1[:-1, divider1:]
    HP2 = parent_2[:-1, divider2:]


    # calculate % to take of each parents
    parent1_len = round(alpha * parent1.shape[1] / 100)
    parent2_len = round(alpha * parent2.shape[1] / 100)

    # vertical slice
    p1_part = parent1[: , :parent1_len]
    p2_part = parent2[: , parent2_len:]

    # take the smallest parent (we need a NxN individual : reservoirs can't be dead-branch or startnodes )
    smallest_parent = np.argmin([p1_part.shape[0], p2_part.shape[0]])

    # if parent 1 is smallest
    if smallest_parent == 0:
        # horizontally stack smallest + largest with rows until smallest
        new_ind = np.hstack((p1_part, p2_part[:p1_part.shape[0], :]))
    # if parent2 is smallest, do the opposite
    else:
        # horizontally stack smallest + largest with rows until smallest
        new_ind = np.hstack((p1_part[:p2_part.shape[0], :], p2_part))


    # if num rows != num cols -1 (leads to dead end or no inputs res)
    if new_ind.shape[0] != new_ind.shape[1] - 1:
        # if parent1 is smallest
        if smallest_parent == 0:
            # take the completing part: rows from on where p1 stops until end (match amount of cols)
            # and take the amount of col that match new_ind

            dif = parent2.shape[1] - new_ind.shape[1]

            complete_part = parent2[p1_part.shape[0]:p1_part.shape[0]+new_ind.shape[1]-1-new_ind.shape[0], dif:]

            # vertically stack
            new_ind = np.vstack((new_ind, complete_part))

        # if parent 2 is smallest, do the opposite
        else:

            complete_part = parent1[p2_part.shape[0]:p2_part.shape[0]+new_ind.shape[1]-1-new_ind.shape[0], :new_ind.shape[1]]
            new_ind = np.vstack((new_ind, complete_part))

    # Define child HP as empty list
    HP_child = np.zeros((new_ind.shape[0], 9))


    # HP always has 8 columns - number of rows can diverge depending on the amount of reservoirs
    for i in range(len(new_ind)):

        # Select 3 place to cut
        c1, c2, c3 = np.sort(np.random.choice(np.arange(2, len((HP_child[0])) - 1), 3, replace=False))
        # Check if current i element is still available with the corresponding parents
        if i >= HP1.shape[0]:
            HP_child[i, :] = HP2[i, :]
        elif i >= HP2.shape[0]:
            HP_child[i, :] = HP1[i, :]

        # If everything is good, cut accordingly
        else:
            HP_child[i][:c1] = HP1[i][:c1]
            HP_child[i][c1:c2] = HP2[i][c1:c2]
            HP_child[i][c2:c3] = HP1[i][c2:c3]
            HP_child[i][c3:] = HP2[i][c3:]

    # construct the readout (last row) of the new individual
    ro_1_part = ro_1[:parent1_len]
    ro_2_part = ro_2[parent2_len:]
    ro = np.hstack((ro_1_part, ro_2_part))
    ro = np.hstack((ro, [-1, 0, 0, 0, 0, 0, 0, 0, 0]))

    #concat back to match structure
    new_ind = np.hstack((new_ind, HP_child))
    new_ind = np.vstack((new_ind, ro))

    return new_ind


# generate HP of reservoir
def generate_hp():
    hp = [-1, int(loguniform(25, 100).rvs()), float(loguniform(1e-4, 1).rvs()), float(loguniform(1e-4, 1e1).rvs()), float(loguniform(1e-4, 1e1).rvs()), float(loguniform(1e-4, 1e1).rvs()), float(loguniform(1e-2, 1).rvs()), float(loguniform(1e-1, 1).rvs()),float(loguniform(1e-2, 1).rvs())]
    return hp


def mutation(individual, rate, hp_rate):
    min_units, max_units = 25, 100

    # divide into connections and HPs
    divider = np.where(np.all(individual == -1, axis=0))[0][0]
    ind = individual[:, :divider]
    HP = individual[:-1, divider:]
    ro = individual[-1, divider:]


    # ---------- CONNECTION MUTATION ----------
    def mutate_connection(val):
        options = [0, 1, 2]
        options.remove(val)
        return val if np.random.rand() >= 0.5 else np.random.choice(options)

    for i in range(len(ind)):
        for j in range(1, ind.shape[1] - 1):  # skip first col
            if np.random.rand() < rate:
                ind[i, j] = mutate_connection(ind[i, j])
    ind[:, 0] = 0  # first column always 0

    # ---------- RESERVOIR MUTATION ----------
    generated_res = 0
    if np.random.rand() < rate:
        new_col = np.random.randint(0, 2, (ind.shape[0], 1))
        new_row = np.random.randint(0, 2, (1, ind.shape[1] + 1))
        ind = np.hstack((ind, new_col))
        ind = np.vstack((ind, new_row))
        generated_res = 1
    elif np.random.rand() < rate:
        ind = np.delete(ind, -1, axis=0)
        ind = np.delete(ind, -2, axis=1)
        generated_res = -1

    # ---------- READOUT MUTATION ----------
    for i in range(len(ro)):
        if np.random.rand() < rate and ro[i] != -1:
            ro[i] = 0 if ro[i] else 1

    # ---------- HP MUTATION ----------
    hp_bound = {
        1: (25, 100, int),  # units
        2: (1e-4, 1, float),  # lr
        3: (1e-4, 1e1, float),  # input scaling
        4: (1e-4, 1e1, float),  # sr
        5: (1e-4, 1e1, float),  # fb scaling
        6: (1e-2, 1, float),  # input connectivity
        7: (1e-1, 1, float),  # reservoir connectivity
        8: (1e-2, 1, float)  # fb connectivity
    }

    sigma = 0.5 # Learning Rate
    for i in range(len(HP)):
        if i == 0:
            continue
        for j in range(1, HP.shape[1]):

            if np.random.rand() < hp_rate:

                hp_old = HP[i, j]
                HP[i, j] = logspace_mutation(HP[i, j], sigma)
                low, high, type = hp_bound[j]
                if HP[i, j] < low or HP[i, j] > high:
                    HP[i, j] = hp_old

                if type is int:
                    HP[i, j] = int(round(HP[i, j]))

    # ---------- ADD/REMOVE HP ROW FOR RESERVOIR ----------
    if generated_res == 1:
        HP = np.vstack((HP, generate_hp()))
    elif generated_res == -1:
        HP = HP[:-1, :]

    # ---------- RECONSTRUCT ----------
    HP = np.vstack((HP, ro))
    return np.hstack((ind, HP))



#check validity
def _is_valid_test_RO(ind, max_res):

    #divide connections/HP
    divider = np.where(np.all(ind == -1, axis=0))[0][0]
    ind_hp = ind[:, divider:]

    ro = ind[-1, :divider]
    #connections
    ind = ind[:-1, :divider]

    #1st column == 0
    ind[:, 0] = 0


    # delete self connection
    np.fill_diagonal(ind, 0)

    # Check for a dead end nodes - row == 0
    dead_end_horizontal = [i for i in range(ind.shape[0]) if np.all(ind[i] == 0)]

    # create a connection from that reservoir -> noise -> readout
    # create an additional column (connection to noise) with a 1 on the i-th row (dead-end res)
    # add epsilon row with a 1 in the column -2 (one before last)

    if dead_end_horizontal:

        #for now, simply add a connection to output
        for i in dead_end_horizontal:
            ind[i, -1] = 1



    #check for no input reservoir (empty column)
    dead_end_vertical = [i for i in range(ind.shape[1]) if np.all(ind[:, i] == 0)]
    dead_end_vertical = dead_end_vertical[1:]  # Input can't receive so always 0

    #if there is an empty col
    if dead_end_vertical:
        #if reservoir has no entrance: connect it to input?
        #add 1 at the top (input --> res)
        ind[0, dead_end_vertical[0]] = 1


    # no input connected to reservoirs - first row is empty
    no_input = [i for i in range(1, ind.shape[1] - 1) if np.all(ind[0, i] == 0)]

    #if list begins with 1 --> means there is a connection so we skip this part
    if 1 in no_input:
        #add input connection to each reservoir lacking just to be safe
        #print(ind)
        #add connection random < 0.5
        #print(no_input)
        while np.all(ind[0, :] == 0):
            for i in no_input:
                if np.random.uniform(low=0, high=1) < 0.4:
                    ind[0, i] = 1
            np.fill_diagonal(ind, 0)


    # set 1st column (input) to be full of zeros as we can't send to input
    ind[:, 0] = 0

    #check if no reservoir is connected to an output
    no_out = [i for i in range(1, ind.shape[0]) if np.all(ind[i, -1] == 0)]

    # add output conection to reservoir with the most connections in
    if len(no_out) == ind.shape[0]-1: # mean all reservoir rows have no out
        #take the res index with most in- conn
        selected_col = np.argmax(np.count_nonzero(ind, axis=0))
        if ind.shape[0] != 1:
            ind[selected_col, -1] = 1


    #check for only fb connection (2) but no input 1 - rasie an error has it cannot send new info
    only_fb_conn = [i for i in range(ind.shape[1]) if np.all(ind[:, i] != 1)] # check in column if all != 1

    #check 0 and plusieru 2
    # if col contains only 0 and 2
    for i in only_fb_conn:
        #if connection is a 2
        if i != 0:
            #switch to a 1
            ind[0, i] = 1

    #fill diagonal with 0 to delete self connection again
    np.fill_diagonal(ind, 0)

    ind[:, -1] = np.where(ind[:, -1] == 2, 1, ind[:, -1])


    ro[-1] = 0
    ro[0] = 0

    ind = np.vstack((ind, ro))
    return np.hstack((ind, ind_hp))


#adaptive ridge
def matrix_to_esn_ridge(ind, seed):
    #assign input
    #assign reservoirs in list
    #if added noise, define noise
    #define last column as readout

    input = Input()
    reservoirs = []
    model = Model()

    divider = np.where(np.all(ind == -1, axis=0))[0][0]
    ind_hp = ind[:, divider+1:]
    #readout = Ridge(ridge=ridge)
    readout = ScikitLearnNode(RidgeCV, model_hypers=dict(alphas = np.logspace(-8, 3, num=12), store_cv_results=True))
    #readout = ScikitLearnNode(RidgeCV, model_hypers=dict(alphas = np.linspace(10**-10, 10**, num=100)))

    units = [ind_hp[i][0] for i in range(1, ind_hp.shape[0])]
    lrs = [ind_hp[i][1] for i in range(1, ind_hp.shape[0])]
    iss = [ind_hp[i][2] for i in range(1, ind_hp.shape[0])]
    srs = [ind_hp[i][3] for i in range(1, ind_hp.shape[0])]
    fbs = [ind_hp[i][4] for i in range(1, ind_hp.shape[0])]
    in_conn = [ind_hp[i][5] for i in range(1, ind_hp.shape[0])]
    res_conn = [ind_hp[i][6] for i in range(1, ind_hp.shape[0])]
    fb_conn = [ind_hp[i][7] for i in range(1, ind_hp.shape[0])]





    #get amount of reservoirs
    for i in range(len(ind)-1):
        if i != 0:
            reservoirs.append(Reservoir(units=int(units[i-1]), lr=lrs[i-1], sr=srs[i-1], input_scaling=iss[i-1], input_connectivity=in_conn[i-1], rc_connectivity=res_conn[i-1], fb_connectivity=fb_conn[i-1], fb_scaling=fbs[i-1],seed=seed))

    ind = ind[:, :divider]
    #get res idx of input --> res
    input_to_res_idx = ind[0] == 1
    #print('ind')
    #print(ind)
    #print('input_to_res_idx')
    #print(input_to_res_idx)

    in_to_out = input_to_res_idx[-1]
    input_to_res_idx = input_to_res_idx[1:-1]
    #print('after slice')
    #print(input_to_res_idx)
    #loop over the input --> res_ids
    for i in range(len(input_to_res_idx)):
        #if there is a true
        if input_to_res_idx[i]:
            #print('current')
            #print(input_to_res_idx[i])
            #define connection Input --> res

            model &= input >> reservoirs[i]
            #print('corresponding  res')
            #print(reservoirs[i])

    #If last index of input to res == True --> create Input() --> readout
    if in_to_out:
        #print('in --> out')
        #print(in_to_out)
        model &= input >> readout

    #create res --> res connection - start at 1 because 0 == Input (already treated above)
    for i in range(1, len(ind)-1):

        #id connections
        conn_res = ind[i, 1:-1]
        conn_to_readout = ind[i, -1]

        #loop over res to res conn
        for res_id in range(len(conn_res)):
            #check if 1 (direct connection)
            if conn_res[res_id] == 1:
                model &= reservoirs[i-1] >> reservoirs[res_id]
            #check if 2 (fb connection)
            elif conn_res[res_id] == 2:
                model &= reservoirs[i-1] << reservoirs[res_id]

        #check for connection to readout
        if conn_to_readout == 1:
            model &= reservoirs[i-1] >> readout

    for i in range(len(ind)):
        if i == len(ind) - 1:
            for j in range(len(ind[i])):
                if (ind[i][j] == 1 or ind[i][j] == 2) and j != 0:
                    model &= reservoirs[j-1] << readout


    return model




#check cycles
def has_cycle_RO(adj_matrix):
    #print(adj_matrix)
    #adj_matrix = adj_matrix[0]
    #print(adj_matrix.shape)
    divider = np.where(np.all(adj_matrix == -1, axis=0))[0][0]

    hp = adj_matrix[:, divider:]

    ro = adj_matrix[-1, :divider]
    adj_matrix = adj_matrix[:-1, :divider]


    n = len(adj_matrix)

    # Initialize distance (reachability) matrix and predecessor list for each node
    dist = [[adj_matrix[i][j] for j in range(n)] for i in range(n)]
    pred = [[-1 for _ in range(n)] for _ in range(n)]  # Store multiple predecessors

    # Floyd-Warshall Algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] != 0 and dist[k][j] != 0:
                    dist[i][j] = 1  # Mark as reachable
                    pred[i][j] = k  # Track predecessor

    cycles = set()  # Use a set to store unique cycles

    # Function to reconstruct cycles by backtracking predecessors
    def reconstruct_cycle(start, current, path, visited):
        if current in visited:  # Cycle detected
            if current == start:
                cycle = tuple(sorted(path + [start]))  # Sort to ensure uniqueness
                cycles.add(cycle)
            return

        visited.add(current)
        path.append(current)

        for next_node in range(n):
            if adj_matrix[current][next_node]:
                #edges.append((current, next_node))# Explore all outgoing edges
                reconstruct_cycle(start, next_node, path[:], visited.copy())

    # Look for cycles by checking reachability (dist[i][i] == 1)
    for i in range(n):
        if dist[i][i]:  # Cycle detected
            reconstruct_cycle(i, i, [], set())

    #check for the highest resevoir (e.g res3 > res1) and replace the 1 by 2 --> allow to differentiate direct conn to fb conn
    already_sent = []
   # print(cycles)
    for i in cycles:
        #print(i)
        if len(i) > 2:
            # Avoid self-loops and trivial cycles
            sorted_nodes = sorted(set(i))
            #print(sorted_nodes)# Unique nodes sorted

            #add probability
            fb_send = sorted_nodes[-1]
            #print(fb_send)# Highest node in cycle

            #PROBLEM IS PROBABLY HERE
            #NEED TO CHECK WHEN IT SHOULD BE NODE -2 OR NODE 0
            fb_receive_candidate = sorted_nodes[:-1]
            #print('cand',fb_receive_candidate)# Lowest node in cycle
            #print(already_sent, fb_send, fb_receive_candidate)

            for fb_receive in fb_receive_candidate:
                if (fb_send, fb_receive) not in already_sent:
                    #stochasticité - check ca car erreur
                    adj_matrix[fb_send, fb_receive] = 2
                    #if adj_matrix[fb_send, fb_receive] == 1:
                        #if np.random.uniform() < 0.5:
                            #
                        #else:
                            #adj_matrix[fb_receive, fb_send] = 2
                    already_sent.append((fb_send, fb_receive))

    adj_matrix = np.vstack((adj_matrix, ro))

    return np.hstack((adj_matrix, hp))

def evaluate_stream_task(ind, task_name, seed_i):

    #rajouter un param pour la difficulté de la tache
    task_data = sb.build_task(task_name, difficulty='small', seed=seed_i)

    X_train = task_data['X_train']
    Y_train = task_data['Y_train']
    T_train = task_data['T_train']
    X_valid = task_data['X_valid']
    Y_valid = task_data['Y_valid']
    T_valid = task_data['T_valid']
    X_test = task_data['X_test']
    Y_test = task_data['Y_test']
    T_test = task_data['T_test']




    # define models, and ridge
    models = []
    #create models with their seed
    for seed in range(n_instances):
        models.append(matrix_to_esn_ridge(ind, seed=seed))

   
    fitness = []

    #loop over each seeds
    counter = 0
    for seed in range(len(models)):

        #train model

        models[seed].fit(X_train, Y_train)

        #get predictions
        preds = models[seed].run(X_test, return_states=True)

        #print(preds)

        # get only the readout state
        readout_key = next((k for k in preds.keys() if k.startswith('ScikitLearnNode')), None)



        #get predictions
        Y_pred = np.reshape(preds[readout_key], Y_test.shape)

        # Compute score
        score = sb.compute_score(Y_test, Y_pred, T_test, task_data['classification'])
        fitness.append(score)


    return np.mean(fitness)




def evaluation(ind, task_name):

    # lists to keep track of all important variable
    scores = []
    task_seed = 5
    #evaluate over different sequence seed
    #for seed_i in range(task_seed):
    score = evaluate_stream_task(ind, task_name, 16)
    scores.append(score)


    return np.mean(scores)













 
















 





