from utils import *
import csv
import argparse



#HPs
generations = 1000
max_res = 10
mut_rate = 0.3
hp_rate = 0.4
hp_alpha = 0.5
frequence_update = 50
pop_size = 100
type_of_task = 'supervised_tasks' 
task_name = 'chaotic_forecasting'


if __name__ == "__main__":

    #run main evolution loop
    (best_individual, min_fitness, mean_fitness, population, fitness, avg_f,
     best_f, pop_saving, fitness_saving) = evolving_res(max_res, pop_size, task_name, generations,
                                                               frequence_update, hp_rate, hp_alpha, mut_rate,
                                                               type_of_task=type_of_task)

    #print info
    print(len(population))
    print("evolution finished")
    print("Average fitness covid: ",mean_fitness)
    print("Best fitness covid: ", min_fitness)
    print('Best Individual covid: ', best_individual)

    #get best indivudal graph
    g = adjacency_matrix_to_graph_2(best_individual)
    g.render("best individual covid", view = True)

