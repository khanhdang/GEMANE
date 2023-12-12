from HeterGenMap.TaskGraph import TaskGraph
from HeterGenMap.Task import Task
from HeterGenMap.Utils import Coordinate
from HeterGenMap.SystemX import SystemX
import numpy as np
import matplotlib.pyplot as plt
import pickle
from joblib import dump, load

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

testcase = 3
NoC_dim = 2
pop_size = 1
n_gen = 1
n_toursel_mem = 1
n_init_mem = 0 

enable_plot = False

if testcase == 0:
    # Test - 0
    TG = TaskGraph("MLP", 46, [20, 20, 6], 20)
    pop_size = 10
    n_gen = 10
    n_toursel_mem = 5
    
    if NoC_dim == 3:
        s = SystemX(Coordinate(2, 2, 4), 6) # 
    elif NoC_dim == 2:
        s = SystemX(Coordinate(1, 4, 4), 6) # 
    else:
        print("unsupported  NoC_dim!")
        exit()

elif testcase == 1:
    # Test - 1
    TG = TaskGraph("MLP", 4096, [2000, 2000, 96], 2000)
    pop_size = 100
    n_gen = 80
    n_toursel_mem = 5
    
    if NoC_dim == 3:
        s = SystemX(Coordinate(2, 2, 4), 256) # 
    elif NoC_dim == 2:
        s = SystemX(Coordinate(1, 4, 4), 256) # 
    else:
        print("unsupported  NoC_dim!")
        exit()
elif testcase == 2:
    # Test - 2
    TG = TaskGraph("MLP", 16384, [10000, 5000, 1300, 84], 2000)
    pop_size = 100
    n_gen = 200
    n_toursel_mem = 5
    if NoC_dim == 3:
        s = SystemX(Coordinate(4, 4, 4), 256) # 
    elif NoC_dim == 2:
        s = SystemX(Coordinate(1, 8, 8), 256) #  
    else:
        print("unsupported  NoC_dim!")
        exit()


elif testcase == 3:
    # Test - MNIST 
    TG = TaskGraph("MLP", 4010, [2000, 2000, 10], 784)
    pop_size = 100
    n_gen = 80
    n_toursel_mem = 5
    if NoC_dim == 3:
        s = SystemX(Coordinate(2, 2, 4), 256) # 
    elif NoC_dim == 2:
        s = SystemX(Coordinate(1, 4, 4), 256) # 
    else:
        print("unsupported  NoC_dim!")
        exit()
elif testcase == 4:
    # Test - CNN CIFAR10 
    TG = TaskGraph("CNN", 25098, [16384,  8192,  512, 10], 3072) #784,400,10
    pop_size = 100
    n_gen = 200
    n_toursel_mem = 5

    if NoC_dim == 3:
        s = SystemX(Coordinate(4, 4, 4), 256) # 
    elif NoC_dim == 2:
        s = SystemX(Coordinate(1, 8, 8), 256) # 
    else:
        print ("unsupported testcase!")
        exit()
else:
    print ("unsupported testcase!")
    exit()

print(">> Perform without FT")

s.gen_routing_cost()

s.naive_assigment(TG.n_tasks)

finalSol, finalFitness, log =  TG.assign_address("GA_DEAP", s,  i_algo_type="eaMuPlusLambda", i_pop_size=pop_size, i_n_generations=n_gen, i_n_members_in_toursel=n_toursel_mem, n_init_mem=n_init_mem)

print(">>>> the final communication cost GA")
GA_cost, GA_dist = TG.CommCostMLP(s)

print("cost: "+str(GA_cost))


s.gen_RTL_simul_conf(TG, "conf/in_conf/test-2", "conf/out_conf/test-2")


print(">> Perform Linear")


TG.assign_address("linearXYZ", s)



print(">> the final communication cost XYZ")

XYZ_cost, XYZ_dist = TG.CommCostMLP(s)

print("cost: "+str(XYZ_cost))


