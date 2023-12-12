import numpy as np
import math
# import pickle
import random
# from joblib import dump, load

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

from HeterGenMap.SystemX import SystemX

class GA_DEAP:
    def __init__(self, algo_type, pop_size, n_generations, n_members_in_toursel, n_init_mem, MU, LAMBDA, topo, n_inputs, sysX):

        self.algo_type = algo_type
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.n_members_in_toursel = n_members_in_toursel
        self.MU = MU
        self.LAMBDA = LAMBDA
        self.topo = np.concatenate(([n_inputs], topo), axis=0)  # the first one count for input layer (no neuron)
        self.mut_parent = False
        self.mutate_prob = 0.2 #0.5
        self.sysX =  sysX

        self.P_C = 0.7
        self.P_M = 0.3

        self.n_Systems = sysX.n_chips # TODO: currently no input since HeterGenMap has no inter-system
        self.system_dim_Y = sysX.system_dim.Y 
        self.system_dim_X = sysX.system_dim.X
        self.system_dim_Z = sysX.system_dim.Z

        self.addr_IO = sysX.interface_address
        self.inter_sys_IO_cost = 10 # TODO: need to fill

        self.enable_plot = False
        self.costfunction = 'sumall'

        self.mutation_func1 = 0.3
        self.mutation_func2 = 0.3
        self.mutation_func3 = 0.3
        
        self.init_case = 0
        self.max_init_cases  = n_init_mem
        if self.max_init_cases > 1 and self.system_dim_Z == 1:
            self.max_init_cases = 1

    def __generate_random_solution(self):
        unmap_topo = self.topo.copy()
        n_NNlayers = len(self.topo)
        popMem = np.zeros((self.n_Systems, self.system_dim_Z, self.system_dim_Y,
                            self.system_dim_X,  n_NNlayers), dtype=int)

        for sys in range(self.n_Systems):
                for z in range(self.system_dim_Z):
                    for y in range(self.system_dim_Y):
                        for x in range(self.system_dim_X):
                            popMem[sys, z, y, x, 0] = self.sysX.nodes[z][y][x].n_assigned_elements
        # normal gen
        if self.init_case >= self.max_init_cases:
            while(np.sum(unmap_topo[1:]) > 0): 
                for sys in range(self.n_Systems):
                    for z in range(self.system_dim_Z):
                        for y in range(self.system_dim_Y):
                            for x in range(self.system_dim_X):
                                if popMem[sys, z, y, x, 0] > 0 :
                                    n_unmapped_neuron = popMem[sys, z, y, x, 0].copy()
                                    for nl in range(1, n_NNlayers):
                                        # generate a random number between 0 and n_unmapped_neuron
                                        mapped = np.random.randint(0, n_unmapped_neuron+1)
                                        # if the number overflow the unmapped topo, truncate to it
                                        if (mapped > unmap_topo[nl]):
                                            mapped = unmap_topo[nl]
                                        # remove from unmap[ed_top
                                        unmap_topo[nl] -= mapped
                                        # remove from n_unmapped
                                        n_unmapped_neuron -= mapped
                                        # set to mapped
                                        popMem[sys, z, y, x, nl] += mapped
                                    # set back the value
                                    popMem[sys, z, y, x, 0] = n_unmapped_neuron
        
        else:
            if self.init_case == 0:
                while(np.sum(unmap_topo[1:]) > 0): 
                    for sys in range(self.n_Systems):
                        for z in range(self.system_dim_Z):
                            for y in range(self.system_dim_Y):
                                for x in range(self.system_dim_X):
                                    if popMem[sys, z, y, x, 0] > 0 :
                                        n_unmapped_neuron = popMem[sys, z, y, x, 0].copy()
                                        for nl in range(1, n_NNlayers):
                                            mapped = n_unmapped_neuron
                                            # if the number overflow the unmapped topo, truncate to it
                                            if (mapped > unmap_topo[nl]):
                                                mapped = unmap_topo[nl]
                                            # remove from unmap[ed_top
                                            unmap_topo[nl] -= mapped
                                            # remove from n_unmapped
                                            n_unmapped_neuron -= mapped
                                            # set to mapped
                                            popMem[sys, z, y, x, nl] += mapped
                                        # set back the value
                                        popMem[sys, z, y, x, 0] = n_unmapped_neuron
            elif self.init_case == 1:
                while(np.sum(unmap_topo[1:]) > 0): 
                    for sys in range(self.n_Systems):
                        for y in range(self.system_dim_Y):
                            for x in range(self.system_dim_X):
                                for z in range(self.system_dim_Z):
                                    if popMem[sys, z, y, x, 0] > 0 :
                                        n_unmapped_neuron = popMem[sys, z, y, x, 0].copy()
                                        for nl in range(1, n_NNlayers):
                                            mapped = n_unmapped_neuron
                                            # if the number overflow the unmapped topo, truncate to it
                                            if (mapped > unmap_topo[nl]):
                                                mapped = unmap_topo[nl]
                                            # remove from unmap[ed_top
                                            unmap_topo[nl] -= mapped
                                            # remove from n_unmapped
                                            n_unmapped_neuron -= mapped
                                            # set to mapped
                                            popMem[sys, z, y, x, nl] += mapped
                                        # set back the value
                                        popMem[sys, z, y, x, 0] = n_unmapped_neuron
            elif self.init_case == 2:
                print("gen 2 ")
                while(np.sum(unmap_topo[1:]) > 0): 
                    for sys in range(self.n_Systems):
                        for x in range(self.system_dim_X):
                            for y in range(self.system_dim_Y):
                                for z in range(self.system_dim_Z):
                                    if popMem[sys, z, y, x, 0] > 0 :
                                        n_unmapped_neuron = popMem[sys, z, y, x, 0].copy()
                                        for nl in range(1, n_NNlayers):
                                            mapped = n_unmapped_neuron
                                            # if the number overflow the unmapped topo, truncate to it
                                            if (mapped > unmap_topo[nl]):
                                                mapped = unmap_topo[nl]
                                            # remove from unmap[ed_top
                                            unmap_topo[nl] -= mapped
                                            # remove from n_unmapped
                                            n_unmapped_neuron -= mapped
                                            # set to mapped
                                            popMem[sys, z, y, x, nl] += mapped
                                        # set back the value
                                        popMem[sys, z, y, x, 0] = n_unmapped_neuron
            # Move to next case
            self.init_case = self.init_case + 1
        return popMem


    def generate_population(self):
        np.random.seed(0) # move  the setting seed to here
        popList = []
        for _ in range(self.pop_size):
            parent = self.__generate_random_solution()
            popList.append(parent)
        return popList

    def validate_pop(self, popList):
        validated = True
        # 1. Check the number of element each layers
        for p in popList:
            sum = np.zeros(len(self.topo), dtype=int)
            for sys in range(self.n_Systems):
                for z in range(self.system_dim_Z):
                    for y in range(self.system_dim_Y):
                        for x in range(self.system_dim_X):
                            for nl in range(0, len(self.topo)):
                                sum[nl] =  sum[nl] + p[sys, z, y, x, nl]
            if np.array_equal(sum[1:], self.topo[1:]):
                pass
            else:
                print(">>> ERROR: Incorrect population members")
                validated = False
        # 2. compare each solution
        # Note: if we set the seed in the generation method(__generate_random_solution), members are identical
        for p1 in range(self.pop_size):
            for p2 in range(self.pop_size):
                if p1 != p2:
                    if np.array_equal(popList[p1], popList[p2]):
                        # NOTE: no longer print, we generate a new member.
                        # WARNING: we must update the Fitness
                        # print(">>> ERROR: Two population members are identical: "+str(p1) +", "+str(p2) )
                        popList[p1] =  self.__generate_random_solution()
                        validated = False
        return validated


    def commcost(self, popMem, printing = False):
        # TODO: The inter-sys cost function is not correct
        if self.costfunction == 'sumall':
            costs = np.zeros((len(self.topo)), dtype=int)
            for nl in range(1,len(self.topo)-1):
                if printing:
                    print("layer:")
                    print(nl)
                for sys1 in range(self.n_Systems):
                    for z1 in range(self.system_dim_Z):
                        for y1 in range(self.system_dim_Y):
                            for x1 in range(self.system_dim_X):
                                for sys2 in range(self.n_Systems):
                                    for z2 in range(self.system_dim_Z):
                                        for y2 in range(self.system_dim_Y):
                                            for x2 in range(self.system_dim_X):
                                                if (popMem[sys1, z1, y1, x1, nl] > 0 and popMem[sys2, z2, y2, x2, nl+1] > 0):
                                                    if (sys1 == sys2):
                                                        # costs[nl] = costs[nl] + (abs(z1-z2) + abs(y1-y2) + abs(x1-x2))*popMem[sys1, z1, y1, x1, nl]
                                                        costs[nl] = costs[nl] + self.sysX.routing_cost[z1][y1][x1][z2][y2][x2]*popMem[sys1, z1, y1, x1, nl]
                                                        if printing:
                                                            print("at "+str(z1)+str(y1)+str(x1))
                                                            print("to "+str(z2)+str(y2)+str(x2))
                                                            print("has "+str(popMem[sys1, z1, y1, x1, nl]))
                                                            print(costs[nl])
                                                    else:
                                                        print("2sys > not supported")
                                                        input("----")
                                                        # costs[nl] = costs[nl] + (abs(self.addr_IO.Z-z1) + abs(self.addr_IO.Y-y1) + abs(self.addr_IO.X-x1) +\
                                                        #     abs(self.addr_IO.Z-z2) + abs(self.addr_IO.Y-y2) + abs(self.addr_IO.X-x2))*popMem[sys1, z1, y1, x1, nl]
                                                        costs[nl] = costs[nl] + (self.sysX.routing_cost[z1][y1][x1][self.addr_IO.Z][self.addr_IO.Y][self.addr_IO.X])*popMem[sys1, z1, y1, x1, nl]
                                                        ## If they are different syss
                                                        costs[nl] += self.inter_sys_IO_cost
            nl = len(self.topo)-1  # last layers to IO
            for sys1 in range(self.n_Systems):
                for z1 in range(self.system_dim_Z):
                    for y1 in range(self.system_dim_Y):
                        for x1 in range(self.system_dim_X):
                            if (popMem[sys1, z1, y1, x1, nl] > 0):
                                if printing:
                                    print(costs[nl])

                                # costs[nl] = costs[nl] + (abs(self.addr_IO.Z-z1) + \
                                #     abs(self.addr_IO.Y-y1) + abs(self.addr_IO.X-x1))*popMem[sys1, z1, y1, x1, nl]
                                costs[nl] = costs[nl] + (self.sysX.routing_cost[z1][y1][x1][self.addr_IO.Z][self.addr_IO.Y][self.addr_IO.X])*popMem[sys1, z1, y1, x1, nl]
                                if printing:
                                    print("at "+str(z1)+str(y1)+str(x1))
                                    print(costs[nl])

                                
            nl = 1  # first layers from IO
            for sys1 in range(self.n_Systems):
                for z1 in range(self.system_dim_Z):
                    for y1 in range(self.system_dim_Y):
                        for x1 in range(self.system_dim_X):
                            if (popMem[sys1, z1, y1, x1, nl] > 0):
                                # if printing:
                                #     print("at "+str(z1)+str(y1)+str(x1))
                                # costs[0] = costs[0] + abs(self.addr_IO.Z-z1) + \
                                #     abs(self.addr_IO.Y-y1) + abs(self.addr_IO.X-x1)
                                costs[0] = costs[0] + self.sysX.routing_cost[self.addr_IO.Z][self.addr_IO.Y][self.addr_IO.X][z1][y1][x1] 
                                # if printing:
                                #     print(costs[0])
            if printing:
                print(costs)
                print(np.sum(costs))
            return np.sum(costs),
        else:
            print("Unsupported cost function!")
            exit()

    def Pop_comcost(self, popList, printing = False):
        fitnessList = np.zeros(len(popList), dtype = int)
        for i in range(len(popList)):
            fitnessList[i] = self.commcost(popMem=popList[i], printing=printing)
        return fitnessList


    # TourSel for popList:
    # k: number of member selected for TourSel
    def tourSel(self, popList, fitnessList, k):
        parentIdxList = np.empty(shape=(len(popList), 2), dtype = int)
        for i in range(int(len(popList))):
            parent1 = -1
            parent2 = -1
            index = np.sort(np.random.choice(len(popList), k, replace=False))



            parent1 = index[np.argmin(fitnessList[index])]
            while (parent1 == parent2 or parent2 == -1):
                index = np.sort(np.random.choice(
                    len(popList), k, replace=False))
                parent2 = index[np.argmin(fitnessList[index])]
            

            parentIdxList[i] = [int(parent1), int(parent2)]

        return parentIdxList


    def random_combine(self, parentIdxList, popList):

        newpopList = []

        child = np.zeros((self.n_Systems, self.system_dim_Z, self.system_dim_Y,
                            self.system_dim_X,  len(self.topo)), dtype=int)
        for i in range(len(parentIdxList)):
            parent1_Idx = parentIdxList[i, 0]
            parent2_Idx = parentIdxList[i, 1]
            parent1 = popList[parent1_Idx]
            parent2 = popList[parent2_Idx]
            child =  np.floor((parent1 + parent2)/2).astype(int)
            self.fixoffspring(child)
            newpopList.append(child)
        self.validate_pop(newpopList)
        return newpopList

    def crossover(self, ind1, ind2):
        offspring1 =  np.floor((ind1 + ind2)/2).astype(int)
        offspring2 =  np.floor((ind1 + ind2)/2).astype(int)
        self.fixoffspring(offspring1)
        self.fixoffspring(offspring2)
        return (creator.Individual(offspring1), creator.Individual(offspring2))

    def fixoffspring(self, child):
        # Please ignore the first layers: for unmapped
        mapped_status = np.sum(np.sum(np.sum(np.sum(child, axis=1), axis = 1), axis = 1), axis = 0)
        umapped_status = self.topo - mapped_status
        for sys1 in range(self.n_Systems):
            for z1 in range(self.system_dim_Z):
                for y1 in range(self.system_dim_Y):
                    for x1 in range(self.system_dim_X):
                        n_unmapped = self.sysX.nodes[z1][y1][x1].n_assigned_elements - np.sum(child[sys1, z1, y1, x1, 1:])
                        # print("unmapped")
                        # print(n_unmapped)
                        # if n_unmapped > 1:
                            # print(child[sys1, :, :, :, :])
                            # print(child[sys1, z1, y1, x1, :])
                        while (n_unmapped > 0 ):
                            nl = np.random.randint(1, len(self.topo))
                            if n_unmapped > 0 and umapped_status[nl] > 0 and n_unmapped <= umapped_status[nl]:
                                # print(child[sys1, :, :, :, :])
                                # print(child[sys1, z1, y1, x1, :]) 
                                child[sys1, z1, y1, x1, nl] = child[sys1, z1, y1, x1, nl] + n_unmapped
                                umapped_status[nl] = umapped_status[nl] - n_unmapped
                                n_unmapped = 0
                            elif  n_unmapped > 0  and umapped_status[nl] > 0 :
                                # print(child[sys1, :, :, :, :])
                                # print(child[sys1, z1, y1, x1, :]) 
                                child[sys1, z1, y1, x1, nl] = child[sys1, z1, y1, x1, nl] + umapped_status[nl]
                                n_unmapped = n_unmapped - umapped_status[nl] 
                                umapped_status[nl] = 0 
        
        return child


    def new_mutation(self, child):
        # for child in cList:
        n_clusters = self.n_Systems*self.system_dim_X*self.system_dim_Y*self.system_dim_Z
        n_exchange = int(self.mutate_prob*n_clusters/2) # divide by two since I must find a pair
        rand_val = random.random()
        if rand_val <= self.mutation_func1:

            for _ in range(int(n_exchange)):
                # do the mutation!
                rand_sys1 = np.random.randint(self.n_Systems)            
                rand_X1 = np.random.randint(self.system_dim_X)
                rand_Y1 = np.random.randint(self.system_dim_Y)
                rand_Z1 = np.random.randint(self.system_dim_Z)
                rand_nl1 =  np.random.randint(1, len(self.topo))
                while child[rand_sys1, rand_Z1, rand_Y1, rand_X1, rand_nl1] == 0:
                    rand_nl1 =  np.random.randint(1, len(self.topo))
                    rand_sys1 = np.random.randint(self.n_Systems)            
                    rand_X1 = np.random.randint(self.system_dim_X)
                    rand_Y1 = np.random.randint(self.system_dim_Y)
                    rand_Z1 = np.random.randint(self.system_dim_Z)

                rand_sys2 = np.random.randint(self.n_Systems)
                rand_Y2 = np.random.randint(self.system_dim_Y)
                rand_Z2 = np.random.randint(self.system_dim_Z)
                rand_X2 = np.random.randint(self.system_dim_X)
                rand_nl2 =  np.random.randint(1, len(self.topo))

                while rand_nl2 ==  rand_nl1 or (rand_sys2 == rand_sys1 and rand_Z2 == rand_Z1 and rand_Y2 == rand_Y1 and rand_X2 ==  rand_X1) or child[rand_sys2, rand_Z2, rand_Y2, rand_X2, rand_nl2] == 0:
                    rand_sys2 = np.random.randint(self.n_Systems)
                    rand_Y2 = np.random.randint(self.system_dim_Y)
                    rand_Z2 = np.random.randint(self.system_dim_Z)
                    rand_X2 = np.random.randint(self.system_dim_X)
                    rand_nl2 =  np.random.randint(1, len(self.topo))
                
                # proceed to find the min value and swap

                if min(child[rand_sys1, rand_Z1, rand_Y1, rand_X1, rand_nl1], child[rand_sys2, rand_Z2, rand_Y2, rand_X2, rand_nl2]) > 0:
                    swap_val =  min(child[rand_sys1, rand_Z1, rand_Y1, rand_X1, rand_nl1], child[rand_sys2, rand_Z2, rand_Y2, rand_X2, rand_nl2])
                    child[rand_sys1, rand_Z1, rand_Y1, rand_X1, rand_nl1] = child[rand_sys1, rand_Z1, rand_Y1, rand_X1, rand_nl1] - swap_val
                    child[rand_sys1, rand_Z1, rand_Y1, rand_X1, rand_nl2] = child[rand_sys1, rand_Z1, rand_Y1, rand_X1, rand_nl2] + swap_val

                    child[rand_sys2, rand_Z2, rand_Y2, rand_X2, rand_nl1] = child[rand_sys2, rand_Z2, rand_Y2, rand_X2, rand_nl1] + swap_val
                    child[rand_sys2, rand_Z2, rand_Y2, rand_X2, rand_nl2] = child[rand_sys2, rand_Z2, rand_Y2, rand_X2, rand_nl2] - swap_val
        elif rand_val <= self.mutation_func1 + self.mutation_func2:
         
            for _ in range(int(n_exchange)):
                # do the mutation!
                rand_sys1 = np.random.randint(self.n_Systems)            
                rand_X1 = np.random.randint(self.system_dim_X)
                rand_Y1 = np.random.randint(self.system_dim_Y)
                rand_Z1 = np.random.randint(self.system_dim_Z)
                rand_nl1 =  np.random.randint(1, len(self.topo))
                while child[rand_sys1, rand_Z1, rand_Y1, rand_X1, rand_nl1] == 0:
                    rand_nl1 =  np.random.randint(1, len(self.topo))
                    rand_sys1 = np.random.randint(self.n_Systems)            
                    rand_X1 = np.random.randint(self.system_dim_X)
                    rand_Y1 = np.random.randint(self.system_dim_Y)
                    rand_Z1 = np.random.randint(self.system_dim_Z)

                rand_sys2 = np.random.randint(self.n_Systems)
                rand_Y2 = np.random.randint(self.system_dim_Y)
                rand_Z2 = np.random.randint(self.system_dim_Z)
                rand_X2 = np.random.randint(self.system_dim_X)
                rand_nl2 =  np.random.randint(1, len(self.topo))

                while rand_nl2 ==  rand_nl1 or (rand_sys2 == rand_sys1 and rand_Z2 == rand_Z1 and rand_Y2 == rand_Y1 and rand_X2 ==  rand_X1) or child[rand_sys2, rand_Z2, rand_Y2, rand_X2, rand_nl2] == 0:
                    rand_sys2 = np.random.randint(self.n_Systems)
                    rand_Y2 = np.random.randint(self.system_dim_Y)
                    rand_Z2 = np.random.randint(self.system_dim_Z)
                    rand_X2 = np.random.randint(self.system_dim_X)
                    rand_nl2 =  np.random.randint(1, len(self.topo))
                
                # proceed to find the min value and swap

                if min(child[rand_sys1, rand_Z1, rand_Y1, rand_X1, rand_nl1], child[rand_sys2, rand_Z2, rand_Y2, rand_X2, rand_nl2]) > 0 and child[rand_sys1, rand_Z1, rand_Y1, rand_X1, rand_nl2] > 0 and child[rand_sys2, rand_Z2, rand_Y2, rand_X2, rand_nl1] > 0:
                    swap_val =  min(child[rand_sys1, rand_Z1, rand_Y1, rand_X1, rand_nl1], child[rand_sys2, rand_Z2, rand_Y2, rand_X2, rand_nl2])
                    # print(swap_val)
                    # input(".....")
                    child[rand_sys1, rand_Z1, rand_Y1, rand_X1, rand_nl1] = child[rand_sys1, rand_Z1, rand_Y1, rand_X1, rand_nl1] - swap_val
                    child[rand_sys1, rand_Z1, rand_Y1, rand_X1, rand_nl2] = child[rand_sys1, rand_Z1, rand_Y1, rand_X1, rand_nl2] + swap_val

                    child[rand_sys2, rand_Z2, rand_Y2, rand_X2, rand_nl1] = child[rand_sys2, rand_Z2, rand_Y2, rand_X2, rand_nl1] + swap_val
                    child[rand_sys2, rand_Z2, rand_Y2, rand_X2, rand_nl2] = child[rand_sys2, rand_Z2, rand_Y2, rand_X2, rand_nl2] - swap_val
        elif rand_val <= self.mutation_func1 + self.mutation_func2 +self.mutation_func3:
            # 3rd mutation
            for _ in range(int(n_exchange)):
                # do the mutation!
                rand_sys1 = np.random.randint(self.n_Systems)            
                rand_X1 = np.random.randint(self.system_dim_X)
                rand_Y1 = np.random.randint(self.system_dim_Y)
                rand_Z1 = np.random.randint(self.system_dim_Z)

                rand_sys2 = np.random.randint(self.n_Systems)
                rand_Y2 = np.random.randint(self.system_dim_Y)
                rand_Z2 = np.random.randint(self.system_dim_Z)
                rand_X2 = np.random.randint(self.system_dim_X)

                # find new node if they are the same one
                while (rand_sys2 == rand_sys1 and rand_Z2 == rand_Z1 and rand_Y2 == rand_Y1 and rand_X2 ==  rand_X1) :
                    rand_sys2 = np.random.randint(self.n_Systems)
                    rand_Y2 = np.random.randint(self.system_dim_Y)
                    rand_Z2 = np.random.randint(self.system_dim_Z)
                    rand_X2 = np.random.randint(self.system_dim_X)
                
                c1_n_neurons = np.sum(child[rand_sys1, rand_Z1, rand_Y1, rand_X1,1:])
                c2_n_neurons = np.sum(child[rand_sys2, rand_Z2, rand_Y2, rand_X2,1:])
                to_be_swapped_c1 = min(c1_n_neurons, c2_n_neurons)
                to_be_swapped_c2 = min(c1_n_neurons, c2_n_neurons)

                for nlx in range(len(self.topo)):
                    swap_val_c1 =  child[rand_sys1, rand_Z1, rand_Y1, rand_X1, nlx]
                    swap_val_c2 =  child[rand_sys2, rand_Z2, rand_Y2, rand_X2, nlx]

                    if swap_val_c1 <= to_be_swapped_c1 and swap_val_c2 <= to_be_swapped_c2:
                        # swap_val unchanged!
                        pass
                        
                    elif swap_val_c1 > to_be_swapped_c1 and swap_val_c2 <= to_be_swapped_c2:
                        swap_val_c1 = to_be_swapped_c1
                    elif swap_val_c1 <= to_be_swapped_c1 and swap_val_c2 > to_be_swapped_c2:
                        swap_val_c2 = to_be_swapped_c2
                    else: # swap_val_c1 > to_be_swapped_c1 and swap_val_c2 > to_be_swapped_c2:
                        swap_val_c1 = to_be_swapped_c1
                        swap_val_c2 = to_be_swapped_c2

                    child[rand_sys1, rand_Z1, rand_Y1, rand_X1, nlx] = child[rand_sys1, rand_Z1, rand_Y1, rand_X1, nlx] - swap_val_c1 + swap_val_c2
                    child[rand_sys2, rand_Z2, rand_Y2, rand_X2, nlx] = child[rand_sys2, rand_Z2, rand_Y2, rand_X2, nlx] - swap_val_c2 + swap_val_c1
                    to_be_swapped_c1 = to_be_swapped_c1 - swap_val_c1
                    to_be_swapped_c2 = to_be_swapped_c2 - swap_val_c2


                    


        return child,
    
    def survivor(self, popList,  popFitnessList, MutpopList, cList, n_best):
        if self.mut_parent:
            cFitList = self.Pop_comcost(cList)
            MutPopFitnessList = self.Pop_comcost(MutpopList)
            newPop = np.zeros((self.pop_size*3, self.n_Systems, self.system_dim_Z, self.system_dim_Y,
                            self.system_dim_X,  len(self.topo)), dtype=int)
            newFitList = np.zeros(len(newPop), dtype = int)

            for i in range(self.pop_size):
                newPop[i] = popList[i]
                newPop[i+(self.pop_size)] = cList[i]
                newPop[i+2*(self.pop_size)] = MutpopList[i]

                newFitList[i] = popFitnessList[i]
                newFitList[i+(self.pop_size)] = cFitList[i]
                newFitList[i+2*(self.pop_size)] = MutPopFitnessList[i]
                


            bestIdx = newFitList.argsort()[:n_best][::-1] ## minimized
            return newPop[bestIdx], newFitList[bestIdx]
        else: 
            cFitList = self.Pop_comcost(cList)
            newPop = np.zeros((self.pop_size*2, self.n_Systems, self.system_dim_Z, self.system_dim_Y,
                            self.system_dim_X,  len(self.topo)), dtype=int)
            newFitList = np.zeros(len(newPop), dtype = int)

            for i in range(self.pop_size):
                newPop[i] = popList[i]
                newPop[i+(self.pop_size)] = cList[i]
                newFitList[i] = popFitnessList[i]
                newFitList[i+(self.pop_size)] = cFitList[i]

            bestIdx = newFitList.argsort()[:n_best][::-1] ## minimized
            return newPop[bestIdx], newFitList[bestIdx]
        
    
    def final_selection(self, popList, popFitnessList):
        bestIdx = np.argmin(popFitnessList)
        return popList[bestIdx], popFitnessList[bestIdx]

    def exec(self):

        # DEAP Init
        creator.create("Fitness", base.Fitness, weights=(-1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.Fitness)

        toolbox = base.Toolbox()

        toolbox.register("attr_item", self.__generate_random_solution)

        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_item)
        toolbox.register("population",  tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.commcost)
        toolbox.register("mate", self.crossover)
        toolbox.register("mutate", self.new_mutation)
        toolbox.register("select", tools.selTournament, tournsize=self.n_members_in_toursel)

        # Init pop
        popList = toolbox.population(n=self.pop_size)
        self.validate_pop(popList)

        # Hall of fame and evolution statistics
        hof = tools.HallOfFame(1, similar=np.array_equal)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        if self.algo_type == "eaSimple":
            popList, log = algorithms.eaSimple(popList, toolbox, cxpb=self.P_C, mutpb=self.P_M, ngen=self.n_generations, stats=stats, halloffame=hof, verbose=True)
        elif self.algo_type == "eaMuPlusLambda":
            popList, log = algorithms.eaMuPlusLambda(popList, toolbox, mu=self.MU, lambda_=self.LAMBDA, cxpb=self.P_C, mutpb=self.P_M, ngen=self.n_generations, stats=stats, halloffame=hof, verbose=True)
        elif self.algo_type == "eaMuCommaLambda":
            popList, log = algorithms.eaMuCommaLambda(popList, toolbox, mu=self.MU, lambda_=self.LAMBDA, cxpb=self.P_C, mutpb=self.P_M, ngen=self.n_generations, stats=stats, halloffame=hof, verbose=True)

        finalSol = hof[0]
        finalFitness = hof[0].fitness.values[0]
        print ("final solution:")
        print(finalSol)
        print ("final cost:")
        print(finalFitness)
     
        return finalSol, finalFitness, log

