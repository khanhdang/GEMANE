from HeterGenMap.Task import Task
from HeterGenMap.SystemX import SystemX
from HeterGenMap.Utils import Coordinate, distance
from HeterGenMap.GA_DEAP import GA_DEAP
import math
import numpy as np

class TaskGraph:
    def __init__(self, type, n_tasks, topo, n_inputs):
        self.type = type
        self.n_tasks = n_tasks
        self.topo = topo
        self.n_inputs = n_inputs
        self.TaskList = []
        for idx in range(self.n_tasks):
            self.TaskList.append(Task(idx))
        self.__assign_task_flow()

    # private method to assign the task flow
    # This method should be called once
    def __assign_task_flow(self):
        if self.type == "MLP" or self.type == "CNN":
            ## Generate layers by indexes
            layer_idx = 0
            inlayer_idx = 0
            done = True
            TaskIdx = 0
            tasks_by_layer = []
            tasks_within_layer = []
            while(done):
                tasks_within_layer.append(TaskIdx)

                self.TaskList[TaskIdx].set_layer_idx(layer_idx) 
                
                if (inlayer_idx == self.topo[layer_idx]-1):
                    tasks_by_layer.append(tasks_within_layer)
                    tasks_within_layer = []

                    if (layer_idx == len(self.topo) - 1):
                        done = False # Finish scanning
                    else: # move to next layer
                        inlayer_idx = 0
                        layer_idx = layer_idx + 1
                else:
                    inlayer_idx = inlayer_idx + 1
                
                TaskIdx = TaskIdx + 1

            # print out the tasks
            TaskIdx = 0
            for l in range(len(self.topo)):
                
                if l > 0:
                    for i in range(len(tasks_by_layer[l])):
                        self.TaskList[TaskIdx].set_incomming_tasks(tasks_by_layer[l-1]) 
                        TaskIdx = TaskIdx + 1
                    TaskIdx = TaskIdx - len(tasks_by_layer[l]) # return to the original address
                
                if l < len(self.topo) - 1:
                    for i in range(len(tasks_by_layer[l])):
                        self.TaskList[TaskIdx].set_outgoing_tasks(tasks_by_layer[l+1]) 
                        TaskIdx = TaskIdx + 1
            
            return 1
        else:
            print("This type of "+self.type+" is not support.")
            return 0
    
    def assign_address(self, mapping_alg, sysX, i_algo_type="eaMuPlusLambda", i_pop_size=100, i_n_generations=80, i_n_members_in_toursel=5, n_init_mem=2):

        
        if (mapping_alg == "linear" or mapping_alg == "linearXYZ"):
            # print("The tasks are mapped linearly.")
            # print(system_dim)
            TaskIdx = 0
            for z in range(sysX.system_dim.Z):
                for y in range(sysX.system_dim.Y):
                    for x in range(sysX.system_dim.X):
                        for e in range(sysX.nodes[z][y][x].n_mapped_elements):
                            self.TaskList[TaskIdx].set_mapped_cluster_address(Coordinate(z, y, x))
                            sysX.nodes[z][y][x].appendTaskList(self.TaskList[TaskIdx])
                            TaskIdx = TaskIdx + 1

            return 1
        
        

        elif (mapping_alg == "GA_DEAP"):
            print("We are running single objective optimization for initialization mapping")
            # Input: sysX : system with information (i.e. n_mapped_elements of each nodes)
            # Input: TaskList: list of task and connection. Each task has connections (incomming and outgoing tasks' IDs)
            
            # Evaluation: using CommCostMLP()
            
            print(">> Create GA")
            # algo_type: eaSimple, eaMuPlusLambda, eaMuCommaLambda
            algo_type = "eaMuPlusLambda"
            GA = GA_DEAP(algo_type=i_algo_type, pop_size=i_pop_size, n_generations=i_n_generations, n_members_in_toursel=i_n_members_in_toursel, n_init_mem=n_init_mem, MU=50, LAMBDA=100, topo=self.topo, n_inputs=self.n_inputs, sysX=sysX)
            print(">> Execute GA")
            print(">> Algorithm: " + algo_type)
            finalSol, finalFitness, log = GA.exec()
            print(">> Finish GA")
            TaskCnt = np.zeros(len(self.topo), dtype=int)
            for z in range(sysX.system_dim.Z):
                for y in range(sysX.system_dim.Y):
                    for x in range(sysX.system_dim.X):
                        for nl in range(len(self.topo)):
                            for i in range(finalSol[0, z, y, x, nl+1]):
                                TaskIdx = int(sum(self.topo[:nl])+ TaskCnt[nl])
                                self.TaskList[TaskIdx].set_mapped_cluster_address(Coordinate(z, y, x))
                                sysX.nodes[z][y][x].appendTaskList(self.TaskList[TaskIdx])
                                TaskCnt[nl] = TaskCnt[nl]+1

            return finalSol, finalFitness, log

        else: 
            print("The program currently does support " + str(mapping_alg))
    
    def migrating_tasks(self, flow_graph, sysX):
        MCost = 0
        if self.type == "MLP" or self.type == "CNN":
            for i in range(len(flow_graph)):
                for j in range(len(flow_graph[i])):
                    if flow_graph[i][j] > 0 and i > 0:
                        sz = math.floor( (i-1) /(sysX.system_dim.X * sysX.system_dim.Y))
                        sy = math.floor(((i-1)-(sz*sysX.system_dim.X * sysX.system_dim.Y))/sysX.system_dim.X)
                        sx = (i-1)-(sz*sysX.system_dim.X * sysX.system_dim.Y) - sy * sysX.system_dim.X
                        dz = math.floor( (j-1) /(sysX.system_dim.X * sysX.system_dim.Y))
                        dy = math.floor(((j-1)-(dz*sysX.system_dim.X * sysX.system_dim.Y))/sysX.system_dim.X)
                        dx = (j-1)-(dz*sysX.system_dim.X * sysX.system_dim.Y) - dy * sysX.system_dim.X
                        
                        if j == len(flow_graph[i])-1:
                            print("Migrating use spare for "+str(flow_graph[i][j])+" tasks in (" +str(sz) + ", " +str(sy)+ ", "+ str(sx)+")") # no particular changes
                        else:
                            
                            print("Migrating from (" +str(sz) + ", " +str(sy)+ ", "+ str(sx)+") to (" +str(dz) + ", " +str(dy)+ ", "+ str(dx)+") : "+str(flow_graph[i][j])+ " tasks")


                            for k in range(flow_graph[i][j]):

                                migrated_tasks = sysX.nodes[sz][sy][sx].TaskList.pop(-1)
                                m = migrated_tasks.migrate(Coordinate(dz, dy, dx))
                                MCost = MCost + m
                                sysX.nodes[dz][dy][dx].TaskList.append(migrated_tasks)
        else:
            print("this topo is unsupported")
        return MCost

    def CommCost(self, sysX):
        Cost = 0
        for TaskIdx in range(self.n_tasks):

            if (len(self.TaskList[TaskIdx].incomming_tasks) == 0): #empty input (receive from sysX.interface_address)
                Cost = Cost + distance(sysX.interface_address, self.TaskList[TaskIdx].mapped_cluster_address)
            else:
                for inTaskIdx in self.TaskList[TaskIdx].incomming_tasks:
                    Cost = Cost + distance(self.TaskList[inTaskIdx].mapped_cluster_address, self.TaskList[TaskIdx].mapped_cluster_address)
            if (len(self.TaskList[TaskIdx].outgoing_tasks) == 0): #empty output (send to sysX.interface_address)
                Cost = Cost + distance(sysX.interface_address, self.TaskList[TaskIdx].mapped_cluster_address)
            
        return Cost
    
    def CommCostMLP(self, sysX):
        Cost = 0
        Distance_Dist = np.zeros((2*(sysX.system_dim.Z+sysX.system_dim.Y+sysX.system_dim.X)))

        Visisted_Address = []
        Visisted_Accessing_IO_Address = []
        for TaskIdx in range(self.n_tasks):
            Visisted_Address = []
            if (len(self.TaskList[TaskIdx].outgoing_tasks) == 0): #empty output (receive from sysX.interface_address)
                dist = distance(sysX.interface_address, self.TaskList[TaskIdx].mapped_cluster_address)
                Cost = Cost + dist
                Distance_Dist[dist] += 1
                
                
            else:
                for outTaskIdx in self.TaskList[TaskIdx].outgoing_tasks:
                    if any((obj.X == self.TaskList[outTaskIdx].mapped_cluster_address.X and obj.Y == self.TaskList[outTaskIdx].mapped_cluster_address.Y and obj.Z == self.TaskList[outTaskIdx].mapped_cluster_address.Z) for obj in Visisted_Address):
                        pass
                    
                    else:
                        dist = distance(self.TaskList[outTaskIdx].mapped_cluster_address, self.TaskList[TaskIdx].mapped_cluster_address)
                        Cost = Cost + dist
                        Distance_Dist[dist] += 1
                        Visisted_Address.append(self.TaskList[outTaskIdx].mapped_cluster_address)

            if (len(self.TaskList[TaskIdx].incomming_tasks) == 0): #empty input (send to sysX.interface_address)
                if any((obj.X == self.TaskList[TaskIdx].mapped_cluster_address.X and obj.Y == self.TaskList[TaskIdx].mapped_cluster_address.Y and obj.Z == self.TaskList[TaskIdx].mapped_cluster_address.Z) for obj in Visisted_Accessing_IO_Address):
                    pass
                else:
                    dist = distance(sysX.interface_address, self.TaskList[TaskIdx].mapped_cluster_address)
                    Cost = Cost + dist
                    Distance_Dist[dist] += 1
                    Visisted_Accessing_IO_Address.append(self.TaskList[TaskIdx].mapped_cluster_address)
            
        return Cost, Distance_Dist
    def FTCommCostMLP(self, sysX):
        Cost = 0
        Visisted_Address = []
        Visisted_Accessing_IO_Address = []
        Distance_Dist = np.zeros((4*(sysX.system_dim.Z+sysX.system_dim.Y+sysX.system_dim.X+10)))

        for TaskIdx in range(self.n_tasks):
            Visisted_Address = []
            if (len(self.TaskList[TaskIdx].outgoing_tasks) == 0): #empty output (receive from sysX.interface_address)
                dist = sysX.routing_cost[self.TaskList[TaskIdx].mapped_cluster_address.Z][self.TaskList[TaskIdx].mapped_cluster_address.Y][self.TaskList[TaskIdx].mapped_cluster_address.X][sysX.interface_address.Z][sysX.interface_address.Y][sysX.interface_address.X]
                Cost = Cost + dist
                Distance_Dist[dist] +=  1
                
            else:
                for outTaskIdx in self.TaskList[TaskIdx].outgoing_tasks:
                    if any((obj.X == self.TaskList[outTaskIdx].mapped_cluster_address.X and obj.Y == self.TaskList[outTaskIdx].mapped_cluster_address.Y and obj.Z == self.TaskList[outTaskIdx].mapped_cluster_address.Z) for obj in Visisted_Address):
                        pass
                    
                    else:
                        
                        dist = sysX.routing_cost[self.TaskList[TaskIdx].mapped_cluster_address.Z][self.TaskList[TaskIdx].mapped_cluster_address.Y][self.TaskList[TaskIdx].mapped_cluster_address.X][self.TaskList[outTaskIdx].mapped_cluster_address.Z][self.TaskList[outTaskIdx].mapped_cluster_address.Y][self.TaskList[outTaskIdx].mapped_cluster_address.X]
                        Cost = Cost + dist
                        Distance_Dist[dist] +=  1
                        Visisted_Address.append(self.TaskList[outTaskIdx].mapped_cluster_address)

            if (len(self.TaskList[TaskIdx].incomming_tasks) == 0): #empty input (send to sysX.interface_address)
                if any((obj.X == self.TaskList[TaskIdx].mapped_cluster_address.X and obj.Y == self.TaskList[TaskIdx].mapped_cluster_address.Y and obj.Z == self.TaskList[TaskIdx].mapped_cluster_address.Z) for obj in Visisted_Accessing_IO_Address):
                    pass
                else:
                    dist =  sysX.routing_cost[sysX.interface_address.Z][sysX.interface_address.Y][sysX.interface_address.X][self.TaskList[TaskIdx].mapped_cluster_address.Z][self.TaskList[TaskIdx].mapped_cluster_address.Y][self.TaskList[TaskIdx].mapped_cluster_address.X]
                    Cost = Cost + dist
                    Distance_Dist[dist] +=  1
                    Visisted_Accessing_IO_Address.append(self.TaskList[TaskIdx].mapped_cluster_address)
            
        return Cost, Distance_Dist
    
    def CommCostMLPMax(self, sysX):
        Max = 0
        Cost = 0
        Visisted_Address = []
        Visisted_Accessing_IO_Address = []
        for TaskIdx in range(self.n_tasks):
            Cost = 0
            Visisted_Address = []
            # I only count the communication cost for incomming_tasks
            # If we count for both incomming and outgoing, we will double the cost
            if (len(self.TaskList[TaskIdx].outgoing_tasks) == 0): #empty output (receive from sysX.interface_address)
                # I Skip this because output spike has no  criticality
                pass
                Cost = Cost + distance(sysX.interface_address, self.TaskList[TaskIdx].mapped_cluster_address)
            else:
                for outTaskIdx in self.TaskList[TaskIdx].outgoing_tasks:
                    if any((obj.X == self.TaskList[outTaskIdx].mapped_cluster_address.X and obj.Y == self.TaskList[outTaskIdx].mapped_cluster_address.Y and obj.Z == self.TaskList[outTaskIdx].mapped_cluster_address.Z) for obj in Visisted_Address):
                        pass
                    else:
                        Cost = Cost + distance(self.TaskList[outTaskIdx].mapped_cluster_address, self.TaskList[TaskIdx].mapped_cluster_address)
                        Visisted_Address.append(self.TaskList[outTaskIdx].mapped_cluster_address)



            if Cost > Max:
                Max = Cost
        return Max
    
