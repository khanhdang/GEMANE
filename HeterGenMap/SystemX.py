from HeterGenMap.Cluster import Cluster
from HeterGenMap.Utils import Coordinate
from HeterGenMap.Utils import Link
from HeterGenMap.ShortestPath import ShortestPath
import random
import math
import copy
import numpy as np


class SystemX:
    def __init__(self, system_dim, n_elements_per_cluster, n_chips = 1):
        self.n_chips = n_chips # currently only support 1 chip
        self.system_dim = system_dim
        self.n_elements_per_cluster = n_elements_per_cluster
        self.interface_address = Coordinate(0, 0, 0) # the default address for IO and host PC
        self.spare_cluster = 0 # spare one cluster by default
        self.nodes = [[[Cluster(Coordinate(z, y, x), self.n_elements_per_cluster) \
            for x in range(self.system_dim.X)] for y in range(self.system_dim.Y)]  \
            for z in range(self.system_dim.Z)]
        self.routing_cost = [[[[[[ math.inf \
            for x1 in range(self.system_dim.X)] for y1 in range(self.system_dim.Y)]  \
            for z1 in range(self.system_dim.Z)] 
            for x2 in range(self.system_dim.X)] for y2 in range(self.system_dim.Y)]  \
            for z2 in range(self.system_dim.Z)]

        self.n_links = 2*(self.system_dim.Z*self.system_dim.X*(self.system_dim.Y-1) + self.system_dim.Z*self.system_dim.Y*(self.system_dim.X-1) + \
            self.system_dim.X*self.system_dim.Y*(self.system_dim.Z-1))
        self.defect_rate = 0.0
        self.n_defect = int(self.n_links*self.defect_rate)

        self.list_of_flinks = self.gen_defected_links()

        # print(self.list_of_flinks)
        #[Link(Coordinate(0,0,0), Coordinate(0,0,1)), Link(Coordinate(1,1,2), Coordinate(1,1,3))]
        # self.list_of_flinks = [Link(Coordinate(0,0,0), Coordinate(0,1,0)), Link(Coordinate(0,0,0), Coordinate(0,0,1))]
        # self.list_of_flinks = [Link(Coordinate(0,1,0), Coordinate(0,0,0)), \
        #                     Link(Coordinate(0,0,1), Coordinate(0,0,0)), \
                            # Link(Coordinate(1,1,0), Coordinate(1,0,0)), \
                            # Link(Coordinate(0,0,0), Coordinate(0,0,1))]
        
        self.slink_cost  = 1
        if (self.system_dim.Z == 2 and self.system_dim.Y == 2  and self.system_dim.X == 4 ):
            self.list_of_slinks = [Link(Coordinate(0,0,1), Coordinate(0,0,2)), \
                                    Link(Coordinate(0,1,1), Coordinate(0,1,2)),\
                                    Link(Coordinate(1,0,1), Coordinate(1,0,2)), \
                                    Link(Coordinate(1,1,1), Coordinate(1,1,2)) ]
        elif (self.system_dim.Z == 1 and self.system_dim.Y == 4  and self.system_dim.X == 4 ):
            self.list_of_slinks = [Link(Coordinate(0,0,1), Coordinate(0,0,2)), \
                                    Link(Coordinate(0,1,1), Coordinate(0,1,2)),\
                                    Link(Coordinate(0,2,1), Coordinate(0,2,2)), \
                                    Link(Coordinate(0,3,1), Coordinate(0,3,2)) ]
        elif (self.system_dim.Z == 4 and self.system_dim.Y == 4  and self.system_dim.X == 4 ):
            self.list_of_slinks = [Link(Coordinate(0,0,1), Coordinate(0,0,2)), \
                                    Link(Coordinate(0,1,1), Coordinate(0,1,2)),\
                                    Link(Coordinate(0,2,1), Coordinate(0,2,2)), \
                                    Link(Coordinate(0,3,1), Coordinate(0,3,2)) , \
                                    Link(Coordinate(1,0,1), Coordinate(1,0,2)), \
                                    Link(Coordinate(1,1,1), Coordinate(1,1,2)),\
                                    Link(Coordinate(1,2,1), Coordinate(1,2,2)), \
                                    Link(Coordinate(1,3,1), Coordinate(1,3,2)), \
                                    Link(Coordinate(2,0,1), Coordinate(2,0,2)), \
                                    Link(Coordinate(2,1,1), Coordinate(2,1,2)),\
                                    Link(Coordinate(2,2,1), Coordinate(2,2,2)), \
                                    Link(Coordinate(2,3,1), Coordinate(2,3,2)), \
                                    Link(Coordinate(3,0,1), Coordinate(3,0,2)), \
                                    Link(Coordinate(3,1,1), Coordinate(3,1,2)),\
                                    Link(Coordinate(3,2,1), Coordinate(3,2,2)), \
                                    Link(Coordinate(3,3,1), Coordinate(3,3,2)),\
                                    Link(Coordinate(0,1,0), Coordinate(0,2,0)), \
                                    Link(Coordinate(0,1,1), Coordinate(0,2,1)),\
                                    Link(Coordinate(0,1,2), Coordinate(0,2,2)), \
                                    Link(Coordinate(0,1,3), Coordinate(0,2,3)) , \
                                    Link(Coordinate(1,1,0), Coordinate(1,2,0)), \
                                    Link(Coordinate(1,1,1), Coordinate(1,2,1)),\
                                    Link(Coordinate(1,1,2), Coordinate(1,2,2)), \
                                    Link(Coordinate(1,1,3), Coordinate(1,2,3)), \
                                    Link(Coordinate(2,1,0), Coordinate(2,2,0)), \
                                    Link(Coordinate(2,1,1), Coordinate(2,2,1)),\
                                    Link(Coordinate(2,1,2), Coordinate(2,2,2)), \
                                    Link(Coordinate(2,1,3), Coordinate(2,2,3)), \
                                    Link(Coordinate(3,1,0), Coordinate(3,2,0)), \
                                    Link(Coordinate(3,1,1), Coordinate(3,2,1)),\
                                    Link(Coordinate(3,1,2), Coordinate(3,2,2)), \
                                    Link(Coordinate(3,1,3), Coordinate(3,2,3))   ]
        elif (self.system_dim.Z == 1 and self.system_dim.Y == 8  and self.system_dim.X == 8 ):
            self.list_of_slinks = [ Link(Coordinate(0,0,3), Coordinate(0,0,4)), \
                                    Link(Coordinate(0,1,3), Coordinate(0,1,4)),\
                                    Link(Coordinate(0,2,3), Coordinate(0,2,4)), \
                                    Link(Coordinate(0,3,3), Coordinate(0,3,4)),  Link(Coordinate(0,4,3), Coordinate(0,4,4)), \
                                    Link(Coordinate(0,5,3), Coordinate(0,5,4)),\
                                    Link(Coordinate(0,6,3), Coordinate(0,6,4)), \
                                    Link(Coordinate(0,7,3), Coordinate(0,7,4)), \
                                    Link(Coordinate(0,3,0), Coordinate(0,4,0)), \
                                    Link(Coordinate(0,3,1), Coordinate(0,4,1)),\
                                    Link(Coordinate(0,3,2), Coordinate(0,4,2)), \
                                    Link(Coordinate(0,3,3), Coordinate(0,4,3)),   Link(Coordinate(0,3,4), Coordinate(0,3,4)), \
                                    Link(Coordinate(0,3,5), Coordinate(0,4,5)),\
                                    Link(Coordinate(0,3,6), Coordinate(0,4,6)), \
                                    Link(Coordinate(0,3,7), Coordinate(0,4,7)),  ]
        elif (self.system_dim.Z == 2 and self.system_dim.Y == 7  and self.system_dim.X == 8 ):
            self.list_of_slinks = [ Link(Coordinate(0,0,3), Coordinate(0,0,4)), \
                                    Link(Coordinate(0,1,3), Coordinate(0,1,4)),\
                                    Link(Coordinate(0,2,3), Coordinate(0,2,4)), \
                                    Link(Coordinate(0,3,3), Coordinate(0,3,4)),  Link(Coordinate(0,4,3), Coordinate(0,4,4)), \
                                    Link(Coordinate(0,5,3), Coordinate(0,5,4)),\
                                    Link(Coordinate(0,6,3), Coordinate(0,6,4)), \
                                    Link(Coordinate(0,3,0), Coordinate(0,4,0)), \
                                    Link(Coordinate(0,3,1), Coordinate(0,4,1)),\
                                    Link(Coordinate(0,3,2), Coordinate(0,4,2)), \
                                    Link(Coordinate(0,3,3), Coordinate(0,4,3)),   Link(Coordinate(0,3,4), Coordinate(0,3,4)), \
                                    Link(Coordinate(0,3,5), Coordinate(0,4,5)),\
                                    Link(Coordinate(0,3,6), Coordinate(0,4,6)), \
                                    Link(Coordinate(0,3,7), Coordinate(0,4,7)), \
                                    Link(Coordinate(1,0,3), Coordinate(1,0,4)), \
                                    Link(Coordinate(1,1,3), Coordinate(1,1,4)),\
                                    Link(Coordinate(1,2,3), Coordinate(1,2,4)), \
                                    Link(Coordinate(1,3,3), Coordinate(1,3,4)),  Link(Coordinate(1,4,3), Coordinate(1,4,4)), \
                                    Link(Coordinate(1,5,3), Coordinate(1,5,4)),\
                                    Link(Coordinate(1,6,3), Coordinate(1,6,4)), \
                                    Link(Coordinate(1,3,0), Coordinate(1,4,0)), \
                                    Link(Coordinate(1,3,1), Coordinate(1,4,1)),\
                                    Link(Coordinate(1,3,2), Coordinate(1,4,2)), \
                                    Link(Coordinate(1,3,3), Coordinate(1,4,3)),   Link(Coordinate(1,3,4), Coordinate(1,3,4)), \
                                    Link(Coordinate(1,3,5), Coordinate(1,4,5)),\
                                    Link(Coordinate(1,3,6), Coordinate(1,4,6)), \
                                    Link(Coordinate(1,3,7), Coordinate(1,4,7)),  ]
        elif (self.system_dim.Z == 1 and self.system_dim.Y == 8  and self.system_dim.X == 14 ):
            self.list_of_slinks = [ Link(Coordinate(0,0,6), Coordinate(0,0,7)), \
                                    Link(Coordinate(0,1,6), Coordinate(0,1,7)),\
                                    Link(Coordinate(0,2,6), Coordinate(0,2,7)), \
                                    Link(Coordinate(0,3,6), Coordinate(0,3,7)),  Link(Coordinate(0,4,6), Coordinate(0,4,7)), \
                                    Link(Coordinate(0,5,6), Coordinate(0,5,7)),\
                                    Link(Coordinate(0,6,6), Coordinate(0,6,7)), \
                                    Link(Coordinate(0,7,6), Coordinate(0,7,7)), \
                                    Link(Coordinate(0,3,0), Coordinate(0,4,0)), \
                                    Link(Coordinate(0,3,1), Coordinate(0,4,1)),\
                                    Link(Coordinate(0,3,2), Coordinate(0,4,2)), \
                                    Link(Coordinate(0,3,3), Coordinate(0,4,3)),   Link(Coordinate(0,3,4), Coordinate(0,4,4)), \
                                    Link(Coordinate(0,3,5), Coordinate(0,4,5)),\
                                    Link(Coordinate(0,3,6), Coordinate(0,4,6)), \
                                    Link(Coordinate(0,3,7), Coordinate(0,4,7)), \
                                    Link(Coordinate(0,3,8), Coordinate(0,4,8)), \
                                    Link(Coordinate(0,3,9), Coordinate(0,4,9)),\
                                    Link(Coordinate(0,3,10), Coordinate(0,4,10)), \
                                    Link(Coordinate(0,3,11), Coordinate(0,4,11)),   Link(Coordinate(0,3,12), Coordinate(0,4,12)), \
                                    Link(Coordinate(0,3,13), Coordinate(0,4,13)),]
        else:
            self.list_of_flinks = [Link(Coordinate(0,0,1), Coordinate(0,0,2))]
    def gen_defected_links(self):
        self.n_defect = int(self.n_links*self.defect_rate)
        flinks = []
        for _ in range(self.n_defect):
            rand_x = random.randint(0, self.system_dim.X-1)
            rand_y = random.randint(0, self.system_dim.Y-1)
            rand_z = random.randint(0, self.system_dim.Z-1)
            if self.system_dim.Z == 1:
                rand_dir = random.randint(0, 3) # 0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z: 5:-Z
            else:
                rand_dir = random.randint(0, 5) # 0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z: 5:-Z
            if rand_dir == 0:
                if rand_x == self.system_dim.X-1:
                    flinks.append(Link(Coordinate(rand_z,rand_y,rand_x), Coordinate(rand_z,rand_y,rand_x-1)))
                else:
                    flinks.append(Link(Coordinate(rand_z,rand_y,rand_x), Coordinate(rand_z,rand_y,rand_x+1)))
            elif rand_dir == 1:
                if rand_x == 0:
                    flinks.append(Link(Coordinate(rand_z,rand_y,rand_x), Coordinate(rand_z,rand_y,rand_x+1)))
                else:
                    flinks.append(Link(Coordinate(rand_z,rand_y,rand_x), Coordinate(rand_z,rand_y,rand_x-1)))
            elif rand_dir == 2:
                if rand_y == self.system_dim.Y-1:
                    flinks.append(Link(Coordinate(rand_z,rand_y,rand_x), Coordinate(rand_z,rand_y-1,rand_x)))
                else:
                    flinks.append(Link(Coordinate(rand_z,rand_y,rand_x), Coordinate(rand_z,rand_y+1,rand_x)))
            elif rand_dir == 3:
                if rand_y == 0:
                    flinks.append(Link(Coordinate(rand_z,rand_y,rand_x), Coordinate(rand_z,rand_y+1,rand_x)))
                else:
                    flinks.append(Link(Coordinate(rand_z,rand_y,rand_x), Coordinate(rand_z,rand_y-1,rand_x)))
            elif rand_dir == 4:
                if rand_z == self.system_dim.Z-1:
                    flinks.append(Link(Coordinate(rand_z,rand_y,rand_x), Coordinate(rand_z-1,rand_y,rand_x)))
                else:
                    flinks.append(Link(Coordinate(rand_z,rand_y,rand_x), Coordinate(rand_z+1,rand_y,rand_x)))
            elif rand_dir == 5:
                if rand_z == 0:
                    flinks.append(Link(Coordinate(rand_z,rand_y,rand_x), Coordinate(rand_z+1,rand_y,rand_x)))
                else:
                    flinks.append(Link(Coordinate(rand_z,rand_y,rand_x), Coordinate(rand_z-1,rand_y,rand_x)))
        return flinks
    def naive_assigment(self, n_tasks):
        # Spare one cluster
        n_clusters = self.system_dim.Z*self.system_dim.Y*self.system_dim.X - self.spare_cluster
        average_task_per_cluster = math.ceil(n_tasks/n_clusters)
        # print(average_task_per_cluster)
        n_remained_tasks = n_tasks
        for z in range(self.system_dim.Z):
            for y in range(self.system_dim.Y):
                for x in range(self.system_dim.X):
                    if (n_remained_tasks >= average_task_per_cluster):
                        self.nodes[z][y][x].set_init_allocation(average_task_per_cluster)
                        n_remained_tasks = n_remained_tasks - average_task_per_cluster
                    else:
                        self.nodes[z][y][x].set_init_allocation(n_remained_tasks)
                        n_remained_tasks = 0
    def migrating_tasks(self, flow_graph):
        for i in range(1, len(flow_graph)-1):
            # subtract 1 for vsource
            # last row: vsink
            z = math.floor( (i-1) /(self.system_dim.X * self.system_dim.Y))
            y = math.floor(((i-1)-(z*self.system_dim.X * self.system_dim.Y))/self.system_dim.X)
            x = (i-1)-(z*self.system_dim.X * self.system_dim.Y) - y * self.system_dim.X
            self.nodes[z][y][x].set_n_spare_elements(self.nodes[z][y][x].n_spare_elements - flow_graph[i][len(flow_graph[i])-1])
            self.nodes[z][y][x].set_n_assigned_elements(self.nodes[z][y][x].n_assigned_elements - sum(flow_graph[i][0:len(flow_graph[i])-1]))
            self.nodes[z][y][x].n_unmapped_elements = self.nodes[z][y][x].n_unmapped_elements + flow_graph[i][0]
            if (self.nodes[z][y][x].n_unmapped_elements > 0):
                print("Unmapped: "+str(self.nodes[z][y][x].n_unmapped_elements) + " at ("+str(z) + "," + str(y) + "," + str(x) +")" )

    def insertFault(self, address, n_faults):
        self.nodes[address.Z][address.Y][address.X].add_n_faults(n_faults)
        
    def SelfRepair(self):
        for z in range(self.system_dim.Z):
            for y in range(self.system_dim.Y):
                for x in range(self.system_dim.X):
                    self.nodes[z][y][x].SelfRepair()

    def get_info (self):
        for z in range(self.system_dim.Z):
            for y in range(self.system_dim.Y):
                for x in range(self.system_dim.X):
                    self.nodes[z][y][x].print_info()
    
    
    def genFGraph(self):
        # Two extra vertex: virtual source, virtual sink

        n_vertex = 2 + self.system_dim.Z*self.system_dim.Y*self.system_dim.X
        vsource_ID = 0
        vsink_ID = n_vertex-1
        
        # Intialize with all zeros
        graph = [[0 for r in range (n_vertex)] for c in range (n_vertex)]

        # assign source = number of unmapped
        for z in range(self.system_dim.Z):
            for y in range(self.system_dim.Y):
                for x in range(self.system_dim.X):
                    # col+1: vsource
                    col = 1 + x + y*self.system_dim.X + z*self.system_dim.Y * self.system_dim.X
                    graph[vsource_ID][col] = self.nodes[z][y][x].n_unmapped_elements

                    # vsink, row+1
                    row = 1 + x + y*self.system_dim.X + z*self.system_dim.Y * self.system_dim.X
                    graph[row][vsink_ID] = self.nodes[z][y][x].n_spare_elements
                    
                    if self.nodes[z][y][x].address.hasDown():
                        col = 1 + x + y*self.system_dim.X + (z-1)*self.system_dim.Y * self.system_dim.X
                        graph[row][col] = self.nodes[z-1][y][x].n_heathy_elements
                    if self.nodes[z][y][x].address.hasUp(self.system_dim.Z):
                        col = 1 + x + y*self.system_dim.X + (z+1)*self.system_dim.Y * self.system_dim.X
                        graph[row][col] = self.nodes[z+1][y][x].n_heathy_elements
                    if self.nodes[z][y][x].address.hasNorth():
                        col = 1 + x + (y-1)*self.system_dim.X + (z)*self.system_dim.Y * self.system_dim.X
                        graph[row][col] = self.nodes[z][y-1][x].n_heathy_elements
                    if self.nodes[z][y][x].address.hasSouth(self.system_dim.Y):
                        col = 1 + x + (y+1)*self.system_dim.X + (z)*self.system_dim.Y * self.system_dim.X
                        graph[row][col] = self.nodes[z][y+1][x].n_heathy_elements
                    if self.nodes[z][y][x].address.hasWest():
                        col = 1 + (x-1) + (y)*self.system_dim.X + (z)*self.system_dim.Y * self.system_dim.X
                        graph[row][col] = self.nodes[z][y][x-1].n_heathy_elements
                    if self.nodes[z][y][x].address.hasEast(self.system_dim.X):
                        
                        col = 1 + (x+1) + (y)*self.system_dim.X + (z)*self.system_dim.Y * self.system_dim.X
                        graph[row][col] = self.nodes[z][y][x+1].n_heathy_elements
        

        return graph, vsource_ID, vsink_ID

    def  replicate_init_alloc(self, sysX):

        for z in range(self.system_dim.Z):
            for y in range(self.system_dim.Y):
                for x in range(self.system_dim.X):
                    sysX.nodes[z][y][x].set_init_allocation(self.nodes[z][y][x].n_assigned_elements)
    
    def conv_val_to_bin(self, val):
        if val == 0:
            return "000"
        elif val == 1:
            return "001"
        elif val == 2:
            return "010"
        elif val == 3:
            return "011"
        elif val == 4:
            return "100"
        elif val == 5:
            return "101"
        elif val == 6:
            return "110"
        elif val == 7:
            return "111"

    def conv_add_to_bin(self, address):
        # input: Coordinate(Z, Y, X)
        
        return self.conv_val_to_bin(address.Z)+self.conv_val_to_bin(address.Y)+self.conv_val_to_bin(address.X)


    def gen_RTL_simul_conf(self, TG, input_folder, output_folder):
        # gen the number of inputs from the all csv files
        # Note: first layer is the input layers
        my_input = np.genfromtxt(input_folder+'/spike_layer_0.csv', delimiter=',')
        neuron_idx = 0
        n_timesteps = 0 # init
        neuron_input = my_input[neuron_idx]
        # Note: TG.topo doesn't consist of the n_inputs; TG.n_inputs is the number of inputs
        n_inputs = np.zeros((len(TG.topo)+1, len(neuron_input)), dtype=int)
        # inputs for the input layers are always zeros
        for layer_idx in range(len(TG.topo)):
            # print(layer_idx)
            my_input = np.genfromtxt(input_folder+'/spike_layer_'+str(layer_idx)+'.csv', delimiter=',')
            n_timesteps = len(neuron_input)
            for neuron_input in (my_input):
                for ts in range (len(neuron_input)):
                    n_inputs[layer_idx+1][ts] +=  neuron_input[ts] 
                    
        file0 = open(output_folder+"/_sys.conf","w")
        line0 = "SYSTEM_CONF = "
        line0 += str(len(neuron_input))
        line0 += " "
        line0 += str(self.system_dim.Z)
        line0 += " "
        line0 += str(self.system_dim.Y)
        line0 += " "
        line0 += str(self.system_dim.X)
        file0.write(line0)
        file0.close()
        # generate example one
        # format: <time-step>,<number of input>,<number of output spikes>,<address of each spikes>
        # Note: input layers are not mapped, they are considered to be sent from the node (0,0) or (0,0,0)
        # for l in range (len(TG.topo)):
        #     layer_idx = l+1
        #     neuron_idx = 10
        #     for neuron_idx in range(TG.topo[l]):
        #         file1 = open(output_folder+"/"+str(layer_idx)+"--"+str(neuron_idx)+".txt","w")
        #         line1 = ""

        #         my_data = np.genfromtxt(input_folder+'/spike_layer_'+str(layer_idx)+'.csv', delimiter=',')
        #         neuron_data = my_data[neuron_idx]
        #         for ts in range (len(neuron_data)):
        #             line1 = ""
        #             # timestep index
        #             # print(str(ts), end = ",")
        #             # print(str(n_inputs[layer_idx][ts]), end=",")
        #             line1 += (str(ts)+",")
        #             line1 += (str(n_inputs[layer_idx][ts])+",")
        #             address_list = []
        #             # == 1: has spike, == 0: no spike
        #             if neuron_data[ts] == 1:
        #                 for outid in TG.TaskList[neuron_idx].outgoing_tasks:
        #                     addr = self.conv_add_to_bin(TG.TaskList[outid].mapped_cluster_address)
        #                     if addr not in address_list:
        #                         address_list.append(addr)
                        
        #                 # number of outputs
        #                 # print(len(address_list), end = ",")
        #                 line1 += (str(len(address_list)) + ",")
        #                 for addr_idx in range(len(address_list)):
        #                     if addr_idx < len(address_list)-1:
        #                         # print(address_list[addr_idx], end=",")
        #                         line1 += (address_list[addr_idx]+",")
        #                     else:
        #                         # print(address_list[addr_idx])
        #                         line1 += (address_list[addr_idx])
        #             else:
        #                 # print("0")
        #                 line1 += ("0")
        #             file1.write(line1+"\n")
        #         file1.close()

        # print("Task allocations")
        for z in range(self.system_dim.Z):
            for y in range(self.system_dim.Y):
                for x in range(self.system_dim.X):
                    core_idx = z*self.system_dim.Y * self.system_dim.X + y * self.system_dim.X +x
                    file2 = open(output_folder+"/"+str(core_idx).zfill(4)+".conf","w")
                    print("z="+str(z)+", y="+str(y)+", x="+str(x), end= ":")
                    # print("[", end= "")
                    for t in self.nodes[z][y][x].TaskList:
                        # print(str(t.ID), end= ": layer=")
                        # print(str(t.layer_idx), end= ", neuron=")

                        layer_idx = t.layer_idx
                        neuron_idx = t.ID - sum(TG.topo[0:layer_idx])
                        # print(neuron_idx, end= ", ")
                        


                    # print("]")

                    for ts in range(n_timesteps):
                        print("\nts = "+str(ts))
                        file2.write(str(ts)+",")
                        input_rev = []
                        task_address_list = []
                        n_inputs_to_core =  0
                        n_outputs_from_core =  0
                        line2 = ""

                        print("load input layers")

                        # input layer are attached to the interface address
                        if (z == self.interface_address.Z and y == self.interface_address.Y and x == self.interface_address.X):
                            layer_data = np.genfromtxt(input_folder+'/spike_layer_0.csv', delimiter=',')
                            for i in layer_data:
                                if i[ts] == 1:
                                    task_address_list = []
                                    # we need to find the destination!
                                    for z1 in range(self.system_dim.Z):
                                        for y1 in range(self.system_dim.Y):
                                            for x1 in range(self.system_dim.X):
                                                for t in self.nodes[z1][y1][x1].TaskList:
                                                    if t.layer_idx == 0 :
                                                        addr = self.conv_add_to_bin(t.mapped_cluster_address)
                                                        if addr not in task_address_list:
                                                            task_address_list.append(addr)
                                    
                                                        
                                    for addr_idx in range(len(task_address_list)):
                                        line2 += (task_address_list[addr_idx]+",")
                                                                
                                    n_outputs_from_core += len(task_address_list)

                        # other layers
                        
                        print("others layers")

                        for t in self.nodes[z][y][x].TaskList:
                            task_address_list = []
                            layer_idx = t.layer_idx
                            neuron_idx = t.ID - sum(TG.topo[0:layer_idx])
                            
                            layer_data = np.genfromtxt(input_folder+'/spike_layer_'+str(layer_idx+1)+'.csv', delimiter=',')
                            neuron_data = layer_data[neuron_idx]
                            
                            # inputs
                            if layer_idx not in input_rev:
                                input_rev.append(layer_idx)
                                n_inputs_to_core += n_inputs[layer_idx+1][ts]
                            
                            # outputs
                            if neuron_data[ts] == 1:
                                for outid in t.outgoing_tasks:
                                    addr = self.conv_add_to_bin(TG.TaskList[outid].mapped_cluster_address)
                                    if addr not in task_address_list:
                                        task_address_list.append(addr)
                                
                                # number of outputs
                                n_outputs_from_core += len(task_address_list)
                                for addr_idx in range(len(task_address_list)):
                                    line2 += (task_address_list[addr_idx]+",")
                            

                        file2.write(str(n_inputs_to_core)+",")
                        file2.write(str(n_outputs_from_core))
                        if len(line2) > 0:
                            file2.write(",")
                            line2 = line2[0:-1]  # remove ","
                            file2.write(line2)
                        # else:
                        #     file2.write("0")


                        file2.write("\n")


                    file2.close()

                print("\n")
                        
        return 0
    
    def gen_routing_cost(self):
        # routing cost without faults >> this function is deprecated
        for x1 in range(self.system_dim.X):
            for y1 in range(self.system_dim.Y):
                for z1 in range(self.system_dim.Z): 
                    for x2 in range(self.system_dim.X):
                        for y2 in range(self.system_dim.Y):
                            for z2 in range(self.system_dim.Z):
                                self.routing_cost[z2][y2][x2][z1][y1][x1] = abs(z1-z2) + abs(y1-y2) + abs(x1-x2)
        return 0

    def gen_heterogeneous_routing_cost(self):
        
        n_clusters = self.system_dim.Z*self.system_dim.Y*self.system_dim.X 
        SP = ShortestPath(n_clusters)

        # routing cost without diversity
        for z1 in range(self.system_dim.Z):
            for y1 in range(self.system_dim.Y):
                for x1 in range(self.system_dim.X): 
                    for z2 in range(self.system_dim.Z):
                        for y2 in range(self.system_dim.Y):
                            for x2 in range(self.system_dim.X):
                                add1 = x1 + y1*self.system_dim.X + z1*self.system_dim.Y*self.system_dim.X
                                add2 = x2 + y2*self.system_dim.X + z2*self.system_dim.Y*self.system_dim.X
                                if x1==x2 and y1==y2 and abs(z1-z2) == 1:
                                    SP.graph[add1][add2] = 1
                                if z1==z2 and y1==y2 and abs(x1-x2) == 1:
                                    SP.graph[add1][add2] = 1
                                if z1==z2 and x1==x2 and abs(y1-y2) == 1:
                                    SP.graph[add1][add2] = 1
        # here, i support list of flinks.
        # in the case all routers are defectives, we must disable all of its links
        for l in self.list_of_slinks:
            add1 = l.start.X + l.start.Y*self.system_dim.X + l.start.Z*self.system_dim.Y*self.system_dim.X
            add2 = l.end.X + l.end.Y*self.system_dim.X + l.end.Z*self.system_dim.Y*self.system_dim.X
            SP.graph[add1][add2] = self.slink_cost

        # vary the links with different costs
        for l in self.list_of_flinks:
            add1 = l.start.X + l.start.Y*self.system_dim.X + l.start.Z*self.system_dim.Y*self.system_dim.X
            add2 = l.end.X + l.end.Y*self.system_dim.X + l.end.Z*self.system_dim.Y*self.system_dim.X
            SP.graph[add1][add2] = 0
        for z1 in range(self.system_dim.Z):
            for y1 in range(self.system_dim.Y):
                for x1 in range(self.system_dim.X): 
                    add1 = x1 + y1*self.system_dim.X + z1*self.system_dim.Y*self.system_dim.X
                    dist = SP.find_path(add1)
                    # print(dist)                    
                    for z2 in range(self.system_dim.Z):
                        for y2 in range(self.system_dim.Y):
                            for x2 in range(self.system_dim.X):
                                add2 = x2 + y2*self.system_dim.X + z2*self.system_dim.Y*self.system_dim.X
                                
                                self.routing_cost[z1][y1][x1][z2][y2][x2] = dist[add2]
        return 0