from HeterGenMap.Utils import Coordinate

class Cluster:
    def __init__(self, address, n_elements):
        
        self.address = address # address of the cluster, format [Z, Y, X]

        self.n_elements = n_elements # orginal number of elements for the cluster
        self.n_faults = 0 # number of faults in the cluster (can be predicted or randomized)
        self.n_heathy_elements = n_elements  # number of healthy and migratable elements

        self.n_assigned_elements = 0 # number of actual assigned tasks
        self.TaskList = [] # List of Tasks were assigned into this Cluster.

        # Due to faults, the number of mapped elements can be lower than number of assigned
        self.n_mapped_elements = 0 # number of assigned but unmapped elements
        self.n_unmapped_elements = 0 # number of assigned but unmapped elements
        self.n_spare_elements = 0 # number of spare after correcting

        self.weight = 0 # weight for cluster, relative to other clusters

        self.connectivity = [1, 1, 1, 1, 1, 1] # distance to the neighbor router: 6 dim



    
    def canSelfRepair(self):
        return (self.n_faults + self.n_assigned_elements <= self.n_elements)

    # Allocating n_Tasks to Elements. See method appendTaskList for assigning actual list
    def set_init_allocation(self, n_assigned):
        # This is under the condition that there are no fault at the beginning
        self.n_assigned_elements = n_assigned
        self.n_mapped_elements = n_assigned
        self.n_spare_elements = self.n_elements - self.n_assigned_elements

    # Setting the number of faults
    def add_n_faults (self, f):
        self.n_faults = self.n_faults + f
        self.n_heathy_elements = self.n_elements - self.n_faults

    def set_n_spare_elements(self, n):
        self.n_spare_elements = n

    def set_n_assigned_elements(self, n):
        self.n_assigned_elements = n

    # Actual mapping to TaskList. see method set_n_assigned_element for allocting number
    def appendTaskList(self, t):
        self.TaskList.append(t)

    def print_info(self):
        print("This is a Cluster object.")
        print("address:" )
        self.address.print()
        print("It has "+ str(self.n_elements) +" elements")
        print("It has "+ str(self.n_assigned_elements) +" assigned tasks")
        print("It has "+ str(self.n_faults) + " faulty elements")
        print("It has "+ str(self.n_spare_elements) + " spare elements ")
        print("It has "+ str(self.n_mapped_elements) + " mapped elements")
        print("It has "+ str(self.n_unmapped_elements) + " unmapped elements")
        print("It has "+ str(self.n_heathy_elements) + " healthy elements")

    def SelfRepair(self):
        if self.canSelfRepair():
            # If there are enough spare, replace faults by spares
            self.n_spare_elements = self.n_heathy_elements - self.n_assigned_elements
            self.n_unmapped_elements = 0
            self.n_mapped_elements = self.n_assigned_elements
        else:
            # If there are not enough spare, start mark unmapped elements
            self.n_unmapped_elements =  self.n_assigned_elements - self.n_heathy_elements
            self.n_spare_elements = 0 
            self.n_mapped_elements = self.n_assigned_elements - self.n_unmapped_elements

