from HeterGenMap.Utils import Coordinate
class Task:
    def __init__(self, ID):
        self.ID = ID
        self.incomming_tasks = []
        self.outgoing_tasks = []
        self.mapped_cluster_address = Coordinate(-1, -1, -1)
        self.layer_idx = -1

    def set_layer_idx(self, l):
        self.layer_idx = l
        
    def set_incomming_tasks(self, i):
        self.incomming_tasks = i
    
    def set_outgoing_tasks(self, o):
        self.outgoing_tasks = o

    def set_mapped_cluster_address(self, a):
        self.mapped_cluster_address = a

    def migrate(self, new_address):
        Mcost = abs(self.mapped_cluster_address.X - new_address.X) \
            + abs(self.mapped_cluster_address.Y - new_address.Y) +abs(self.mapped_cluster_address.Z - new_address.Z)
        self.mapped_cluster_address = new_address
        return Mcost
