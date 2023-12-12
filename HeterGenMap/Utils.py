class Coordinate:
    def __init__(self, Z, Y, X):
        self.X = X
        self.Y = Y
        self.Z = Z

    def print(self):
        print("("+str(self.Z)+", "+str(self.Y)+", "+str(self.X)+")")

    def setValue (self, Z, Y, X):

        self.X = X
        self.Y = Y
        self.Z = Z
    
    def getValue (self):
        return self
    
    def getNorth (self):
        return self.Z, self.Y-1, self.X
    def hasNorth (self):
        return self.Y > 0

    def getSouth (self):
        return self.Z, self.Y+1, self.X
    def hasSouth (self, maxY):
        return self.Y < (maxY-1)
    
    def getWest (self):
        return self.Z, self.Y, self.X-1
    def hasWest (self):
        return self.X > 0
    
    def getEast (self):
        return self.Z, self.Y, self.X+1
    def hasEast (self, maxX):
        # print(self.X)
        return self.X < (maxX-1)

    
    def getUp (self):
        return self.Z+1, self.Y, self.X
    def hasUp (self, maxZ):
        # print("hasUp")
        # print(self.Z)
        # print(maxZ)
        return self.Z < (maxZ-1)
        
    def getDown (self):
        return self.Z-1, self.Y, self.X
    def hasDown (self):
        return self.Z > 0


    def getNeighbors(self, dim, Neighbors):
        if self.hasDown():
            Neighbors.append((self.getDown))
        if self.hasUp(dim.Z):
            Neighbors.append((self.getUp))
        if self.hasNorth():
            Neighbors.append((self.getNorth))
        if self.hasSouth(dim.Y):
            Neighbors.append((self.getSouth))
        if self.hasWest():
            Neighbors.append((self.getWest))
        if self.hasEast(dim.X):
            Neighbors.append((self.getEast))
        
        

def distance(c1, c2):
    return abs(c1.X-c2.X)+abs(c1.Y - c2.Y) + abs(c1.Z - c2.Z)


def route_distance(c1, c2, network_status):
    return abs(c1.X-c2.X)+abs(c1.Y - c2.Y) + abs(c1.Z - c2.Z)

def CommCostforSeparatedSolution(TaskListIn,  sysX):
        Cost = 0
        for TaskIdx in range(len(TaskListIn)):

            if (len(TaskListIn[TaskIdx].incomming_tasks) == 0): #empty input (receive from sysX.interface_address)
                Cost = Cost + distance(sysX.interface_address, TaskListIn[TaskIdx].mapped_cluster_address)
            else:
                for inTaskIdx in TaskListIn[TaskIdx].incomming_tasks:
                    Cost = Cost + distance(TaskListIn[inTaskIdx].mapped_cluster_address, TaskListIn[TaskIdx].mapped_cluster_address)
            if (len(TaskListIn[TaskIdx].outgoing_tasks) == 0): #empty output (send to sysX.interface_address)
                Cost = Cost + distance(sysX.interface_address, TaskListIn[TaskIdx].mapped_cluster_address)
            
        return Cost


class Link:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def print(self):
        print("Link:(")
        (self.start.print())
        (self.end.print())
        print(')')
    def match(self, s, e):
        if self.start.Z == s.Z and self.start.Y == s.Y \
            and self.start.X == s.X and self.end.Z == e.Z \
            and self.end.Y == e.Y and self.end.X == e.X:
            return True
        else:
            return False