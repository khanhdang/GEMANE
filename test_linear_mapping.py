from HeterGenMap.TaskGraph import TaskGraph
from HeterGenMap.Task import Task
from HeterGenMap.Utils import Coordinate
from HeterGenMap.SystemX import SystemX

TG = TaskGraph("MLP", 23, [10, 10, 3], 10)
s = SystemX(Coordinate(2, 2, 2), 5) # 2x2x2 5 elements/ cluster

print("========================")
print(">> incomming task for task 12")
print(TG.TaskList[12].incomming_tasks)

print(">> outgoing task for task 12")
print(TG.TaskList[12].outgoing_tasks)

print(">> run naive assignment")
s.naive_assigment(TG.n_tasks)

TG.assign_address("linearXYZ", s)

print(">> the current address of task 12")

TG.TaskList[12].mapped_cluster_address.print()

print(">> the ID task[0] of cluster[0][1][1]")

print(s.nodes[0][1][1].TaskList[0].ID)

print(">> the final communication cost")

print(TG.CommCost(s))
print(TG.CommCostMLP(s))

print("max-group: "+str(TG.CommCostMLPMax(s)))
