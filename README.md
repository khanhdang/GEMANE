# HeterGenMap:  An Evolutionary Mapping Framework for Heterogeneous NoC-based Neuromorphic Systems

Source code for the paper: Khanh N. Dang, Nguyen Anh Vu Doan, Ngo-Doanh Nguyen, Abderazek Ben Abdallah, ''HeterGenMap: An Evolutionary Mapping Framework for Heterogeneous NoC-based Neuromorphic Systems'', IEEE Access, (accepted), 2023. \[[PDF](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10366249)\]

## Dependencies

In general, the HeterGenMap has very few dependencies:

- Python (3.6 or higher)
- deap: [https://deap.readthedocs.io/en/master/](https://deap.readthedocs.io/en/master/)

## Examples

- [GA_map_deap_noFT.py](GA_map_deap_noFT.py): mapping for no fault-tolerance.
- [GA_map_deap_FT.py](GA_map_deap_FT.py): mapping for no fault-tolerance. See [this](https://colab.research.google.com/drive/1C9W8oRPbxi7NqIPawyb3EBWi3ex_KscT?usp=sharing) for Google Collab example. 
- [GA_map_deap_MC.py](GA_map_deap_MC.py): mapping for multi-chip system.


## Configurations

### Multi-chip

The source came with a pre-defined set of `slink` (links between chips). Please check out [HeterGenMap/SystemX.py](HeterGenMap/SystemX.py) from line 43-149.
The cost of `slink` can be adjusted.

For instance, the following snippet indicates the special links between two chips of (2x2x2) connected to form a 2x2x4 system.


```
        if (self.system_dim.Z == 2 and self.system_dim.Y == 2  and self.system_dim.X == 4 ):
            self.list_of_slinks = [Link(Coordinate(0,0,1), Coordinate(0,0,2)), \
                                    Link(Coordinate(0,1,1), Coordinate(0,1,2)),\
                                    Link(Coordinate(1,0,1), Coordinate(1,0,2)), \
                                    Link(Coordinate(1,1,1), Coordinate(1,1,2)) ]
```

To change the cost of `slink` of a system `s`: 
```
s.slink_cost = value_that_you_want
```


### Fault-tolerance

The forbidden (faulty and cannot be used for routing) paths can be adjusted in  [HeterGenMap/SystemX.py](HeterGenMap/SystemX.py) at the function `gen_defected_links`.
This function will generate randomized faulty links inside the system.
Note: If the fault rate is high, there is a chance of generating an isolated area that cannot be routed to even with non-minimal routing.


## Contact

If you have any questions or requests, please get in touch with Khanh N. Dang at `khanh` @ `u-aizu` dot `ac` dot `jp`
