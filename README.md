# HeterGenMap:  An Evolutionary Mapping Framework for Heterogeneous NoC-based Neuromorphic Systems

Source code for the paper: Khanh N. Dang, Nguyen Anh Vu Doan, Ngo-Doanh Nguyen, Abderazek Ben Abdallah, ''HeterGenMap: An Evolutionary Mapping Framework for Heterogeneous NoC-based Neuromorphic Systems'', IEEE Access, (accepted), 2023.

## Dependencies

In general, the HeterGenMap has very few dependencies:

- Python (3.6 or higher)
- deap: [https://deap.readthedocs.io/en/master/](https://deap.readthedocs.io/en/master/)

## Examples

- [GA_map_deap_noFT.py](GA_map_deap_noFT.py): mapping for no fault-tolerance.
- [GA_map_deap_FT.py](GA_map_deap_FT.py): mapping for no fault-tolerance.
- [GA_map_deap_MC.py](GA_map_deap_MC.py): mapping for multi-chip system.


## Configurations

### Multi-chip

The source came with a pre-defined set of `slink` (links between chips). Please check out [HeterGenMap/SystemX.py](HeterGenMap/SystemX.py) from line 43-149.
The cost of `slink` can be adjusted.

### Fault-tolerance

The forbidden (faulty and cannot be used for routing) paths can be adjusted in  [HeterGenMap/SystemX.py](HeterGenMap/SystemX.py) at function `gen_defected_links`


## Contact

If you have any questions or requests, please get in touch with Khanh N. Dang at `khanh` @ `u-aizu` dot `ac` dot `jp`
