# MatrixIBD
MatrixIBD software was designed to reconstruct recent multi-population migration history by using identical-by-descent sharing.
## Modified ms
Modified ms is also called ms-IBD in this paper.
## Requirements
- python 3.7.16
- nlopt 2.7.1

## Data
For three-population, six list used. IBD sharing between pop11, pop22, pop33, pop12, pop13, pop23. 

## Example

> python main.py --mode nd_md --data-path ./data/nd_md.csv --par-path  ./par/par_nd_md.json --result-path ./result/ps_n3_nd_md_n500.txt


> python main.py --mode nd_ms --data-path ./data/nd_ms.csv --par-path  ./par/par_nd_ms.json --result-path ./result/ps_n3_nd_ms_n500.txt


> python main.py --mode ns_md --data-path ./data/ns_md.csv --par-path  ./par/par_ns_md.json --result-path ./result/ps_n3_ns_md_n500.txt


> python main.py --mode ns_ms --data-path ./data/ns_ms.csv --par-path  ./par/par_ns_ms.json --result-path ./result/ps_n3_ns_ms_n500.txt

If you use our software in your research, please cite the following paper:
Reconstruct recent multi-population migration history by using identical-by-descent sharing.
