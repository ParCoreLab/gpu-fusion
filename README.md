# GPU Fusion

This repository includes the open source code for GPU fusion, enabling memory-access optimized kernel fusion for GPU batch jobs.

Jobs have to be defined in a certain format (see `BFSJob` inside the code for example).

After compilation with CUDA, the code can be run the following way:

```
ALGO=BFS FUSION=1 ./fusion path-to-graph number
```

Where `path-to-graph` is path to the adjacency graph file (converted via Ligra's [SNAPtoAdj](https://github.com/jshun/ligra/blob/master/utils/SNAPtoAdj.C) for example), and `number` is the number of concurrent jobs to run at the same time.

The composition of jobs can be modified in `make_jobs` function (which is unrelated to the fusion code, and is just a utility function).

Currently implemented algorithms/jobs are:

* BFS
* SSSP
* PageRank
* LabelPropagation

The code provides comparisons with sequential execution as well as fine-grained profiling information if fusion is used.

The entirety of the code is bundled in one file `fusion.cu` for ease of transport, but can be separated into multiple files if needed.

