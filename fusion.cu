#include <vector>
#include <cassert>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <set>



#include <chrono>
namespace custom {
	// Public
	class Timer {
		private:
			bool running = false;
			double total = 0.0;
			std::string name;
			std::chrono::steady_clock::time_point t_start;
		public:
			Timer(std::string name) : name(name) {}
			void pause() {
				if (!running) {
					throw new std::runtime_error("Cannot pause a paused timer");
				}
				running = false;
				total += std::chrono::duration_cast<std::chrono::duration<double>>
					(std::chrono::steady_clock::now() - t_start).count();
			}
			void start() {
				running = true;
				t_start = std::chrono::steady_clock::now();
			}
			void reset() {
				total = 0;
				running = false;
			}
			void report() {
				if (running)
					throw new std::runtime_error("Report called on a running timer");
				std::cout << name << " took " << seconds() << " seconds." << std::endl;
			}
			double seconds() {
				return total;
			}
	};
}


using namespace std;

typedef uint64_t u64;
typedef int64_t i64;

#define GPU_ERROR_CHECK(x) { gpuAssert((x), __FILE__, __LINE__); }

#define safeCudaPeekAtLastError(...) GPU_ERROR_CHECK(cudaPeekAtLastError(__VA_ARGS__));
#define safeCudaMalloc(...) GPU_ERROR_CHECK(cudaMalloc(__VA_ARGS__));
#define safeCudaMemcpy(...) GPU_ERROR_CHECK(cudaMemcpy(__VA_ARGS__));
#define safeCudaDeviceSynchronize(...) GPU_ERROR_CHECK(cudaDeviceSynchronize(__VA_ARGS__));

auto GPU_time = custom::Timer("GPU");
auto GPU_prep_time = custom::Timer("GPU preparation");

auto fusion_time = custom::Timer("Fusion");

inline const string env2str(const char * env) {
	char const* env_value = getenv(env);
	string str = (env_value == NULL ? string() : string(env_value));
	return str;
}
inline bool isFusionOn() {
	return env2str("FUSION") == "1";
}
inline bool isSameRootOn() {
	return env2str("SAMEROOT") == "1";
}
inline const string getAlgo() {
	string algo = env2str("ALGO");
	return algo;
}

__host__ inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
	if (code == cudaSuccess) return;
	fprintf(stderr, "GPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
	if (abort) exit(code);
}

class AdjacencyGraph {
	public:
		u64 n;
		u64 m;
		u64* offsets;
		u64* edges;
		AdjacencyGraph(u64 n, u64 m) : n(n), m(m), offsets{new u64[n]}, edges{new u64[m]} {}
};

__host__ AdjacencyGraph load_adjacency_graph(const string& path) {
	auto file = ifstream(path);

	string t;
	u64 n, m;

	file >> t >> n >> m;

	assert(t == "AdjacencyGraph");

	auto graph = AdjacencyGraph(n, m);

	for (auto i = 0; i < n; i++) file >> graph.offsets[i];
	for (auto i = 0; i < m; i++) file >> graph.edges[i];

	return graph;
}
// TODO(metacompiler): create this enum based on job list
enum JobTypes {
	Abstract,
	BFS,
	SSSP,
	PageRank,
	LabelPropagation,
	// Jaccard,
};

class AbstractJob {
public:
	JobTypes job_type; // holds type of job for static polymorphism

	bool done = false; // iteration done
	bool xdone = false; // permanently done

	bool active = true; // active per fusion decision or not
	bool* frontier; // frontier, i.e., active set of nodes that are being worked on

	__host__ void init(const AdjacencyGraph& graph) {};


	__device__ void iter(const u64 id, const u64* d_offsets, const u64* d_edges, const u64 n, const u64 m) {};


	template <typename T>
	T* allocAndCopy(T* h_data, u64 size) {
		auto total_size = size * sizeof(T);
		T *d_pointer;
		safeCudaMalloc(&d_pointer, total_size);
		safeCudaMemcpy(d_pointer, h_data, total_size, cudaMemcpyHostToDevice);
		return d_pointer;
	}
};

class BFSJob: public AbstractJob {
	public:

//PRAGMA: job data
	u64 root;
	i64* parents;
//PRAGMA: end job data

	BFSJob(u64 root) : root(root) {job_type = BFS;}

	__host__ void init(const AdjacencyGraph& graph) {
		bool* h_frontier = new bool[graph.n];
		i64* h_parents = new i64[graph.n];
		for (u64 i = 0; i < graph.n; i++) {
			h_frontier[i] = i == this->root;
			h_parents[i] = i == this->root ? (i64) this->root : (i64) -1;
		}

		this->frontier = this->allocAndCopy<bool>(h_frontier, graph.n);
		this->parents = this->allocAndCopy<i64>(h_parents, graph.n);
	}

	__device__ void iter(const u64 id, const u64* d_offsets, const u64* d_edges, const u64 n, const u64 m) {
		const u64 end_node = id < n - 1 ? d_offsets[id + 1] : n;

		this->frontier[id] = false;

		for (auto i = d_offsets[id]; i < end_node; i++) {
			const auto next = d_edges[i];

			if (this->parents[next] != -1) continue;
			this->frontier[next] = true;

			this->parents[next] = id;
			this->done = false; // Updated results, mark as not finished.
		}
	}
};



class SSSPJob: public AbstractJob {
	public:

//PRAGMA: job data
	u64 root;
	i64* distance;
//PRAGMA: end job data

	SSSPJob(u64 root) : root(root) {job_type = SSSP;}

	__host__ void init(const AdjacencyGraph& graph) {
		bool* h_frontier = new bool[graph.n];
		i64* h_distance = new i64[graph.n];
		for (u64 i = 0; i < graph.n; i++) {
			h_frontier[i] = i == this->root;
			h_distance[i] = (i == this->root ? (i64) 0 : (i64) INT_FAST64_MAX);
		}

		this->frontier = this->allocAndCopy<bool>(h_frontier, graph.n);
		this->distance = this->allocAndCopy<i64>(h_distance, graph.n);
	}

	__device__ void iter(const u64 id, const u64* d_offsets, const u64* d_edges, const u64 n, const u64 m) {
		const u64 end_node = id < n - 1 ? d_offsets[id + 1] : n;

		this->frontier[id] = false;

		for (auto i = d_offsets[id]; i < end_node; i++) {
			const auto next = d_edges[i];

			if (this->distance[next] != INT_FAST64_MAX) continue;
			this->frontier[next] = true;


			if (this->distance[id] + 1 < this->distance[next])
				this->distance[next] = this->distance[id] + 1;
				// atomicAdd((long long *)&this->distance[next], (long long )1);

			this->done = false; // Updated results, mark as not finished.
		}
	}
};



class PageRankJob: public AbstractJob {
	public:

//PRAGMA: job data
	u64 root;
	u64* visited;
	double* pagerank;
	u64 max_iterations;
//PRAGMA: end job data

	PageRankJob(const u64 root, const u64 max_iterations = 5)
	:root(root), max_iterations(max_iterations) {
		job_type = PageRank;
		// printf("0Max iterations: %d\n", this->max_iterations);
	}

	__host__ void init(const AdjacencyGraph& graph) {
		// printf("1Max iterations: %d\n", this->max_iterations);
		const double initial_pagerank = double(1)/graph.n;

		bool* h_frontier = new bool[graph.n];\
		double* h_pagerank = new double[graph.n];
		u64* h_visited = new u64[graph.n];

		for (u64 i = 0; i < graph.n; i++) {
			h_frontier[i] = i == this->root;
			h_pagerank[i] = initial_pagerank;
			h_visited[i] = 0;
		}

		this->frontier = this->allocAndCopy<bool>(h_frontier, graph.n);
		this->pagerank = this->allocAndCopy<double>(h_pagerank, graph.n);
		this->visited = this->allocAndCopy<u64>(h_visited, graph.n);
	}

	__device__ void iter(const u64 id, const u64* d_offsets, const u64* d_edges, const u64 n, const u64 m) {
		const u64 end_node = id < n - 1 ? d_offsets[id + 1] : n;

		this->frontier[id] = false;

		if (++this->visited[id] >= this->max_iterations) return;

		const double share = this->pagerank[id] / (end_node - d_offsets[id] + 1);
		this->pagerank[id] = 0; // clear out my share

		for (auto i = d_offsets[id]; i < end_node; i++) {
			const auto next = d_edges[i];

			this->frontier[next] = true;

			this->pagerank[next] += share;
			// atomicAdd(&this->pagerank[next], share); // needs sm6_1

			this->done = false; // Updated results, mark as not finished.
		}
	}
};


class LabelPropagationJob: public AbstractJob {
	public:

//PRAGMA: job data
	u64 root;
	u64* visited;
	u64* label;
	u64 max_labels;
//PRAGMA: end job data

	LabelPropagationJob(u64 root, const u64 max_labels = 10)
	:root(root),max_labels(max_labels) {job_type = LabelPropagation;}

	__host__ void init(const AdjacencyGraph& graph) {
		bool* h_frontier = new bool[graph.n];
		u64* h_label = new u64[graph.n];
		u64* h_visited = new u64[graph.n];
		for (u64 i = 0; i < graph.n; i++) {
			h_frontier[i] = i == this->root;
			h_label[i] = (i * 13) % max_labels;
			h_visited[i] = 0;
		}

		this->frontier = this->allocAndCopy<bool>(h_frontier, graph.n);
		this->label = this->allocAndCopy<u64>(h_label, graph.n);
		this->visited = this->allocAndCopy<u64>(h_visited, graph.n);
	}

	__device__ void iter(const u64 id, const u64* d_offsets, const u64* d_edges, const u64 n, const u64 m) {
		const u64 end_node = id < n - 1 ? d_offsets[id + 1] : n;

		this->frontier[id] = false;

		if (this->visited[id]++) return; // Visit each once.


		int *counting_sort = new int[this->max_labels];
		for (int i=0; i<this->max_labels; ++i) counting_sort[i]=0;

		// Moore's majority count
		// int x;
		// int c = 0;

		for (auto i = d_offsets[id]; i < end_node; i++) {
			const auto next = d_edges[i];

			this->frontier[next] = true;

			counting_sort[this->label[next]]++;

			// if (c == 0) x = this->label[next];
			// else if (x == this->label[next]) c++;
			// else c--;


			this->done = false; // Updated results, mark as not finished.
		}
		int max = -1, max_index = -1;
		for (int i=0; i<this->max_labels; ++i)
			if (counting_sort[i] > max) {
				max = counting_sort[i];
				max_index = i;
			}
		this->label[id] = max_index; // Update my label to the max of my neighbors.
		// this->label[id] = x; // Update my label to the max of my neighbors.

		delete[] counting_sort;
	}
};

/**
 * Calculate Jaccard similarity score of all nodes with respect to "root"
class JaccardJob: public AbstractJob {
	public:

//PRAGMA: job data
	u64 root;
	u64 *visited;
	double* jaccard;
//PRAGMA: end job data

	JaccardJob(u64 root) : root(root) {job_type = Jaccard;}

	__host__ void init(const AdjacencyGraph& graph) {
		bool* h_frontier = new bool[graph.n];
		u64* h_visited = new u64[graph.n];
		double* h_jaccard = new double[graph.n];
		for (u64 i = 0; i < graph.n; i++) {
			h_frontier[i] = i == this->root;
			h_jaccard[i] = (i == this->root ? (double) 1 : (i64) 0.5);
			h_visited[i] = 0;
		}

		this->frontier = this->allocAndCopy<bool>(h_frontier, graph.n);
		this->jaccard = this->allocAndCopy<double>(h_jaccard, graph.n);
		this->visited = this->allocAndCopy<u64>(h_visited, graph.n);
	}

	__device__ void iter(const u64 id, const u64* d_offsets, const u64* d_edges, const u64 n, const u64 m) {
		const u64 start_node = d_offsets[id];
		const u64 end_node = id < n - 1 ? d_offsets[id + 1] : n;

		this->frontier[id] = false;


		const u64 root_end_node = this->root < n - 1 ? d_offsets[this->root + 1] : n;
		const u64 root_start_node = d_offsets[this->root];
		const auto root_neighborCount = root_end_node - root_start_node + 1;

		for (auto i = start_node; i < end_node; i++) {
			const auto next = d_edges[i];
			if (next == this->root) continue;
			if (this->visited[next]) continue;

			this->visited[next] = this->visited[id] + 1;

			// Only 2 layers of expansion can have shared neighbors with root
			if (this->visited[next] < 2)
				this->frontier[next] = true;

			const auto next_end_node = next < n - 1 ? d_offsets[next + 1] : n;
			const auto next_start_node = d_offsets[next];
			const auto next_neighborCount = next_end_node - next_start_node + 1;

			// Find shared neighbors between nodes root and next:
			int shared_neighborCount = 0;
			// Assumes edge list is sorted for each node, doing a shared count:
			// . . 2 3 4 5 . . . . .
			// . . . . . . . 1 2 5 7 . . . .
			auto root_pivot = root_start_node;
			auto next_pivot = next_start_node;
			int j = 0;
			while (root_pivot < root_end_node || next_pivot < next_end_node) {
				if (j++ > n) {
					printf("Infinite loop!\n");
					break;
				}
				const auto root_value = d_edges[root_pivot];
				const auto next_value = d_edges[next_pivot];

				if (root_value == next_value) {
					shared_neighborCount++;
					root_pivot++;
					next_pivot++;
				} else if (root_value < next_value) {
					root_pivot++;
				} else if (root_value > next_value) {
					next_pivot++;
				}

			}
			// printf("Shared neighborcount %lu and %lu: %d\n", root, next, shared_neighborCount);
			this->jaccard[next] = shared_neighborCount / double(root_neighborCount + next_neighborCount);

			this->done = false; // Updated results, mark as not finished.
		}
	}
};
*/

// TODO(metacompiler): This class needs to be generated dynamically based on all job data
struct FusionJob: public AbstractJob {
	u64 root;
	union {
		i64* parents;
		i64* distance;
		u64* visited;
	};
	union {
		double* pagerank;
		u64* label;
		// double* jaccard;
	};
	union {
		u64 max_iterations;
		u64 max_labels;
	};
};

// Kernel:
__global__ void iter_jobs(FusionJob* jobs, const u64 count,
	const u64* d_offsets, const u64* d_edges, const u64 n, const u64 m) {
	// Get our global thread ID
	const u64 id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= n) return;

	for (auto i = 0; i < count; i++) {

		if (!jobs[i].active) continue; // This job is not active for this iteration, skip.
		if (jobs[i].xdone) continue; // This job is already completely done.
		if (!jobs[i].frontier[id]) continue; // This thread id is not in this job's frontier.

		// jobs[i].iter(id, d_offsets, d_edges, n, m);

		FusionJob * job = &jobs[i];
		// TODO(metacompiler): generate dynamically based on JobTypes enum:
		if (job->job_type == JobTypes::BFS)
			((BFSJob *)job)->iter(id, d_offsets, d_edges, n, m);
		else if (job->job_type == JobTypes::SSSP)
			((SSSPJob *)job)->iter(id, d_offsets, d_edges, n, m);
		else if (job->job_type == JobTypes::PageRank)
			((PageRankJob *)job)->iter(id, d_offsets, d_edges, n, m);
		else if (job->job_type == JobTypes::LabelPropagation)
			((LabelPropagationJob *)job)->iter(id, d_offsets, d_edges, n, m);
		// else if (job->job_type == JobTypes::Jaccard)
		// 	((JaccardJob *)job)->iter(id, d_offsets, d_edges, n, m);
		else
			printf("Unsupported job type!\n");

	}
}

__global__ void micro_fusion_kernel_phase1(const int count, const int graph_size,
	FusionJob* d_jobs, int *d_shared_frontier, int *d_job_frontier_size) {
	const u64 id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id >= graph_size) return;

	for (int i = 0; i < count; i++) {
		if (d_jobs[i].xdone) continue; // this job is already finished, ignore.
		if (!d_jobs[i].frontier[id]) continue; // not active frontier
		atomicAdd(&d_job_frontier_size[i], 1);
		d_shared_frontier[id]++;
	}
}

__global__ void micro_fusion_kernel_phase2(const int count, const int graph_size,
	const int max_index, FusionJob* d_jobs, int *active_count) {
	const u64 id = blockIdx.x * blockDim.x + threadIdx.x;

	if (id != 0) {
		printf("This shouldn't happen!\n");
		return;
	}

	*active_count = 0;

	for (int i = 0; i < count; i++) {
		if (d_jobs[i].xdone) continue; // this job is already finished, ignore.

		// The max vertex is used in this job, mark as active:
		if (d_jobs[i].frontier[max_index]) {
			d_jobs[i].active = true;
			(*active_count)++;
		}
		else
			d_jobs[i].active = false;
	}
}

auto t1 = custom::Timer("t1");
auto t2 = custom::Timer("t2");
auto t3 = custom::Timer("t3");
auto t4 = custom::Timer("t4");
auto t5 = custom::Timer("t5");
bool fusion(const int count, const int depth, const int graph_size, FusionJob* h_jobs, FusionJob* d_jobs) {
	if (depth < 1) // Heuristic: don't fuse early on, small frontier and no overlap
		return false;

	fusion_time.start();

	static int *shared_frontier = 0;
	/// FUSION ///

	// Find shared frontier, i.e., union of all job frontiers:
	t1.start();
	cout << "\tIteration " << depth << ") " << "Finding shared frontier..." << endl;
	if (shared_frontier == 0)
		shared_frontier = new int[graph_size];
	if (!shared_frontier) perror ("Could not allocate shared frontier");

	int *job_frontier_size = new int[count];
	if (!job_frontier_size) perror ("Could not allocate job frontier size");
	t1.pause();

	t2.start();
	// Doing the shared_frontier calculation in GPU
	static int * d_shared_frontier = 0;
	static int * d_job_frontier_size = 0;
	if (d_shared_frontier == 0) {
		safeCudaMalloc(&d_shared_frontier, sizeof(int) * graph_size);
		if (!d_shared_frontier) perror ("Could not allocate shared frontier");
	}
	if (d_job_frontier_size == 0) {
		safeCudaMalloc(&d_job_frontier_size, sizeof(int) * count);
		if (!d_job_frontier_size) perror ("Could not allocate shared frontier");
	}

	const u64 blockSize = 1024;
	u64 gridSize = (u64) ceil((float) graph_size / blockSize);

	cudaMemset(d_shared_frontier, 0, sizeof(int) * graph_size); // set to zero
	cudaMemset(d_job_frontier_size, 0, sizeof(int) * count); // set to zero

	micro_fusion_kernel_phase1<<<gridSize, blockSize>>>(count, graph_size, d_jobs, d_shared_frontier, d_job_frontier_size);

	safeCudaMemcpy(shared_frontier, d_shared_frontier, sizeof(int)*graph_size, cudaMemcpyDeviceToHost);
	safeCudaMemcpy(job_frontier_size, d_job_frontier_size, sizeof(int)*count, cudaMemcpyDeviceToHost);
	t2.pause();


	t3.start();
	cout << "\t\tJob frontier sizes: ";
	for (int job_id = 0; job_id < count; ++job_id)
		cout << job_frontier_size[job_id] << ", ";
	cout << endl;
	t3.pause();


	t4.start();
	cout << "\tIteration " << depth << ") " << "Finding max frontier (largest vertex of shared frontier)..." << endl;
	// Find the frontier vertex with most jobs:
	int max = -1;
	int max_index = 0;
	for (int i=0; i<graph_size; ++i) {
		if (shared_frontier[i] > max) {
			max = shared_frontier[i];
			max_index = i;
		}
	}
	cout << "\tIteration " << depth << ") " << "Max frontier vertex id " << max_index << " shared among " << max << " jobs." << endl;
	t4.pause();

	if (max < 3) { // Heuristic: too few jobs to be worth for fusion
		fusion_time.pause();
		return false;
	}

	t5.start();
	int active_count, *d_active_count;
	safeCudaMalloc(&d_active_count, sizeof(int));
	micro_fusion_kernel_phase2<<<1, 1>>>(count, graph_size, max_index, d_jobs, d_active_count);
	safeCudaMemcpy(&active_count, d_active_count, sizeof(int), cudaMemcpyDeviceToHost);
	// safeCudaFree(d_active_count);
	cout  << "\tIteration " << depth << ") " << "Marked " << active_count << " jobs active." << endl;
	t5.pause();

	/// END FUSION ///

	fusion_time.pause();
	return true;
}

int run_jobs(FusionJob *jobs, const int count,
	const AdjacencyGraph& graph, const u64* d_offsets, const u64* d_edges) {

	// FusionJob* h_jobs = (FusionJob*) malloc(count * sizeof(FusionJob));
	FusionJob* d_jobs;
	safeCudaMalloc(&d_jobs, count * sizeof(FusionJob));
	safeCudaMemcpy(d_jobs, jobs, count * sizeof(FusionJob), cudaMemcpyHostToDevice);

	const int graph_size = graph.n;
	const u64 blockSize = 1024;
	u64 gridSize = (u64) ceil((float) graph_size / blockSize);

	// This while runs as much as the max depth of the job:
	int depth = 0;
	while (true) {
		if (count > 1 && isFusionOn())
			fusion(count, depth, graph_size, jobs, d_jobs);

		GPU_prep_time.start();
		auto done = true; // Assume all done.
		bool job_done;
		bool job_active;
		for (auto i = 0; i < count; i++) {
			safeCudaMemcpy(&job_active, &(d_jobs[i].active), sizeof(bool), cudaMemcpyDeviceToHost);
			jobs[i].active = job_active; // update jobs object for fusion
			if (!job_active) continue;

			// Check if this job is done:
			safeCudaMemcpy(&job_done, &(d_jobs[i].done), sizeof(bool), cudaMemcpyDeviceToHost);
			jobs[i].done = job_done; // update job objcet for fusion
			if (!job_done) {
				done = false; // There was an update on this job, all is not yet done.
				job_done = true; // Mark this specific job as done again.
				safeCudaMemcpy(&(d_jobs[i].done), &job_done, sizeof(bool), cudaMemcpyHostToDevice);
			} else { // This job is not updated
				// printf("Marking job %d as fully done!\n", i);
				job_done = true; // Mark this specific job as fully done (done 2 iterations in a row).
				safeCudaMemcpy(&(d_jobs[i].xdone), &job_done, sizeof(bool), cudaMemcpyHostToDevice);
				jobs[i].xdone = true; // update job object for fusion
			}
		}
		GPU_prep_time.pause();

		if (done) break; // All done.
		depth++;

		GPU_time.start();
		iter_jobs<<<gridSize, blockSize>>>(d_jobs, count, d_offsets, d_edges, graph.m, graph.n);
		safeCudaPeekAtLastError();
		safeCudaDeviceSynchronize();
		GPU_time.pause();

		/// DEBUG INFO
		if (graph_size < 10) {
			FusionJob *h_jobs = (FusionJob*)malloc(sizeof(FusionJob) * count);
			safeCudaMemcpy(h_jobs, d_jobs, sizeof(FusionJob)*count, cudaMemcpyDeviceToHost);
			for (int i=0; i<count; ++i) {
				FusionJob * job = &h_jobs[i];
				printf("Job %d, Iteration %d:\n", i, depth);
				printf("\troot: %lu\n", job->root);
				printf("\tactive: %d\n", job->active);
				printf("\tdone: %d\n", job->done);
				printf("\txdone: %d\n", job->xdone);
				bool *h_frontier = new bool[graph_size];
				safeCudaMemcpy(h_frontier, job->frontier, sizeof(bool)*graph_size, cudaMemcpyDeviceToHost);
				printf("\tfrontier: ");
					for (int j=0; j<graph_size; ++j) if (h_frontier[j]) printf("%d, ", j);
				printf("\n");
				if (job->job_type == JobTypes::BFS) {
					i64 *h_parents = new i64[graph_size];
					safeCudaMemcpy(h_parents, job->parents, sizeof(i64)*graph_size, cudaMemcpyDeviceToHost);
					printf("\tparents: ");
					for (int j=0; j<graph_size; ++j) {
						printf("%ld, ", h_parents[j]);
					}
				} else if (job->job_type == JobTypes::SSSP) {
					i64 *h_distance = new i64[graph_size];
					safeCudaMemcpy(h_distance, job->distance, sizeof(i64)*graph_size, cudaMemcpyDeviceToHost);
					printf("\tdistance: ");
					for (int j=0; j<graph_size; ++j) {
						printf("%ld, ", h_distance[j]);
					}

				// } else if (job->job_type == JobTypes::Jaccard) {
				// 	double *h_jaccard = new double[graph_size];
				// 	safeCudaMemcpy(h_jaccard, job->jaccard, sizeof(double)*graph_size, cudaMemcpyDeviceToHost);
				// 	printf("\tjaccard: ");
				// 	for (int j=0; j<graph_size; ++j) {
				// 		printf("%lf, ", h_jaccard[j]);
				// 	}

				}
				printf("\n");
			}
		}
	}
	return depth;
}
/**
 * Creates count jobs of the same type, intializes them and memcpy into fusionjob array
 */
template <class JobType>
__host__ void create_jobs(FusionJob *jobs, const int count, const AdjacencyGraph& graph, const int offset = 0) {
	for (auto i = 0; i < count; i++) {
		int root_vertex_index;

		if (isSameRootOn())
			root_vertex_index = 0;
		else
			// root_vertex_index = ((i + offset) ) % graph.n;
			root_vertex_index = ((i + offset) * 10) % graph.n;

		JobType job = JobType(root_vertex_index);
		job.init(graph);
		// jobs[i] = (FusionJob) job;
		// memcpy(&jobs[i], &job, sizeof(FusionJob)); // Careful here!
		memcpy(&jobs[i], &job, sizeof(job)); // Careful here!
	}
}

/**
 * Helper to create jobs based on chosen algorithm
 */
__host__ FusionJob *make_jobs(const string algo, const int count, const AdjacencyGraph& graph) {

	auto job_creation_time = custom::Timer("Creating jobs");
	job_creation_time.start();

	FusionJob * h_jobs = (FusionJob*) malloc(count * sizeof(FusionJob));
	if (algo=="BFS")
		create_jobs<BFSJob>(h_jobs, count, graph);
	else if (algo == "SSSP")
		create_jobs<SSSPJob>(h_jobs, count, graph);
	else if (algo == "PAGERANK")
		create_jobs<PageRankJob>(h_jobs, count, graph);
	else if (algo == "LABEL")
		create_jobs<LabelPropagationJob>(h_jobs, count, graph);
	else if (algo == "MIX") {
		for (int i=0; i<count; ++i) {
			if (i%4 == 0)
				create_jobs<BFSJob>(&h_jobs[i], 1, graph, i);
			else if (i%4 == 1)
				create_jobs<SSSPJob>(&h_jobs[i], 1, graph, i);
			else if (i%4 == 2)
				create_jobs<PageRankJob>(&h_jobs[i], 1, graph, i);
			else if (i%4 == 3)
				create_jobs<LabelPropagationJob>(&h_jobs[i], 1, graph, i);
		}
	}
	else if (algo == "HETERO1") {
		for (int i=0; i<count; ++i) {
			if (i%2 == 0)
				create_jobs<BFSJob>(&h_jobs[i], 1, graph, i);
			else if (i%2 == 1)
				create_jobs<SSSPJob>(&h_jobs[i], 1, graph, i);
		}
	}
	else if (algo == "HETERO2") {
		for (int i=0; i<count; ++i) {
			if (i%3 == 0)
				create_jobs<BFSJob>(&h_jobs[i], 1, graph, i);
			else if (i%3 == 1)
				create_jobs<SSSPJob>(&h_jobs[i], 1, graph, i);
			else if (i%3 == 2)
				create_jobs<PageRankJob>(&h_jobs[i], 1, graph, i);
		}
	}
	else if (algo == "HETERO3") {
		for (int i=0; i<count; ++i) {
			if (i%2 == 0)
				create_jobs<BFSJob>(&h_jobs[i], 1, graph, i);
			else if (i%2 == 1)
				create_jobs<LabelPropagationJob>(&h_jobs[i], 1, graph, i);
		}
	}
	else if (algo == "HETERO4") {
		for (int i=0; i<count; ++i) {
			if (i%2 == 0)
				create_jobs<SSSPJob>(&h_jobs[i], 1, graph, i);
			else if (i%2 == 1)
				create_jobs<LabelPropagationJob>(&h_jobs[i], 1, graph, i);
		}
	}
	else if (algo == "HETERO5") {
		for (int i=0; i<count; ++i) {
			if (i%2 == 0)
				create_jobs<PageRankJob>(&h_jobs[i], 1, graph, i);
			else if (i%2 == 1)
				create_jobs<LabelPropagationJob>(&h_jobs[i], 1, graph, i);
		}
	}
	else if (algo == "HETERO6") {
		for (int i=0; i<count; ++i) {
			if (i%3 == 0)
				create_jobs<BFSJob>(&h_jobs[i], 1, graph, i);
			else if (i%3 == 1)
				create_jobs<SSSPJob>(&h_jobs[i], 1, graph, i);
			else if (i%3 == 2)
				create_jobs<LabelPropagationJob>(&h_jobs[i], 1, graph, i);
		}
	}
	else if (algo == "HETERO7") {
		for (int i=0; i<count; ++i) {
			if (i%4 == 0)
				create_jobs<BFSJob>(&h_jobs[i], 1, graph, i);
			else if (i%4 == 1)
				create_jobs<SSSPJob>(&h_jobs[i], 1, graph, i);
			else if (i%4 == 2)
				create_jobs<PageRankJob>(&h_jobs[i], 1, graph, i);
			else if (i%4 == 3)
				create_jobs<LabelPropagationJob>(&h_jobs[i], 1, graph, i);
		}
	}
	else {
		cout << "Unsupport algorithm '" << algo << "'." << endl;
		exit(-1);
	}

	job_creation_time.pause();
	job_creation_time.report();
	return h_jobs;
}

__host__ void print_graph(AdjacencyGraph& graph) {
	for (int i=0; i<graph.n; ++i) {
		const int start_node = graph.offsets[i];
		const int end_node = i < graph.n - 1 ? graph.offsets[i + 1] : graph.m;

		printf("\tNode %d (%d-%d) has edges to: ", i, start_node, end_node);
		for (auto j = start_node; j < end_node; j++) {
			auto next = graph.edges[j];
			printf("%lu, ", next);
		}
		printf("\n");
	}
}

__host__ int main(int argc, char **argv) {
	if (argc < 3) {
		printf("Usage: %s graph count\n", argv[0]);
		exit(-1);
	}
	auto graph_path = string(argv[1]);
	auto job_count = (u64) atoi(argv[2]);

	string algo = getAlgo();
	if (algo == "") {
		algo = "BFS";
		cout <<"No algorithm specified, defaulting to " << algo << endl;
	}
	cout <<"Running algorithm: " << algo << endl;

	if (isFusionOn())
		printf("Running with: FUSION\n");
	else
		printf("Running without: FUSION\n");

	if (isSameRootOn())
		cout << "SAMEROOT: Starting all jobs on the same root.\n";


	cout << "Graph Path: " << graph_path << endl;
	cout << "Job Count: " << job_count << endl;


	auto host_io_time = custom::Timer("Loading graph to CPU");
	host_io_time.start();
	auto graph = load_adjacency_graph(graph_path);
	host_io_time.pause();
	host_io_time.report();
	cout << "Graph size: " << graph.n << " nodes, " << graph.m << " edges" << endl;


	if (graph.n < 10) print_graph(graph);

	auto device_io_time = custom::Timer("Loading graph to GPU");
	u64* d_offsets;
	u64* d_edges;
	auto offsets_size = graph.n * sizeof(u64);
	auto edges_size = graph.m * sizeof(u64);

	device_io_time.start();
	// Allocate memory for each vector on GPU
	safeCudaMalloc(&d_offsets, offsets_size);
	safeCudaMalloc(&d_edges, edges_size);

	// Copy host vectors to device
	safeCudaMemcpy(d_offsets, graph.offsets, offsets_size, cudaMemcpyHostToDevice);
	safeCudaMemcpy(d_edges, graph.edges, edges_size, cudaMemcpyHostToDevice);
	device_io_time.pause();
	device_io_time.report();


	cout << "-----------------------" << std::endl;


	int depth;
	FusionJob* h_jobs;

	/// Separated jobs:
	auto separated_jobs_time = custom::Timer("Running separated jobs");
	GPU_prep_time.reset();
	GPU_time.reset();

	separated_jobs_time.start();

	h_jobs = make_jobs(algo, job_count, graph);

	for (auto i = 0; i < job_count; i++) {
		// FIXME: remove speedup that is used in batch evaluation
		if (i > 5) {
			printf("Skipping other (>5) sequential jobs as there are too many.\n");
			break;
		}
		printf("%d) Jobtype: %d, ", i, h_jobs[i].job_type);
		depth = run_jobs(&h_jobs[i], /* count */ 1, graph, d_offsets, d_edges);
	}
	printf("\n");

	delete[] h_jobs;
	separated_jobs_time.pause();
	separated_jobs_time.report();
	cout << "Max depth on separated jobs was " << depth << endl;
	GPU_prep_time.report();
	GPU_time.report();

	cout << "Running one job took " << (separated_jobs_time.seconds() / job_count) << " seconds." << std::endl;
	cout << "-----------------------" << std::endl;

	/// Merged jobs:
	auto merged_jobs_time = custom::Timer("Running merged jobs");
	GPU_prep_time.reset();
	GPU_time.reset();

	merged_jobs_time.start();

	h_jobs = make_jobs(algo, job_count, graph);

	depth = run_jobs(h_jobs, job_count, graph, d_offsets, d_edges);

	delete[] h_jobs;

	merged_jobs_time.pause();
	merged_jobs_time.report();
	cout << "Max depth on merged jobs was " << depth << endl;
	GPU_prep_time.report();
	GPU_time.report();

	if (isFusionOn()) {
		fusion_time.report();
		t1.report();
		t2.report();
		t3.report();
		t4.report();
		t5.report();
	}
	cerr << graph_path << ","
		<< algo << "," << isFusionOn() << "," << job_count << ","
		<< merged_jobs_time.seconds() << "," << GPU_time.seconds() << ","
		<< fusion_time.seconds() << endl;
}
