import pickle
import json
from hardwares import construct_30_qubit_hardware, simple_10_qubit_coupling_map, simple_20_qubit_coupling_map, torino_coupling_map, construct_fake_ibm_torino, construct_10_qubit_hardware, simple_30_qubit_coupling_map
from instruction import instruction, Instype, parse_program_from_file, construct_qiskit_circuit
from process import all_pairs_distances, plot_process_schedule_on_torino, process, ProcessStatus
from typing import Dict, List, Optional, Tuple
import random
import math
import queue
import threading
import time
from enum import Enum
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator  # <- use AerSimulator (Qiskit 2.x)
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit_ibm_runtime import QiskitRuntimeService



#Mute the qiskit_ibm_runtime logging info
import logging
logging.getLogger("qiskit_ibm_runtime").setLevel(logging.ERROR)



class benchmarktype(Enum):
    RANDOM_SMALL = 0
    RANDOM_MEDIUM = 1
    MULTI_CONTROLLED_X_SMALL = 2
    MULTI_CONTROLLED_X_MEDUIM = 3
    STABILIZER_MEASUREMENT_SMALL =4
    STABILIZER_MEASUREMENT_MEDUIM =5
    CLASSICAL_LOGIC_SMALL =6
    CLASSICAL_LOGIC_MEDUIM = 7
    LCU_SMALL =8
    LCU_MEDUIM =9
    MIX_SMALL = 10
    MIX_MEDUIM = 11



benchmark_file_path={
    benchmarktype.RANDOM_SMALL:"benchmarkdata//randomsmall//",
    benchmarktype.RANDOM_MEDIUM:"benchmarkdata//randommeduim//",
    benchmarktype.MULTI_CONTROLLED_X_SMALL:"benchmarkdata//multiXsmall//",
    benchmarktype.MULTI_CONTROLLED_X_MEDUIM:"benchmarkdata//multiXmeduim//",
    benchmarktype.STABILIZER_MEASUREMENT_SMALL:"benchmarkdata//qecsmall//",
    benchmarktype.STABILIZER_MEASUREMENT_MEDUIM:"benchmarkdata//qecmeduim//",
    benchmarktype.CLASSICAL_LOGIC_SMALL:"benchmarkdata//arithsmall//",
    benchmarktype.CLASSICAL_LOGIC_MEDUIM:"benchmarkdata//arithmeduim//",
    benchmarktype.LCU_SMALL:"benchmarkdata//hamsimsmall//",
    benchmarktype.LCU_MEDUIM:"benchmarkdata//hamsimmeduim//",
}



random_small_benchmark={
    0: "data_8_syn_8_gc_40_1",
    1: "data_10_syn_10_gc_40_1",
    2: "data_10_syn_10_gc_40_2",
}


random_medium_benchmark={
    0: "data_20_syn_20_gc_100_1",
    1: "data_20_syn_20_gc_100_2",
    2: "data_25_syn_25_gc_100_1",
}




qec_small_benchmark={
    0: "cat_state_prep_n4",
    1: "cat_state_verification_n4",
    2: "repetition_code_distance3_n3",
    3: "shor_parity_measurement_n4",
    4: "shor_stabilizer_XZZX_n3",
    5: "shor_stabilizer_ZZZZ_n4",
    6: "syndrome_extraction_surface_n4"
}


qec_meduim_benchmark={
    0: "cat_state_prep_n7",
    1: "cat_state_verification_n7",
    2: "repetition_code_distance5_n5",
    3: "shor_parity_measurement_n7",
    4: "shor_stabilizer_XZZX_n5",
    5: "shor_stabilizer_ZZZZZZ_n6",
    6: "syndrome_extraction_surface_n9"
}


multix_small_benchmark={
    0:"mcx_2",
    1:"mcx_3",
    2:"mcx_4",
    3:"mcx_5",
    4:"mcx_6",
    5:"mcx_7",
    6:"mcx_8",
    7:"mcx_9",
    8:"mcx_10"
}


multix_medium_benchmark={
    0:"mcx_11",
    1:"mcx_12",
    2:"mcx_13",
    3:"mcx_14",
    4:"mcx_15",
    5:"mcx_16",
    6:"mcx_17",
    7:"mcx_18",
    8:"mcx_19"
}


classical_logic_small_benchmark={
    0:"adder_5bit",
    1:"multiplier_3bit",
    2:"comparator_4bit",
    3:"subtracter_5bit",
    4:"divider_4bit",
    5:"modulo_4bit",
}


classical_logic_medium_benchmark={
    0:"adder_10bit",
    1:"multiplier_5bit",
    2:"comparator_8bit",
    3:"subtracter_10bit",
    4:"divider_8bit",
    5:"modulo_8bit",
}


hamilsim_small_benchmark={
    0:"hamilsim_4q_2terms",
    1:"hamilsim_6q_3terms",
    2:"hamilsim_8q_4terms",
    3:"hamilsim_10q_5terms",
}


hamilsim_medium_benchmark={
    0:"hamilsim_6q_4terms",
    1:"hamilsim_8q_6terms",
    2:"hamilsim_10q_8terms",
    3:"hamilsim_12q_10terms"
}


benchmark_type_to_benchmark={
    benchmarktype.MULTI_CONTROLLED_X_SMALL: multix_small_benchmark,
    benchmarktype.MULTI_CONTROLLED_X_MEDUIM: multix_medium_benchmark,
    benchmarktype.STABILIZER_MEASUREMENT_SMALL: qec_small_benchmark,
    benchmarktype.STABILIZER_MEASUREMENT_MEDUIM: qec_meduim_benchmark,
    benchmarktype.CLASSICAL_LOGIC_SMALL: classical_logic_small_benchmark,
    benchmarktype.CLASSICAL_LOGIC_MEDUIM: classical_logic_medium_benchmark,
    benchmarktype.LCU_SMALL: hamilsim_small_benchmark,
    benchmarktype.LCU_MEDUIM: hamilsim_medium_benchmark,
}




class SchedulingOptions(Enum):
    BASELINE_SEQUENTIAL = 0
    NO_SHARING = 1
    HALO =2 
    SHOT_UNAWARE=3


def distribution_fidelity(dist1: dict, dist2: dict) -> float:
    """
    Compute fidelity between two distributions based on L1 distance.
    1. Normalize both distributions to get probability distributions.
    2. Compute the L1 distance between the two probability distributions.
    3. Fidelity = 1 - L1/2, which lies in [0,1].
    
    Args:
        dist1 (dict): key=str (event), value=int (count)
        dist2 (dict): key=str (event), value=int (count)
    
    Returns:
        float: fidelity in [0, 1]
    """
    # Step 1: normalize distributions
    total1 = sum(dist1.values())
    total2 = sum(dist2.values())
    prob1 = {k: v / total1 for k, v in dist1.items()}
    prob2 = {k: v / total2 for k, v in dist2.items()}
    
    # Step 2: union of keys
    all_keys = set(prob1.keys()).union(set(prob2.keys()))
    
    # Step 3: compute L1 distance
    l1_distance = sum(abs(prob1.get(k, 0) - prob2.get(k, 0)) for k in all_keys)
    
    # Step 4: fidelity = 1 - L1/2
    fidelity = 1 - l1_distance / 2
    return fidelity



def load_ideal_count_output(benchmark_type:benchmarktype, benchmark_id: int) -> Dict[str, int]:
    """
    Load the ideal count output from the benchmark suit file
    """
    fileroot=benchmark_file_path[benchmark_type]
    benchmark_dict= benchmark_type_to_benchmark[benchmark_type]
    filename=benchmark_dict[benchmark_id]
    filename_full = fileroot +"result2000shots//"+ filename + "_counts.pkl"
    with open(filename_full, 'rb') as f:
        ideal_counts = pickle.load(f)
    return ideal_counts




"""
Hyper parameters for mapping cost calculation:
alpha: weight for intra-process cost
beta: weight for inter-process cost
gamma: weight for helper qubit cost
delta: weight for compact cost
"""
alpha=0.5
beta=0.3
gamma=1
delta=0.3
eps = 0.3

#Set the scheduling option here
Scheduling_Option=SchedulingOptions.HALO
benchmark_Option=benchmarktype.STABILIZER_MEASUREMENT_SMALL


# N_qubits=133
# hardware_distance_pair=all_pairs_distances(N_qubits, torino_coupling_map())
fake_torino_backend=construct_fake_ibm_torino()


N_qubits=133
hardware_distance_pair=all_pairs_distances(N_qubits, torino_coupling_map())
DIST_MATRIX = np.asarray(hardware_distance_pair, dtype=float)
fake_backend=construct_fake_ibm_torino()


data_hardware_ratio=0.8
#The maximum distance between a data qubit and a helper qubit
max_data_helper_distance=10

# N_qubits=10
# hardware_distance_pair=all_pairs_distances(N_qubits, simple_10_qubit_coupling_map())






def calculate_intro_cost_for_process(process_list: List[process],proc_mapping: Dict[int, Dict[int, int]]) -> float:
    """
    Calculate the intra-process cost for all processes in the batch given the mapping
    The intro cost is defined as the layout mapping cost within each process
    """
    intro_cost=0.0
    #First step, calculate the intra cost of all process
    for proc in process_list:
        intro_cost+=proc.intro_costs(proc_mapping[proc.get_process_id()],DIST_MATRIX)
        # intro_cost+=proc.intro_costs(proc_mapping[proc.get_process_id()],hardware_distance_pair)

    return intro_cost




def calculate_helper_cost_for_process(process_list: List[process],proc_mapping: Dict[int, Dict[int, int]],helper_qubit_list: List[int]) -> float:
    """
    Calculate the helper qubit cost for all processes in the batch given the mapping
    This is defined as the total distance between data qubits and their nearest helper qubits, weighted by the helper qubit weight
    """
    helper_cost=0.0

    if len(helper_qubit_list)==0:
        return float('inf')


    helpers = np.asarray(helper_qubit_list, dtype=int)


    min_dist_to_helper = DIST_MATRIX[:, helpers].min(axis=1)

    #Accumulate helper cost
    helper_cost = 0.0
    for proc in process_list:
        pid = proc.get_process_id()
        mapping = proc_mapping[pid]
        topo = proc.get_topology()
        num_data = proc.get_num_data_qubits()

        for data_qubit in range(num_data):
            helper_weight = topo.get_data_helper_weight(data_qubit)
            if not helper_weight:
                continue
            phys = mapping[data_qubit]
            helper_cost += helper_weight * min_dist_to_helper[phys]

    return helper_cost




def calculate_inter_cost_for_process(process_list: List[process],proc_mapping: Dict[int, Dict[int, int]]) -> float:
    """
    Calculate the inter-process cost for all processes in the batch given the mapping
    The cost is defined as the distance between data qubits of different processes
    """

    #Precompute the list of physical qubits per process
    #Don't look up the dict in the inner most loop
    proc_ids: List[int] = []
    proc_phys_lists: List[np.ndarray] = []

    for proc in process_list:
        pid = proc.get_process_id()
        mapping = proc_mapping[pid]
        num_data = proc.get_num_data_qubits()
        #Create numpy array of phys indices in data-qubit order
        phys_arr = np.fromiter([mapping[dq] for dq in range(num_data)], dtype=int, count=num_data)
        proc_ids.append(pid)
        proc_phys_lists.append(phys_arr)


    inter_cost=0.0
    n = len(process_list)

    for i in range(n):
        proc_i = proc_phys_lists[i]
        if proc_i.size == 0:
            continue

        for j in range(i + 1, n):
            proc_j = proc_phys_lists[j]
            if proc_j.size == 0:
                continue
            # Compute pairwise distances between data qubits of proc_i and proc_j
            sub = DIST_MATRIX[np.ix_(proc_i, proc_j)]
            total_distance = float(sub.sum())
            count = sub.size
            if count > 0:
                inter_cost += total_distance / count


    return inter_cost



def calculate_mapping_cost(process_list: List[process],mapping: Dict[int, tuple[int, int]]) -> float:
    """
    Input:
        mapping: Dict[int, tuple[int, int]]
            A dictionary where the key is physical, and the value is a tuple (process_id, data_qubit_index)
        
        Output:
            A float representing the total mapping cost, calculated as:
            cost = alpha * intro_cost - beta * inter_cost + gamma * helper_cost + delta * compact_cost
    """


    used_phys = set(mapping.keys())
    helper_qubit_list=[phys for phys in range(N_qubits) if phys not in used_phys]


    #----Convert to per-process mapping format----
    proc_mapping: Dict[int, Dict[int, int]] = {pid:{} for pid in [proc.get_process_id() for proc in process_list]}
    for phys,(pid,data_qubit) in mapping.items():
        proc_mapping[pid][data_qubit]=phys


    intro_cost = calculate_intro_cost_for_process(process_list,proc_mapping)
    inter_cost = calculate_inter_cost_for_process(process_list,proc_mapping)
    helper_cost= calculate_helper_cost_for_process(process_list,proc_mapping,helper_qubit_list)


    return alpha * intro_cost - beta * inter_cost + gamma * helper_cost

class MappingCostState:
    """
    FULL general incremental cost engine supporting ALL local changes:
    - swap
    - move to helper
    - move back from helper
    - reassign arbitrary phys

    Normalized components:
      intro_norm   ~ [0, 1]   (intra-process layout cost)
      inter_norm   ~ [0, 1]   (mean distance between processes)
      helper_norm  ~ [0, 1]   (data→helper distances, weighted)
      compact_norm ~ [0, 1]   (cluster tightness per process)
      path_norm    ~ [0, 1]   (NEW: foreign qubits along shortest paths)

    Total cost:
      cost = alpha * intro_norm
           - beta  * inter_norm
           + gamma * helper_norm
           + delta * compact_norm
           + eps   * path_norm
    """

    def __init__(self, process_list, mapping, n_qubits, dist_matrix):
        self.process_list = process_list
        self.n_qubits = n_qubits
        self.dist = dist_matrix

        # phys -> (pid, dq)
        self.mapping = dict(mapping)

        # Process metadata
        self.pid_to_proc = {p.get_process_id(): p for p in process_list}
        self.pids = [p.get_process_id() for p in process_list]
        self.num_proc = len(self.pids)

        # Logical→physical per process
        self.proc_mapping = {pid: {} for pid in self.pids}
        for phys, (pid, dq) in self.mapping.items():
            self.proc_mapping[pid][dq] = phys

        # Used vs helper phys
        used_mask = np.zeros(n_qubits, dtype=bool)
        for phys in self.mapping:
            used_mask[phys] = True
        self.used_mask = used_mask
        self.helper_mask = ~used_mask
        self.helper_qubits = np.where(self.helper_mask)[0]

        # Which process occupies each phys?  -1 means helper/unused
        self.pid_at_phys = np.full(self.n_qubits, -1, dtype=int)
        for phys, (pid, dq) in self.mapping.items():
            self.pid_at_phys[phys] = pid

        # Build adjacency & canonical shortest paths on the hardware graph
        self._build_adjacency()
        self._precompute_shortest_paths()

        # Normalization bounds
        self._init_normalization_bounds()

        # Cost caches (unnormalized sums)
        self.proc_intro_cost   = {}
        self.proc_helper_cost  = {}
        self.proc_compact_cost = {}
        self.inter_cost_pair   = {}

        self.sum_intro   = 0.0
        self.sum_helper  = 0.0
        self.sum_compact = 0.0
        self.sum_inter   = 0.0

        # NEW: path-conflict (unnormalized)
        self.path_conflict_sum = 0.0

        # Precompute helper distances for INITIAL state
        self._compute_min_dist_to_helper()

        # Initialize all pieces
        self._init_process_costs()
        self._init_inter_costs()
        self._compute_path_conflict_full()

    # ------------------------------------------------------------
    # Adjacency + shortest paths on hardware graph
    # ------------------------------------------------------------
    def _build_adjacency(self):
        """Adjacency list from dist == 1 (physical couplings)."""
        self.adj = [[] for _ in range(self.n_qubits)]
        for u in range(self.n_qubits):
            neighs = np.where(self.dist[u] == 1)[0]
            for v in neighs:
                self.adj[u].append(v)

    def _precompute_shortest_paths(self):
        """
        For every pair (u,v), u < v, store one canonical shortest path
        as a list of vertices [u, ..., v].
        """
        from collections import deque

        self.shortest_paths = {}
        for s in range(self.n_qubits):
            prev = [-1] * self.n_qubits
            q = deque([s])
            prev[s] = -2  # root marker

            # BFS
            while q:
                u = q.popleft()
                for v in self.adj[u]:
                    if prev[v] == -1:
                        prev[v] = u
                        q.append(v)

            # Recover paths s→t
            for t in range(s + 1, self.n_qubits):
                if prev[t] == -1:
                    continue  # disconnected (shouldn't happen in your hardware)
                path = []
                cur = t
                while cur != -2:
                    path.append(cur)
                    cur = prev[cur]
                path.append(s)
                path.reverse()
                self.shortest_paths[(s, t)] = path

    # ------------------------------------------------------------
    # Normalization bounds
    # ------------------------------------------------------------
    def _init_normalization_bounds(self):
        """
        Compute heuristic upper bounds for intro, helper, inter, compact,
        and path-conflict so we can normalize each term to ~[0,1].
        """
        max_dist = float(self.dist.max()) if self.dist.size > 0 else 1.0
        self.max_dist = max_dist

        # intro_max: complete graph over data qubits per process
        intro_max = 0.0
        for p in self.process_list:
            d = p.get_num_data_qubits()
            num_edges = d * (d - 1) / 2.0
            intro_max += num_edges * max_dist
        if intro_max <= 0.0:
            intro_max = 1.0
        self.intro_max = intro_max

        # helper_max: total helper weight * max_dist
        total_helper_weight = 0.0
        for p in self.process_list:
            topo = p.get_topology()
            for dq in range(p.get_num_data_qubits()):
                w = topo.get_data_helper_weight(dq)
                if w:
                    total_helper_weight += w
        helper_max = total_helper_weight * max_dist
        if helper_max <= 0.0:
            helper_max = 1.0
        self.helper_max = helper_max

        # inter_max: mean distance between any two processes
        self.inter_max = max_dist if self.num_proc > 1 else 1.0

        # compact_max: each process has diameter ~max_dist
        self.compact_max = self.num_proc * max_dist if self.num_proc > 0 else 1.0

        # path_conflict_max: worst case every intermediate node is foreign
        # for every pair of data qubits in each process.
        max_intermediate = max(0.0, max_dist - 1.0)
        path_max = 0.0
        for p in self.process_list:
            d = p.get_num_data_qubits()
            num_pairs = d * (d - 1) / 2.0
            path_max += num_pairs * max_intermediate
        if path_max <= 0.0:
            path_max = 1.0
        self.path_conflict_max = path_max

    # ------------------------------------------------------------
    # Helper distances
    # ------------------------------------------------------------
    def _compute_min_dist_to_helper(self):
        if self.helper_qubits.size > 0:
            self.min_dist_to_helper = self.dist[:, self.helper_qubits].min(axis=1)
        else:
            self.min_dist_to_helper = np.zeros(self.n_qubits)

    # ------------------------------------------------------------
    # Compactness & initial process costs
    # ------------------------------------------------------------
    def _compute_compact_for_pid(self, pid):
        """
        Compactness cost for one process:
          mean pairwise distance between all its data qubits.
        """
        proc = self.pid_to_proc[pid]
        num_data = proc.get_num_data_qubits()
        if num_data <= 1:
            return 0.0

        phys_arr = np.fromiter(
            (self.proc_mapping[pid][dq] for dq in range(num_data)),
            dtype=int,
            count=num_data,
        )
        if phys_arr.size <= 1:
            return 0.0

        sub = self.dist[np.ix_(phys_arr, phys_arr)]
        total = sub.sum()
        n = phys_arr.size
        num_pairs = n * (n - 1)  # each unordered pair counted twice
        if num_pairs == 0:
            return 0.0
        return float(total / num_pairs)

    def _init_process_costs(self):
        for pid in self.pids:
            proc = self.pid_to_proc[pid]
            L = self.proc_mapping[pid]

            # intro
            intro = proc.intro_costs(L, self.dist)
            self.proc_intro_cost[pid] = intro
            self.sum_intro += intro

            # helper
            topo = proc.get_topology()
            hcost = 0.0
            for dq in range(proc.get_num_data_qubits()):
                w = topo.get_data_helper_weight(dq)
                if w == 0:
                    continue
                phys = L[dq]
                hcost += w * self.min_dist_to_helper[phys]
            self.proc_helper_cost[pid] = hcost
            self.sum_helper += hcost

            # compact
            ccost = self._compute_compact_for_pid(pid)
            self.proc_compact_cost[pid] = ccost
            self.sum_compact += ccost

    # ------------------------------------------------------------
    # Inter-process distance
    # ------------------------------------------------------------
    def _init_inter_costs(self):
        for i in range(self.num_proc):
            pid_i = self.pids[i]
            proc_i = self.pid_to_proc[pid_i]
            phys_i = np.array(
                [self.proc_mapping[pid_i][dq] for dq in range(proc_i.get_num_data_qubits())]
            )

            for j in range(i + 1, self.num_proc):
                pid_j = self.pids[j]
                proc_j = self.pid_to_proc[pid_j]
                phys_j = np.array(
                    [self.proc_mapping[pid_j][dq] for dq in range(proc_j.get_num_data_qubits())]
                )

                if phys_i.size == 0 or phys_j.size == 0:
                    val = 0.0
                else:
                    sub = self.dist[np.ix_(phys_i, phys_j)]
                    val = float(sub.mean())

                self.inter_cost_pair[(pid_i, pid_j)] = val
                self.sum_inter += val

    # ------------------------------------------------------------
    # Path-conflict (foreign qubits along shortest paths)
    # ------------------------------------------------------------
    def _compute_path_conflict_full(self):
        """
        Recompute path_conflict_sum from scratch.

        For each process pid and each pair of its data-qubit physical
        locations (u, v), we look up a precomputed shortest path.
        We count intermediate vertices whose pid != pid and pid != -1.

        Sum over all processes & pairs.
        """
        total = 0.0
        for pid in self.pids:
            proc = self.pid_to_proc[pid]
            num_data = proc.get_num_data_qubits()
            if num_data <= 1:
                continue

            phys_arr = [
                self.proc_mapping[pid][dq] for dq in range(num_data)
            ]

            for i in range(num_data):
                u = phys_arr[i]
                for j in range(i + 1, num_data):
                    v = phys_arr[j]
                    if u < v:
                        key = (u, v)
                    else:
                        key = (v, u)
                    path = self.shortest_paths.get(key, None)
                    if path is None or len(path) <= 2:
                        continue

                    # skip endpoints; count foreign occupants in between
                    for w in path[1:-1]:
                        pid_w = self.pid_at_phys[w]
                        if pid_w != -1 and pid_w != pid:
                            total += 1.0

        self.path_conflict_sum = total

    # ------------------------------------------------------------
    # Recompute helpers after a local change
    # ------------------------------------------------------------
    def _recompute_process(self, pid):
        proc = self.pid_to_proc[pid]
        L = self.proc_mapping[pid]

        # intro
        old_intro = self.proc_intro_cost[pid]
        new_intro = proc.intro_costs(L, self.dist)
        self.proc_intro_cost[pid] = new_intro
        self.sum_intro += new_intro - old_intro

        # helper
        old_helper = self.proc_helper_cost[pid]
        topo = proc.get_topology()
        hcost = 0.0
        for dq in range(proc.get_num_data_qubits()):
            w = topo.get_data_helper_weight(dq)
            if w == 0:
                continue
            phys = L[dq]
            hcost += w * self.min_dist_to_helper[phys]
        self.proc_helper_cost[pid] = hcost
        self.sum_helper += hcost - old_helper

        # compact
        old_compact = self.proc_compact_cost[pid]
        new_compact = self._compute_compact_for_pid(pid)
        self.proc_compact_cost[pid] = new_compact
        self.sum_compact += new_compact - old_compact

    def _recompute_inter_for(self, pid_x, pid_y):
        if pid_x == pid_y:
            return
        if pid_x < pid_y:
            key = (pid_x, pid_y)
        else:
            key = (pid_y, pid_x)

        old = self.inter_cost_pair[key]

        proc_x = self.pid_to_proc[pid_x]
        proc_y = self.pid_to_proc[pid_y]
        phys_x = np.array(
            [self.proc_mapping[pid_x][dq] for dq in range(proc_x.get_num_data_qubits())]
        )
        phys_y = np.array(
            [self.proc_mapping[pid_y][dq] for dq in range(proc_y.get_num_data_qubits())]
        )

        if phys_x.size == 0 or phys_y.size == 0:
            new = 0.0
        else:
            sub = self.dist[np.ix_(phys_x, phys_y)]
            new = float(sub.mean())

        self.inter_cost_pair[key] = new
        self.sum_inter += new - old

    # ------------------------------------------------------------
    # GENERAL LOCAL MOVE: reassign
    # ------------------------------------------------------------
    def reassign(self, phys_src, phys_dst):
        """
        Move the data-qubit at phys_src to phys_dst.
        If phys_dst was occupied, we perform a swap.
        If phys_dst was helper, phys_src becomes helper afterwards.
        """
        src_pid, src_dq = self.mapping[phys_src]
        dst_was_used = self.used_mask[phys_dst]

        if dst_was_used:
            # SWAP CASE
            dst_pid, dst_dq = self.mapping[phys_dst]

            # swap mapping
            self.mapping[phys_src], self.mapping[phys_dst] = \
                (dst_pid, dst_dq), (src_pid, src_dq)

            self.proc_mapping[src_pid][src_dq] = phys_dst
            self.proc_mapping[dst_pid][dst_dq] = phys_src

            affected = {src_pid, dst_pid}

        else:
            # MOVE-TO-HELPER CASE
            self.mapping[phys_dst] = (src_pid, src_dq)
            del self.mapping[phys_src]

            # update proc mapping
            self.proc_mapping[src_pid][src_dq] = phys_dst

            # update used/helper masks
            self.used_mask[phys_src] = False
            self.helper_mask[phys_src] = True
            self.used_mask[phys_dst] = True
            self.helper_mask[phys_dst] = False

            self.helper_qubits = np.where(self.helper_mask)[0]
            self._compute_min_dist_to_helper()

            affected = {src_pid}

        # recompute process costs (intro, helper, compact) for affected
        for pid in affected:
            self._recompute_process(pid)

        # recompute intercost pairs
        for pid in self.pids:
            if pid in affected:
                continue
            for pid_x in affected:
                self._recompute_inter_for(pid_x, pid)
        if len(affected) == 2:
            a, b = tuple(affected)
            self._recompute_inter_for(a, b)

        # update pid_at_phys from mapping
        self.pid_at_phys[:] = -1
        for phys, (pid, dq) in self.mapping.items():
            self.pid_at_phys[phys] = pid

        # and recompute global path-conflict
        self._compute_path_conflict_full()

    # ------------------------------------------------------------
    # cost
    # ------------------------------------------------------------
    def get_cost(self):
        """
        Return normalized total cost.
        """
        intro_norm  = self.sum_intro   / self.intro_max
        helper_norm = self.sum_helper  / self.helper_max
        inter_norm  = self.sum_inter   / self.inter_max
        compact_norm = self.sum_compact / self.compact_max
        path_norm   = self.path_conflict_sum / self.path_conflict_max

        return (
            alpha * intro_norm
            - beta  * inter_norm
            + gamma * helper_norm
            + delta * compact_norm
            + eps   * path_norm
        )



# NEW: limit how far a move/swap can go on the hardware graph
LOCAL_RADIUS = 2   # 1 = strictly nearest-neighbour, 2 = slightly more flexible

def propose_move(
    state: MappingCostState,
    move_prob: float = 0.3,
) -> Optional[Tuple[int, int, bool]]:
    """
    Propose a *local* move on the given MappingCostState.

    Returns:
        (phys_src, phys_dst, dst_was_used) or None if no move is possible.

    Semantics:
        - If dst_was_used == True: this is a SWAP between two mapped phys,
          but ONLY within the same process and within LOCAL_RADIUS.
        - If dst_was_used == False: this is MOVE-TO-HELPER: phys_src (used) -> phys_dst (helper),
          and phys_dst must be within LOCAL_RADIUS of phys_src.

    IMPORTANT:
        - We NEVER swap qubits belonging to different processes.
        - All motion is local (bounded by LOCAL_RADIUS in the hardware distance).
    """
    used_phys = list(state.mapping.keys())
    num_used = len(used_phys)

    if num_used == 0:
        return None

    helper_qubits = state.helper_qubits
    has_helpers = helper_qubits.size > 0

    # Pick a random source data qubit
    phys_src = random.choice(used_phys)
    src_pid, src_dq = state.mapping[phys_src]

    # Precompute distances from src to all phys
    d_from_src = state.dist[phys_src]

    # --------------------------------------------------------
    # Helper candidates: local helpers only
    # --------------------------------------------------------
    local_helpers = []
    if has_helpers:
        for h in helper_qubits:
            if d_from_src[h] <= LOCAL_RADIUS:
                local_helpers.append(int(h))

    # --------------------------------------------------------
    # Decide whether to attempt a move-to-helper or a swap
    # --------------------------------------------------------
    do_move = (len(local_helpers) > 0) and (random.random() < move_prob)

    if do_move:
        # ---- LOCAL MOVE TO HELPER ----
        phys_dst = random.choice(local_helpers)
        return phys_src, phys_dst, False

    # --------------------------------------------------------
    # LOCAL SWAP *within the same process* only
    # --------------------------------------------------------
    same_pid_local_targets: List[int] = []
    for phys in used_phys:
        if phys == phys_src:
            continue
        pid, dq = state.mapping[phys]
        if pid != src_pid:
            continue  # different process => forbidden
        if d_from_src[phys] <= LOCAL_RADIUS:
            same_pid_local_targets.append(phys)

    if not same_pid_local_targets:
        # No valid local swap partner; fall back to local move if possible
        if local_helpers:
            phys_dst = random.choice(local_helpers)
            return phys_src, phys_dst, False
        # No valid move at all
        return None

    # Do a local, intra-process swap
    phys_dst = random.choice(same_pid_local_targets)
    return phys_src, phys_dst, True


def random_initial_mapping(process_list: List[process],
                           n_qubits: int) -> Dict[int, tuple[int, int]]:
    """
    Randomly assign all data qubits of all processes to distinct physical qubits.
    Remaining physical qubits are helpers (implicitly).
    """
    total_data_qubits = sum(p.get_num_data_qubits() for p in process_list)
    if total_data_qubits > n_qubits:
        raise ValueError("Not enough physical qubits for all data qubits")

    phys_indices = list(range(n_qubits))
    random.shuffle(phys_indices)
    used_phys = phys_indices[:total_data_qubits]

    mapping = {}
    k = 0
    for proc in process_list:
        pid = proc.get_process_id()
        for dq in range(proc.get_num_data_qubits()):
            mapping[used_phys[k]] = (pid, dq)
            k += 1
    return mapping



def greedy_initial_mapping(process_list: List[process],
                           n_qubits: int,
                           distance: List[List[int]]) -> Dict[int, tuple[int, int]]:
    """
    Greedy, *cluster-by-process* placement.

    Strategy:
    1) Choose one seed physical qubit per process.
       - First process gets phys 0.
       - Each subsequent process gets the unused phys that is farthest
         (maximin distance) from all previously chosen seeds.
       This spreads processes across the chip.

    2) For each process, grow its own cluster around its seed:
       - For each remaining data qubit of that process, pick the unused
         physical qubit that is closest to *that process's existing cluster*
         (not to the global cluster).
       This keeps each process contiguous but avoids interleaving processes.

    Remaining physical qubits are left unused (potential helper zone).
    """
    total_data_qubits = sum(p.get_num_data_qubits() for p in process_list)
    if total_data_qubits > n_qubits:
        raise ValueError("Not enough physical qubits for all data qubits")

    # numpy distance matrix (n_qubits x n_qubits)
    dist_mat = np.asarray(distance, dtype=float)

    mapping: Dict[int, tuple[int, int]] = {}
    used_mask = np.zeros(n_qubits, dtype=bool)

    # ------------------------------------------------------------
    # Step 1: choose a "seed" phys for each process, far apart
    # ------------------------------------------------------------
    seeds: Dict[int, int] = {}
    cluster_phys: Dict[int, List[int]] = {}

    for idx, proc in enumerate(process_list):
        pid = proc.get_process_id()

        if idx == 0:
            # First process: seed at 0
            seed_phys = 0
        else:
            # Other processes: pick the unused phys that maximizes
            # distance to the set of existing seeds (maximin).
            candidates = np.where(~used_mask)[0]
            if candidates.size == 0:
                raise ValueError("No free physical qubits left for new process seed.")

            existing_seeds = np.fromiter(seeds.values(), dtype=int)
            # distances from each candidate to nearest seed
            # shape: (num_candidates, num_seeds) -> min over axis=1
            cand_dists = dist_mat[np.ix_(candidates, existing_seeds)].min(axis=1)
            # pick candidate with largest min distance to seeds
            seed_idx = int(cand_dists.argmax())
            seed_phys = int(candidates[seed_idx])

        # Assign first data qubit (dq=0) of this process to the seed
        mapping[seed_phys] = (pid, 0)
        used_mask[seed_phys] = True
        seeds[pid] = seed_phys
        cluster_phys[pid] = [seed_phys]

    # ------------------------------------------------------------
    # Step 2: grow each process's cluster around its own seed
    # ------------------------------------------------------------
    for proc in process_list:
        pid = proc.get_process_id()
        num_data = proc.get_num_data_qubits()

        # dq = 0 already placed at the seed
        for dq in range(1, num_data):
            # Current cluster for this process
            cluster = np.array(cluster_phys[pid], dtype=int)

            # For every physical qubit, compute distance to this cluster
            # (min distance to any qubit in the cluster)
            dists_to_cluster = dist_mat[:, cluster].min(axis=1)

            # Can't reuse occupied qubits
            dists_to_cluster = np.where(used_mask, float("inf"), dists_to_cluster)

            best_phys = int(dists_to_cluster.argmin())
            if not np.isfinite(dists_to_cluster[best_phys]):
                raise ValueError("No available physical site found during greedy mapping.")

            mapping[best_phys] = (pid, dq)
            used_mask[best_phys] = True
            cluster_phys[pid].append(best_phys)

    # ------------------------------------------------------------
    # Optional: visualize the initial territories
    # ------------------------------------------------------------
    plot_process_schedule_on_torino(
        coupling_edges=torino_coupling_map(),
        process_list=process_list,
        mapping=mapping,
        out_png="greedy_initial.png",
    )

    return mapping




def iteratively_find_the_best_mapping_for_data(
    process_list: List[process],
    n_qubits: int,
    n_restarts: int = 100,
    steps_per_restart: int = 500,
    move_prob: float = 0.3,
) -> Dict[int, Tuple[int, int]]:
    """
    Heuristic search for a good mapping using simulated annealing
    with multiple random restarts.

    This version uses MappingCostState to maintain the cost incrementally.
    Local moves:
      - swap two mapped physical qubits
      - move a mapped qubit to a helper qubit (changing helper zone)
    """
    global_best_mapping: Optional[Dict[int, Tuple[int, int]]] = None
    global_best_cost = float("inf")

    # Geometric cooling rate: smaller -> more aggressive cooling
    cooling_rate = 0.995  # try 0.99 if you want even more aggressive cooling

    for r in range(n_restarts):
        # 1) Greedy initial mapping (already reasonably compact)
        init_mapping = greedy_initial_mapping(
            process_list, n_qubits, hardware_distance_pair
        )

        # 2) Build incremental cost state
        state = MappingCostState(process_list, init_mapping, n_qubits, DIST_MATRIX)
        current_cost = state.get_cost()

        # Initial temperature scaled to cost magnitude
        T0 = max(1.0, abs(current_cost) * 0.1)

        for step in range(steps_per_restart):
            # Geometric cooling schedule
            T = T0 * (cooling_rate ** step) + 1e-9  # avoid T == 0

            # 3) Propose a local move
            move = propose_move(state, move_prob=move_prob)
            if move is None:
                # No valid move (should be very rare)
                break

            phys_src, phys_dst, dst_was_used = move

            old_cost = current_cost

            # 4) Apply move
            state.reassign(phys_src, phys_dst)
            new_cost = state.get_cost()
            delta = new_cost - old_cost

            # 5) Acceptance rule (simulated annealing)
            if delta <= 0:
                accept = True
            else:
                accept = math.exp(-delta / T) > random.random()

            if accept:
                current_cost = new_cost

                # Update global best
                if new_cost < global_best_cost:
                    global_best_cost = new_cost
                    # Copy the current mapping (phys -> (pid, dq))
                    global_best_mapping = dict(state.mapping)
            else:
                # 6) Reject: revert the move
                if dst_was_used:
                    # Swap again restores original
                    state.reassign(phys_src, phys_dst)
                else:
                    # Move-to-helper was src->dst; revert by dst->src
                    state.reassign(phys_dst, phys_src)
                # current_cost stays old_cost

        print(f"[Restart {r}] best so far: {global_best_cost}")

    print("Final best cost:", global_best_cost)
    if global_best_mapping is None:
        # Should not happen unless there were no valid mappings
        raise RuntimeError("No valid mapping found during annealing.")
    return global_best_mapping


def iteratively_find_the_best_mapping_for_all(process_list: List[process],
                                      n_qubits: int,
                                      n_restarts: int = 1,
                                      steps_per_restart: int = 2000
                                      ) -> Dict[int, tuple[int, int]]:
    """
    Find the best mapping for all qubits, including the helper qubits.
    Thus is used for the bechmark without helper qubit sharing, or without parallel execution.
    """
    pass



def load_ibm_api_key_from_file(filename: str) -> str:
    """
    Load the IBMQ API key from a file
    """
    with open(filename, "r") as f:
        api_key = f.read().strip()
    return api_key


APIKEY = load_ibm_api_key_from_file("apikey")

class jobManager:
    """
    This class manage the job and the job result.

    The job include:
    1. Manage the interface with IBMQ or simulator
    2. Decode the job result and distribution to process output
    """
    def __init__(self, ibmkey: str = APIKEY, use_simulator: bool = False):
        self._result_queue = queue.Queue()
        self._ibmkey = ibmkey
        self._use_simulator = use_simulator
        self._virtual_measurement_size = 1



    def execute_on_hardware(self, shots: int, measurement_size: int,measurement_to_process_map: Dict[int, int], scheduled_instructions: List[instruction]):
        """
        Send the scheduled instructions to hardware.
        Use the jobManager class to submit the job to IBMQ or simulator

        Input:
            measurement_to_process_map: Dict[int, int]
                The mapping from measurement index to process id

        Return:

            Result of the execution, which is a dictionary of all process results
            For example, if there are two processes in the batch, the result is:
            {
                process_id_1: {"result_key_1": result_value_1, ...},
                process_id_2: {"result_key_2": result_value_2, ...}
            }
        """
        self._virtual_measurement_size = measurement_size
        if self._use_simulator:
            # Wait for 2 second to simulate the job submission time
            time.sleep(7)
            raw_result = self.submit_job_to_simulator(shots, scheduled_instructions)
        else:
            raw_result = self.submit_job_to_ibmq(shots,scheduled_instructions)


        # print("Measurement to process map:", measurement_to_process_map)
        # print("Raw result from hardware/simulator:", raw_result)

        redistributed_result = self.redistribute_job_result(measurement_to_process_map, raw_result)
        return redistributed_result


    def submit_job_to_ibmq(self, shots: int,  scheduled_instructions: List[instruction]) -> Dict[str, int]:
        """
        Submit the scheduled instructions to IBMQ or simulator
        Get the raw result back
        
        Input:
            scheduled_instructions: List[instruction]
                The scheduled instruction list to be submitted
        """

        qiskit_circuit = construct_qiskit_circuit_for_hardware_instruction(self._virtual_measurement_size, scheduled_instructions)
        fake_hardware = fake_torino_backend
        initial_layout = [i for i in range(N_qubits)]  # logical i -> physical i

        transpiled = transpile(
            qiskit_circuit,
            backend= fake_hardware,
            initial_layout=initial_layout,
            optimization_level=3,
        )

        service = QiskitRuntimeService(channel="ibm_cloud",token=self._ibmkey)
        
        #backend = service.least_busy(simulator=False, operational=True)

        backend = service.backend("ibm_torino")

        # Convert to an ISA circuit and layout-mapped observables.
        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        isa_circuit = pm.run(transpiled)
        
        # run SamplerV2 on the chosen backend
        sampler = Sampler(mode=backend)
        sampler.options.default_shots = shots

        job = sampler.run([isa_circuit])
        # Ensure it’s done so metrics/timestamps are populated
        job.wait_for_final_state()

         # --- results ---
        pub = job.result()[0]  # first (and only) PUB result
        counts = pub.join_data().get_counts()

        return counts


    def redistribute_job_result(self, measurement_to_process_map: Dict[int, int], raw_result: Dict[str, int]) -> Dict[int, Dict[str, int]]:
        """
        Redistribute the raw result to process output.

        For example, if the count result from hardware is:
        {
            "00": 500,
            "01": 300,
            "10": 200
        }
        Where the first bit(Reverse order) belongs to process 1, and the second bit(Reverse order) belongs to process 2.
        Then the ouput is:
        {
            process_id_1: {"0": 700, "1": 300},
            process_id_2: {"0": 800, "1": 200}
        }

        
        When count result from hardware is:
        {
            "001": 500,
            "010": 300,
            "101": 200
        }
        Where the first two bits(Reverse order) belongs to process 1, and the third bit(Reverse order) belongs to process 2.
        Then the ouput is:
        {
            process_id_1: {"01": 500, "10": 300,"01":200},
            process_id_2: {"0": 800, "1": 200}
        }


        Input:
            raw_result: Any
                The raw result returned from IBMQ or simulator
        """
        redistributed_result = {}
        for bitstring, count in raw_result.items():
            # Reverse the bitstring to match measurement order
            bitstring = bitstring[::-1]

            """
            First, we get the distributed set of bitstrings for all processes
            """
            proc_string_map = {}
            for meas_index, proc_id in measurement_to_process_map.items():
                if proc_id not in proc_string_map:
                    proc_string_map[proc_id] = bitstring[meas_index]
                else:
                    proc_string_map[proc_id] += bitstring[meas_index]
            """
            Reverse the bitstring for each process to match the original order
            """
            for proc_id in proc_string_map:
                proc_string_map[proc_id] = proc_string_map[proc_id][::-1]
            """
            Update the count for each process
            """
            for proc_id, proc_bitstring in proc_string_map.items():
                if proc_id not in redistributed_result:
                    redistributed_result[proc_id] = {}
                if proc_bitstring not in redistributed_result[proc_id]:
                    redistributed_result[proc_id][proc_bitstring] = count
                else:
                    redistributed_result[proc_id][proc_bitstring] += count


        return redistributed_result



    def submit_job_to_simulator(self, shots, scheduled_instructions)-> Dict[str, int]:
        """
        Submit the scheduled instructions to simulator

        Input:
            scheduled_instructions: List[instruction]
                The scheduled instruction list to be submitted
        """
        sim = AerSimulator()
        qiskit_circuit = construct_qiskit_circuit_for_hardware_instruction(self._virtual_measurement_size, scheduled_instructions)
        
        #qiskit_circuit.draw("mpl").savefig("simulated_circuit.png")
        
        tqc = transpile(qiskit_circuit, sim)
        # Run with 1000 shots
        result = sim.run(tqc, shots=shots).result()

        counts = result.get_counts(tqc)


        # print("result:", counts)
        return counts


class haloScheduler:
    """
    The scheduler in our quantum operating system.

    It should maintain a process Queue from the user
    """
    def __init__(self, use_simulator: bool = False):
        self._process_queue = []


        # the stop event
        self._start_time=0
        self._stop_event = threading.Event()
        self._scheduler_thread: Optional[threading.Thread] = None   
        # Log of the history
        self._log=[]
        #Store the process waiting time
        #The process fidelity
        #The total running time
        #The finished process count
        self._process_source_id={}
        self._process_start_time={}
        self._process_end_time={}
        self._process_waiting_time={}
        self._process_fidelity={}
        self._total_running_time=0.0
        self._finished_process_count=0
        self._jobmanager=jobManager(use_simulator=use_simulator)



    def store_log(self, filename: str):
        """
        Store the log to a file
        """
        with open(filename, 'w') as f:
            f.write("----------------------Log:---------------------------------\n")
            for entry in self._log:
                f.write(entry + '\n')
            """
            Also store the statistics in the same file:
            """
            f.write("------------------Statistics:---------------------------------")
            f.write(f"Total running time: {self._total_running_time}\n")
            f.write(f"Finished process count: {self._finished_process_count}\n")
            f.write("Average waiting time per process:\n")
            if self._finished_process_count > 0:
                avg_waiting_time = self._total_running_time / self._finished_process_count
                f.write(f"{avg_waiting_time}\n")
            else:
                f.write("N/A\n")
            f.write("Throughput: \n")
            if self._total_running_time > 0:
                throughput = self._finished_process_count / self._total_running_time
                f.write(f"{throughput}\n")
            else:
                f.write("N/A\n")
            f.write("----------------------Process waiting times-----------------------:\n")

            for process_id, waiting_time in self._process_waiting_time.items():
                f.write(f"Process {process_id}: {waiting_time}\n")

            f.write("----------------------Process fidelities-----------------------:\n")
            for process_id, fidelity in self._process_fidelity.items():
                f.write(f"Process {process_id}: {fidelity}\n")
            average_fidelity = sum(self._process_fidelity.values()) / len(self._process_fidelity) if self._process_fidelity else 0.0    
            f.write(f"Average fidelity across all processes: {average_fidelity}\n")


    def start(self):
        """
        Start the scheduling thread
        """
        if Scheduling_Option == SchedulingOptions.BASELINE_SEQUENTIAL:
            self._scheduler_thread = threading.Thread(target=self.base_line_sequential_scheduling, daemon=True)
        elif Scheduling_Option == SchedulingOptions.NO_SHARING:
            self._scheduler_thread = threading.Thread(target=self.halo_scheduling, daemon=True)
        elif Scheduling_Option == SchedulingOptions.HALO:
            self._scheduler_thread = threading.Thread(target=self.halo_scheduling, daemon=True)
        elif Scheduling_Option == SchedulingOptions.SHOT_UNAWARE:
            assert False, "Shot unaware scheduling not implemented yet."

        self._scheduler_thread.start()
    

    def stop(self):
        """
        Stop the scheduling thread
        """
        self._stop_event.set()
        if self._scheduler_thread is not None:
            self._scheduler_thread.join()
        self._log.append(f"[STOP] Scheduling stopped at time {time.time()-self._start_time}.")


    def add_process(self, process, source_id: int = -1):
        """
        Add another process in the currecnt queue
        The source id is used to identify the circuit in the benchmark suit
        """
        self._process_queue.append(process)
        self._process_start_time[process.get_process_id()] = time.time()-self._start_time
        self._process_source_id[process.get_process_id()] = source_id
        self._log.append(f"[ADD] Process {process.get_process_id()} added to the queue at time {self._process_start_time[process.get_process_id()] }, shots: {process.get_remaining_shots()}.")



    def get_next_batch_baseline(self)-> Optional[Tuple[int, List[process]]]:
        """
        Get the next process batch for baseline scheduling.
        Just return the next process in the queue.
        Return: List[process]
        Just need to make sure the total data qubits in the batch is less than N_qubits
        """
        if len(self._process_queue) == 0:
            return None
        process = self._process_queue[0]
        if process.get_num_data_qubits() > int(N_qubits * data_hardware_ratio):
            assert False, f"Process {process.get_process_id()} requires {process.get_num_data_qubits()} data qubits, which exceeds the maximum allowed {int(N_qubits * data_hardware_ratio)}."
        return process





    def get_next_batch(self, force_return: bool = False) -> Optional[Tuple[int, List[process]]]:
        """
        Get the next process batch.
        Return: List[process]
        Just need to make sure the total data qubits in the batch is less than N_qubits

        1. Try to fill up the data qubit Zone as much as possible
        2. If the force_return is True, return the batch even when the total data qubit usage is less than 50%
        3. If the total data qubit usage is less than 50%, return None

        """
        if len(self._process_queue) == 0:
            return None
        total_utility=0.0
        used_qubits=0
        remain_qubits=int(N_qubits*data_hardware_ratio)
        process_batch=[]
        for proc in self._process_queue:
            if proc.get_num_data_qubits()<=remain_qubits:
                process_batch.append(proc)
                used_qubits+=proc.get_num_data_qubits()
                remain_qubits-=proc.get_num_data_qubits()

        total_utility=used_qubits/(N_qubits*data_hardware_ratio)
        if total_utility<=0.5 and not force_return:
            return None
        min_shots= min([proc.get_remaining_shots() for proc in process_batch])  
        self._log.append(f"[BATCH] Process batch with {[proc.get_process_id() for proc in process_batch]} selected with min shots {min_shots}.")
        return min_shots,process_batch
    


    def allocate_data_qubit(self, process_batch: List[process]) -> Dict[int, tuple[int, int]]:
        """
        Allocate data qubit territory for all processes in the batch

        Return a Dict[int, tuple[int, int]] represent the data qubit mapping
        """
        best_mapping=iteratively_find_the_best_mapping_for_data(process_batch,N_qubits)
        return best_mapping
    

    def allocate_all_qubits(self, process_batch: List[process]) -> Dict[int, tuple[int, int]]:
        """
        Allocate all qubits (data + helper) for all processes in the batch

        Return a Dict[int, tuple[int, int]] represent the data qubit mapping
        """
        best_mapping=iteratively_find_the_best_mapping_for_all(process_batch,N_qubits)
        return best_mapping




    def sequential_scheduling(self, L: Dict[int, tuple[int, int]], process_batch: List[process])-> Tuple[Dict[int, int], List[instruction]]:
        """
        Sequentially schedule instructions for all processes in the batch.
        We just run one process after another.
        Thus is helpful for debugging and baseline comparison.
        Return:
            scheduled_instructions: List[instruction]
                The scheduled instruction list
            Also, all instructions in the process are updated with helper qubit qubit assignment
        """
        final_scheduled_instructions=[]
        measurement_to_process_map = {}
        for proc in process_batch:

            scheduled_insts, is_finished, meas_to_proc_map = proc.schedule_all_instructions(L)
            final_scheduled_instructions.extend(scheduled_insts)
            measurement_to_process_map.update(meas_to_proc_map)
        return measurement_to_process_map, final_scheduled_instructions



    def scheduling_without_sharing(self, L: Dict[int, tuple[int, int]], process_batch: List[process])-> Tuple[Dict[int, int], List[instruction]]:
        """
        Not sharing helper qubits across processes
        Here L stands for the mapping of all qubits, including helper qubits.
        """
        pass




    def dynamic_helper_scheduling(self, L: Dict[int, tuple[int, int]], process_batch: List[process])-> Tuple[int, Tuple[Dict[int, int], List[instruction]]]:
        """
        Dynamically assigne helper qubits and schedule instructions
        Return:
            total_measurement_size: int
                The total measurement size across all processes

            scheduled_instructions: List[instruction]
                The scheduled instruction list
            Also, all instructions in the process are updated with helper qubit qubit assignment
        """
        final_scheduled_instructions=[]
        num_finished_process=0 
        process_finish_map = {i: False for i in process_batch}



        """
        Reform the mapping L to per process form
        """
        process_data_qubit_map = {proc.get_process_id(): {} for proc in process_batch}
        for phy, (proc_id, data_qubit) in L.items():
            process_data_qubit_map[proc_id][data_qubit] = phy


        current_measurement_index=0
        measurement_to_process_map = {}


        #All the remaining qubits are helper qubits
        #The helper qubit can be taken, but will also be released after used.
        #TODO: Not only assign the nearest helper qubits, but also bound the maximum distance between data qubit and helper qubit
        helper_qubit_zone = {phys for phys in range(N_qubits)} - set(L.keys())
        helper_qubit_available = {phys: True for phys in helper_qubit_zone}


        #Store the current helper qubit assignment for each process:
        #For example, {1:{"s0":3, "s1":5}, 2:{"s0":10, "s1":12}}
        all_proc_id = {proc.get_process_id() for proc in process_batch}
        current_helper_qubit_map = {proc_id: {} for proc_id in all_proc_id}
        num_available_helper_qubits = len(helper_qubit_zone)

        while num_finished_process < len(process_batch):
            """
            Round robin style instruction scheduling.

            Take turn to each process in the batch to schedule its next instruction.
            Assign helper qubits
            """
            for proc in process_batch:
                if process_finish_map[proc]:
                    continue
                current_proc_id = proc.get_process_id()
                """
                If the process is already finished, update the finish map
                Release all data qubits
                Put these qubits back to the helper qubit pool
                """
                if proc.is_finished():
                    process_finish_map[proc] = True
                    num_finished_process += 1
                    all_data_qubit_addresses = [x for x in L.keys() if L[x][0] == current_proc_id]
                    #Release all helper qubits assigned to this process
                    #Add reset instruction for each released helper qubit
                    for phy in all_data_qubit_addresses:
                        if phy in current_helper_qubit_map[current_proc_id]:
                            helper_qubit_available[phy] = True
                            num_available_helper_qubits += 1                            
                            final_scheduled_instructions.append(instruction(Instype.RESET, qubitaddress=[], reset_address=phy))
                    continue

                """
                No matter what is the next instruction, we need to update the scheduled mapped address for all data qubits
                """
                next_inst = proc.get_next_instruction()
                all_data_qubit_addresses = next_inst.get_all_data_qubit_addresses()
                all_helper_qubit_addresses = next_inst.get_all_helper_qubit_addresses()
                for d_addr in all_data_qubit_addresses:
                    d_index=int(d_addr[1:])
                    next_inst.set_scheduled_mapped_address(d_addr, process_data_qubit_map[current_proc_id][d_index])
                for h_addr in all_helper_qubit_addresses:
                    if h_addr in current_helper_qubit_map[current_proc_id]:
                        next_inst.set_scheduled_mapped_address(h_addr, current_helper_qubit_map[current_proc_id][h_addr])


                """
                Get the next instruction from the current process
                We have several cases:
                1. The instruction is a measurement, we need to record the measurement to process mapping
                2. The instruction is a release helper qubit instruction, we need to release the helper qubits
                3. The instruction needs no helper qubit at all, just schedule it directly
                4. The instruction needs helper qubits, there are also enough helper qubits available, assign them and schedule the instruction
                5. The instruction needs helper qubits, but there are not enough helper qubits available, skip this process this round
                """    


                """
                Case1: Measurement instruction
                We need to record the measurement to process mapping
                For example, measurement address 0 -> process id 1, measurement address 1 -> process id 2
                The mapping is:
                measurement_to_process_map:  {0: 1, 1: 2}

                We need to initialize a new measurement instruction
                """
                if next_inst.is_measurement():
                    next_inst.set_scheduled_classical_address(current_measurement_index)
                    final_scheduled_instructions.append(next_inst)
                    measurement_to_process_map[current_measurement_index] = current_proc_id
                    current_measurement_index += 1
                    proc.execute_next_instruction()
                    continue


                """
                Case2: A deallocating helper qubit instruction
                Add a reset instruction to release the helper qubit!
                """
                if next_inst.is_release_helper_qubit():
                    helper_qubit_address = next_inst.get_all_helper_qubit_addresses()
                    for h_addr in helper_qubit_address:
                        if h_addr in current_helper_qubit_map[current_proc_id]:
                            phys = current_helper_qubit_map[current_proc_id][h_addr]
                            helper_qubit_available[phys] = True
                            helper_qubit_zone.add(phys)
                            num_available_helper_qubits += 1
                            del current_helper_qubit_map[current_proc_id][h_addr]
                            final_scheduled_instructions.append(instruction(Instype.RESET, qubitaddress=[], reset_address=phys))
                    proc.execute_next_instruction()
                    continue



                helper_qubit_needed=next_inst.get_helper_qubit_count()
                """
                Case3: No helper qubit needed. all qubits are data qubits
                We should update the scheduled mapped address for all data qubits
                """
                if helper_qubit_needed==0:
                    final_scheduled_instructions.append(next_inst)
                    proc.execute_next_instruction()
                    continue



                all_helper_qubit=next_inst.get_all_helper_qubit_addresses()

                """
                Case3.5: Helper qubit needed but already assigned.
                But need to update the instruction with the current mapping
                """
                for h_addr in all_helper_qubit:
                    if h_addr not in current_helper_qubit_map[current_proc_id]:
                        break
                    else:
                        all_helper_qubit.remove(h_addr)
                        helper_qubit_needed-=1

                if helper_qubit_needed==0:
                    for h_addr in all_helper_qubit:
                        next_inst.set_scheduled_mapped_address(current_helper_qubit_map[current_proc_id][h_addr])    
                    final_scheduled_instructions.append(next_inst)
                    proc.execute_next_instruction()
                    continue


                """
                Case4: Helper qubit needed and available.
                Find the nearest available helper qubits.

                Two subcases here:
                4.1: The gate has one data qubit only, and another helper qubit
                4.2: The gate has two helper qubits
                """

                availble_helper_qubit_list=[phys for phys, available in helper_qubit_available.items() if available]

                if helper_qubit_needed<= num_available_helper_qubits:
                    topology=proc.get_topology()
                    data_qubit_physical_addresses = process_data_qubit_map[current_proc_id]
                    #Assign helper qubits
                    data_qubit_addresses = next_inst.get_all_data_qubit_addresses()
                    """
                    Case 4.1: We find the nearest helper qubit for the specific data qubit
                    """
                    if helper_qubit_needed==1:
                        helper_qubit_index=int(all_helper_qubit[0][1:])


                        selected_helper_qubit = topology.best_helper_qubit_location(hardware_distance=hardware_distance_pair,
                                                                                  data_qubit_mapping=data_qubit_physical_addresses,
                                                                                  available_helper_qubits=availble_helper_qubit_list,
                                                                                  helper_qubit_index=helper_qubit_index)
                        


                        #Assign the helper qubit
                        if selected_helper_qubit is not None:
                            next_inst.set_scheduled_mapped_address(all_helper_qubit[0],selected_helper_qubit)
                            current_helper_qubit_map[current_proc_id][all_helper_qubit[0]]=selected_helper_qubit
                            final_scheduled_instructions.append(next_inst)
                            helper_qubit_available[selected_helper_qubit]=False
                            num_available_helper_qubits-=1
                            proc.execute_next_instruction()
                            continue
                        else:
                            raise ValueError("No available helper qubit found, but should be available.")


                    """
                    This is a rare case, but is not impossible

                    For example, when preparing a cat state, we need add gates on two helper qubits first,
                    before any data qubit is involved.

                    Case 4.2: We find the nearest set of helper qubits to the entire process data qubit territory
                    """
                    if len(data_qubit_addresses)==0 and helper_qubit_needed==2:
                        helper_qubit_index1=int(all_helper_qubit[0][1:])
                        selected_helper_qubit_1 = topology.best_helper_qubit_location(hardware_distance=hardware_distance_pair,
                                                                                  data_qubit_physical_addresses=data_qubit_physical_addresses,
                                                                                  available_helper_qubits=availble_helper_qubit_list,
                                                                                  helper_qubit_index=helper_qubit_index1)
                        current_helper_qubit_map[current_proc_id][all_helper_qubit[0]]=selected_helper_qubit_1
                        next_inst.set_scheduled_mapped_address(all_helper_qubit[0],selected_helper_qubit_1)

                        helper_qubit_available[selected_helper_qubit_1]=False
                        num_available_helper_qubits-=1
                        availble_helper_qubit_list.remove(selected_helper_qubit_1)


                        helper_qubit_index2=int(all_helper_qubit[1][1:])
                        selected_helper_qubit_2 = topology.best_helper_qubit_location(hardware_distance=hardware_distance_pair,
                                                                                  data_qubit_physical_addresses=data_qubit_physical_addresses,
                                                                                  available_helper_qubits=availble_helper_qubit_list,
                                                                                  helper_qubit_index=helper_qubit_index2)
                        



                        current_helper_qubit_map[current_proc_id][all_helper_qubit[1]]=selected_helper_qubit_2
                        next_inst.set_scheduled_mapped_address(all_helper_qubit[1],selected_helper_qubit_2)


                        helper_qubit_available[selected_helper_qubit_2]=False
                        num_available_helper_qubits-=1


                        final_scheduled_instructions.append(next_inst)
                        proc.execute_next_instruction()
                        continue


                """
                Case4: Helper qubit needed and not available. The process has to wait
                """
                proc.set_status(ProcessStatus.WAIT_FOR_HELPER)


                # Release helper qubits

        if current_measurement_index==0:
            for proc in process_batch:
                print(f"[ERROR] Process {proc.get_process_id()} has instructions but no measurement scheduled.")
            for proc in process_batch:
                print("Instructions:")
                for inst in final_scheduled_instructions:
                    print(inst)
            raise ValueError("No measurement scheduled in this batch, something is wrong.")
        return current_measurement_index, measurement_to_process_map, final_scheduled_instructions



    def show_queue(self, add_to_log: bool = True):
        """
        Show the current process queue. 

        For example, process 1: num_data_qubits=3, num_helper_qubits=2, ramaining_shots=100,  process 2: num_data_qubits=3, num_helper_qubits=2, ramaining_shots=500

        Then print:
            [Process Queue] P1: DQ=3, HQ=2, Shots=100----P2: DQ=3, HQ=2, Shots=500
        """
        temp_list = []
        tmp_str = "[PROCESS QUEUE] "
        for proc in self._process_queue:
            temp_list.append(proc)
            tmp_str += f"P{proc.get_process_id()}: DQ={proc.get_num_data_qubits()}, HQ={proc.get_num_helper_qubits()}, Shots={proc._remaining_shots}----"

        if add_to_log:
            self._log.append(f"[QUEUE STATUS] {tmp_str}")


    def base_line_sequential_scheduling(self):
        """
        The baseline scheduling algorithm.
        1) Update the process queue
        2) Get the next process, allocate qubit for that single process
        3) Schedule all instructions for that single process
        4) Send the scheduled instructions to hardware
        5) Update the process queue after one process execution
        """
        start_time=time.time()
        self._start_time=start_time
        while not self._stop_event.is_set() or not len(self._process_queue)==0:


            # Step 1: Get the next batch of processes
            if len(self._process_queue) == 0:
                continue

            next_proc=self._process_queue[0]

            shots = next_proc.get_remaining_shots()

            if shots ==0:
                continue


            # Step 2: Allocate data qubit territory for all processes
            L=self.allocate_data_qubit([next_proc])

            # Step 2.5: Update the data qubit mapping in each process
            # for proc in process_batch:
            #     proc.update_data_qubit_mapping(L)


            # Step 3: Dynamically assign helper qubits and schedule instructions
            total_measurements,measurement_to_process_map, scheduled_instructions = self.dynamic_helper_scheduling(L,[next_proc])

            # Step 4: Send the scheduled instructions to hardware
            result=self._jobmanager.execute_on_hardware(shots,total_measurements,measurement_to_process_map,scheduled_instructions)


            self._log.append(f"[HARDWARE RESULT] {result}")
            # Step 5: Update the process queue after one batch execution
            self.update_process_queue(shots,result)


            #Clear the queue
            #self._process_queue.remove(next_proc)


            print("[FINISH] Finish current process.")

            self.show_queue(add_to_log=True)
        end_time=time.time()
        self._total_running_time=end_time-start_time






    def halo_scheduling(self):
        """
        The main scheduling algorithm. There are multiple steps:
        1) Get the next batch of processes
        2) Allocate data qubit territory for all processes
        3) Dynamically assign helper qubits and schedule instructions
        4) Send the scheduled instructions to hardware
        5) Update the process queue after one batch execution
        6) Repeat until the process queue is empty
        """
        start_time=time.time()
        self._start_time=start_time

        """
        Keep track of how long the scheduler has been spinning without doing any work
        We need to make sure the scheduler can force return a batch after a long spinning time
        """
        previous_spinning_start_time=0
        is_spinning=False
        max_spinning_time=1  # Set a default max spinning time (in seconds)
        while not self._stop_event.is_set() or not len(self._process_queue)==0:


            # Step 1: Get the next batch of processes
            # If the next_batch doesn't use enough qubits, wait for the next round
            #print("Starting to get the next batch...")
            if is_spinning:
                spinning_time=time.time()-previous_spinning_start_time
                if spinning_time>=max_spinning_time:
                    batchresult=self.get_next_batch(force_return=True)
            else:
                batchresult=self.get_next_batch()

            if batchresult is None:
                if not is_spinning:
                    previous_spinning_start_time=time.time()
                is_spinning=True
                continue

            shots, process_batch = batchresult

            if shots ==0:
                break

            if len(process_batch) == 0:
                if not is_spinning:
                    previous_spinning_start_time=time.time()
                is_spinning=True
                continue


            is_spinning=False


            # Step 2: Allocate data qubit territory for all processes
            L=self.allocate_data_qubit(process_batch)

            # Step 2.5: Update the data qubit mapping in each process
            # for proc in process_batch:
            #     proc.update_data_qubit_mapping(L)


            # Step 3: Dynamically assign helper qubits and schedule instructions
            print(f"[BATCH START] Scheduling batch with processes {[proc.get_process_id() for proc in process_batch]} for {shots} shots.")
            total_measurements,measurement_to_process_map, scheduled_instructions = self.dynamic_helper_scheduling(L,process_batch)
            print(f"[BATCH END] Finished scheduling batch with processes {[proc.get_process_id() for proc in process_batch]} for {shots} shots.")
            # Step 4: Send the scheduled instructions to hardware
            result=self._jobmanager.execute_on_hardware(shots,total_measurements,measurement_to_process_map,scheduled_instructions)


            self._log.append(f"[HARDWARE RESULT] {result}")
            # Step 5: Update the process queue after one batch execution
            self.update_process_queue(shots,result)


            # Clear the queue
            for proc in list(self._process_queue):  # iterate over a shallow copy
                if proc.finish_all_shots():
                    self._process_queue.remove(proc)


            # Step 6: Reset for the next batch
            for proc in process_batch:
                proc.reset_all_mapping()

            print("[FINISH] BATCHFINISH.")

            self.show_queue(add_to_log=True)
        end_time=time.time()
        self._total_running_time=end_time-start_time



    def update_process_queue(self, shots: int ,result: Dict[int, Dict[str, int]]):
        """
        Update the process queue after one batch execution.
        The result has a clear form such as:
        {
            process_id_1: {"result_key_1": result_value_1, ...},
            process_id_2: {"result_key_2": result_value_2, ...}
        }
        """


        """
        First, update the count stored in each process
        1) Find the process in the current process queue
        2) Update the process result with the result from hardware
        """
        for proc_id, proc_result in result.items():
            for proc in self._process_queue:
                if proc.get_process_id() == proc_id:
                    proc.update_result(shots, proc_result)


        """
        Second, check if the process is finished
        If finished, remove it from the process queue
        Also, calculate the waiting time and fidelity for statistics
        """
        for proc in self._process_queue:
            if proc.finish_all_shots():

                print(f"[PROCESS FINISH] Process {proc.get_process_id()} finished all shots!")
                self._process_queue.remove(proc)


                # Update process waiting time
                self._process_end_time[proc.get_process_id()] = time.time()-self._start_time
                self._process_waiting_time[proc.get_process_id()] = self._process_end_time[proc.get_process_id()] - self._process_start_time[proc.get_process_id()]


                # Update the process fidelity
                benchmark_id = self._process_source_id[proc.get_process_id()]
                ideal_result = load_ideal_count_output(benchmark_id)
                self._process_fidelity[proc.get_process_id()] = distribution_fidelity(ideal_result, proc.get_result_counts())
                print(f"[PROCESS FIDELITY] Process {proc.get_process_id()} fidelity: {self._process_fidelity[proc.get_process_id()]}")
                print(f"[PROCESS WAITING TIME] Process {proc.get_process_id()} waiting time: {self._process_waiting_time[proc.get_process_id()]} seconds")

                self._log.append(f"[PROCESS FINISH] Process {proc.get_process_id()} finished all shots at time {self._process_end_time[proc.get_process_id()]}.")
                self._log.append(f"[PROCESS FIDELITY] Process {proc.get_process_id()} fidelity: {self._process_fidelity[proc.get_process_id()]}")
                self._log.append(f"[PROCESS WAITING TIME] Process {proc.get_process_id()} waiting time: {self._process_waiting_time[proc.get_process_id()]} seconds")
                self._finished_process_count += 1




def construct_qiskit_circuit_for_hardware_instruction(num_measurements:int, instruction_list: List[instruction]) -> QuantumCircuit:
    """
    Construct a qiskit circuit from the instruction list.
    Also help to visualize the circuit.
    """
    dataqubit = QuantumRegister(N_qubits, "q")


    # Classical registers (optional, if you want measurements)
    classicalbits = ClassicalRegister(num_measurements, "c")

    # Combine them into one circuit
    qiskit_circuit = QuantumCircuit(dataqubit, classicalbits)

    for inst in instruction_list:
        if inst.is_system_call():
            continue
        addresses = inst.get_qubitaddress()
        qiskitaddress=[]
        for addr in addresses:
            phy_index = inst.get_scheduled_mapped_address(addr)
            qiskitaddress.append(dataqubit[phy_index])

        match inst.get_type():
            case Instype.H:
                qiskit_circuit.h(qiskitaddress[0])
            case Instype.X:
                qiskit_circuit.x(qiskitaddress[0]) 
            case Instype.Y:
                qiskit_circuit.y(qiskitaddress[0]) 
            case Instype.Z:
                qiskit_circuit.z(qiskitaddress[0])  
            case Instype.T:
                qiskit_circuit.t(qiskitaddress[0])
            case Instype.Tdg:
                qiskit_circuit.tdg(qiskitaddress[0])
            case Instype.S:
                qiskit_circuit.s(qiskitaddress[0])
            case Instype.Sdg:
                qiskit_circuit.sdg(qiskitaddress[0])
            case Instype.SX:
                qiskit_circuit.sx(qiskitaddress[0])
            case Instype.RZ:
                params=inst.get_params()
                qiskit_circuit.rz(params[0], qiskitaddress[0])
            case Instype.RX:
                params=inst.get_params()
                qiskit_circuit.rx(params[0], qiskitaddress[0])
            case Instype.RY:
                params=inst.get_params()
                qiskit_circuit.ry(params[0], qiskitaddress[0])
            case Instype.U3:
                params=inst.get_params()
                qiskit_circuit.u3(params[0], params[1], params[2], qiskitaddress[0])
            case Instype.U:
                params=inst.get_params()
                qiskit_circuit.u(params[0], params[1], params[2], qiskitaddress[0])
            case Instype.Toffoli:
                qiskit_circuit.ccx(qiskitaddress[0], qiskitaddress[1], qiskitaddress[2])
            case Instype.CNOT:
                qiskit_circuit.cx(qiskitaddress[0], qiskitaddress[1])
            case Instype.CH:
                qiskit_circuit.ch(qiskitaddress[0], qiskitaddress[1])
            case Instype.SWAP:
                qiskit_circuit.swap(qiskitaddress[0], qiskitaddress[1])
            case Instype.CSWAP:
                qiskit_circuit.cswap(qiskitaddress[0], qiskitaddress[1], qiskitaddress[2])
            case Instype.CP:
                params=inst.get_params()
                qiskit_circuit.cp(params[0], qiskitaddress[0], qiskitaddress[1])
            case Instype.RESET:
                qiskit_circuit.reset(dataqubit[inst.get_reset_address()])
            case Instype.MEASURE:
                scheduled_classical_address=inst.get_scheduled_classical_address()
                qiskit_circuit.measure(qiskitaddress[0], scheduled_classical_address)

    return qiskit_circuit



def random_arrival_generator(scheduler: haloScheduler,
                             arrival_rate: float = 1,
                             max_time: float = 100.0,
                             share_qubit: bool = True):
    """
    Producer thread: generate processes according to a Poisson process
    (exponential inter-arrival times with mean 1/arrival_rate).

    All processes are generated from one benchmark suit if the option is not mixed.

    TODO: Support mixed benchmark suit in the future. Just need to randomly select a benchmark suit for each arrival.
    """
    start = time.time()
    pid = 0
    benchmark_suit = benchmark_type_to_benchmark[benchmark_Option]
    benchmark_root_path = benchmark_file_path[benchmark_Option]
    while time.time() - start < max_time:
        # Wait random time until next arrival
        wait = random.expovariate(arrival_rate)  # mean 1/lambda
        time.sleep(wait)

        # Generate a random process

        #shots = random.choice(np.arange(500, 2500, 100))
        shots = 1000
        #Generate a random process from benchmark suit
        benchmark_id = random.randint(0, len(benchmark_suit) - 1)
        proc = generate_process_from_benchmark(benchmark_root_path, benchmark_suit, benchmark_id, pid, shots, share_qubit=share_qubit)
        print(f"[ARRIVAL] New process {benchmark_suit[benchmark_id]} arriving, pid: {pid}, shots: {shots}")
        scheduler.add_process(proc, source_id=benchmark_id)
        pid += 1

    print("[STOP] Finished generating processes.")
    



def generate_process_from_benchmark(benchmark_root_path:str,benchmark_suit: Dict[int,str], benchmark_id: int, pid: int, shots: int, share_qubit=True) -> process:
    """
    Generate a process from the benchmark suit, given the pid and shots
    1. Load the benchmark data from the file
    2. Create a process instance
    3. Return the process instance
    """
    if not share_qubit:
        file_path = f"{benchmark_root_path}nohelper//{benchmark_suit[benchmark_id]}"
    else:
        file_path = f"{benchmark_root_path}{benchmark_suit[benchmark_id]}"
    # Load the benchmark data from the file
    (inst_list, data_n, syn_n, measure_n)=parse_program_from_file(file_path)
    proc = process(pid, data_n, syn_n, shots, inst_list)
    return proc



def test_scheduling():
    """
    A simple test function for the haloScheduler
    """
    process1 = generate_process_from_benchmark(0, 1, 100)
    process2 = generate_process_from_benchmark(1, 2, 200)
    process3 = generate_process_from_benchmark(2, 3, 300)
    process4 = generate_process_from_benchmark(3, 4, 100)
    process5 = generate_process_from_benchmark(4, 5, 200)
    process6 = generate_process_from_benchmark(5, 6, 300)


    print("Start testing haloScheduler...")
    scheduler = haloScheduler()
    scheduler.add_process(process1)
    scheduler.add_process(process2)
    scheduler.add_process(process3)
    scheduler.add_process(process4)
    scheduler.add_process(process5)
    scheduler.add_process(process6)

 

    print("Start getting next batch...")
    shots, next_batch = scheduler.get_next_batch()


    
    L = scheduler.allocate_data_qubit(next_batch)


    plot_process_schedule_on_torino(
        torino_coupling_map(),
        next_batch,
        L,
        out_png="best_torino_mapping_6proc.png",
    )

    total_measurements, measurement_to_process_map, scheduled_instructions = scheduler.dynamic_helper_scheduling(L, next_batch)


    print("Instructions:")
    for inst in scheduled_instructions:
        print(inst)


    qiskit_circuit = construct_qiskit_circuit_for_hardware_instruction(total_measurements, scheduled_instructions)



    



# if __name__ == "__main__":
#     test_scheduling()








if __name__ == "__main__":

    random.seed(42)


    haloScheduler_instance=haloScheduler(use_simulator=False)
    haloScheduler_instance.start()


    producer_thread = threading.Thread(
        target=random_arrival_generator,
        args=(haloScheduler_instance, 0.8, 50.0, True),
        daemon=False
    )
    producer_thread.start()


    simulation_time = 80  # seconds
    time.sleep(simulation_time)


    # Wait for producer to finish generating all processes
    producer_thread.join()


    haloScheduler_instance.stop()
    print("Simulation finished.")


    haloScheduler_instance.store_log("halo_scheduler_same_shot_80.txt")

