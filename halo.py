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


benchmark_suit_file_root_no_helper="benchmarkdata//"

benchmark_suit_file_root="benchmark//"
benchmark_suit={
    0:"cat_state_prep_n4",
    1:"cat_state_verification_n4",
    2:"repetition_code_distance3_n3",
    3:"shor_parity_measurement_n4",
    4:"shor_stabilizer_XZZX_n3",
    5:"shor_stabilizer_ZZZZ_n4",
    6:"syndrome_extraction_surface_n4"
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




def load_ideal_count_output(benchmark_id: int) -> Dict[str, int]:
    """
    Load the ideal count output from the benchmark suit file
    """
    filename = benchmark_suit_file_root + "result2000shots//"+ benchmark_suit[benchmark_id] + "_counts.pkl"
    with open(filename, 'rb') as f:
        ideal_counts = pickle.load(f)
    return ideal_counts




"""
Hyper parameters for mapping cost calculation:
alpha: weight for intra-process cost
beta: weight for inter-process cost
gamma: weight for helper qubit cost
delta: weight for compact cost
"""
alpha=0.3
beta=100
gamma=300
delta=200

#Set the scheduling option here
Scheduling_Option=SchedulingOptions.HALO

# N_qubits=133
# hardware_distance_pair=all_pairs_distances(N_qubits, torino_coupling_map())
fake_torino_backend=construct_fake_ibm_torino()


N_qubits=30
hardware_distance_pair=all_pairs_distances(N_qubits, simple_30_qubit_coupling_map())
fake_backend=construct_30_qubit_hardware()


data_hardware_ratio=0.8
#The maximum distance between a data qubit and a helper qubit
max_data_helper_distance=10

# N_qubits=10
# hardware_distance_pair=all_pairs_distances(N_qubits, simple_10_qubit_coupling_map())

def calculate_mapping_cost(process_list: List[process],mapping: Dict[int, tuple[int, int]]) -> float:
    """
    Input:
        mapping: Dict[int, tuple[int, int]]
            A dictionary where the key is physical, and the value is a tuple (process_id, data_qubit_index)
        
        Output:
            A float representing the total mapping cost, calculated as:
            cost = alpha * intro_cost - beta * inter_cost + gamma * helper_cost + delta * compact_cost
    """

    # Initialize the helper qubit Zone
    is_helper_qubit={phys:True for phys in range(N_qubits)}
    for phys in mapping.keys():
        is_helper_qubit[phys]=False
    helper_qubit_list=[phys for phys in range(N_qubits) if is_helper_qubit[phys]]
    #print("Helper qubit list:", helper_qubit_list)

    #Convert the mapping L to the mapping per process
    proc_mapping={pid:{} for pid in [proc.get_process_id() for proc in process_list]}
    for phys,(pid,data_qubit) in mapping.items():
        proc_mapping[pid][data_qubit]=phys


    intro_cost=0.0
    #First step, calculate the intra cost of all process
    for proc in process_list:
        intro_cost+=proc.intro_costs(proc_mapping[proc.get_process_id()],hardware_distance_pair)


    #print("Intro cost:", intro_cost)


    compact_cost=0.0
    #Second step, calculate the compact cost of all process
    #This is defined as the distance between the furthest data qubits of a process
    for proc in process_list:
        max_distance=0.0
        for data_qubit_i in range(proc.get_num_data_qubits()):
            phys_i=proc_mapping[proc.get_process_id()][data_qubit_i]
            for data_qubit_j in range(data_qubit_i+1,proc.get_num_data_qubits()):
                phys_j=proc_mapping[proc.get_process_id()][data_qubit_j]
                max_distance=max(max_distance,hardware_distance_pair[phys_i][phys_j])
        compact_cost+=max_distance



    inter_cost=0.0
    #First step, calculate the inter cost across all processes
    #This is done by calculatin the average distance between all mapped data qubits of two processes 
    for i in range(len(process_list)):
        for j in range(i+1,len(process_list)):
            proc_i=process_list[i]
            proc_j=process_list[j]
            pid_i=proc_i.get_process_id()
            pid_j=proc_j.get_process_id()
            total_distance=0.0
            count=0
            for data_qubit_i in range(proc_i.get_num_data_qubits()):
                phys_i=proc_mapping[pid_i][data_qubit_i]
                for data_qubit_j in range(proc_j.get_num_data_qubits()):
                    phys_j=proc_mapping[pid_j][data_qubit_j]
                    total_distance+=hardware_distance_pair[phys_i][phys_j]
                    count+=1
            if count>0:
                inter_cost+=total_distance/count


    #print("Inter cost:", inter_cost)


    helper_cost=0.0
    #Last step, calculate the helper cost of all process
    #This is done by calculating the weighted distance from data qubits to unmapped helper qubit Zone
    for proc in process_list:
        for data_qubit in range(proc.get_num_data_qubits()):
            phys=proc_mapping[proc.get_process_id()][data_qubit]
            helper_weight=proc.get_topology().get_data_helper_weight(data_qubit)
            if helper_weight==0:
                continue
            min_helper_distance=10000
            for helper_qubit in helper_qubit_list:
                min_helper_distance=min(min_helper_distance,hardware_distance_pair[phys][helper_qubit])
            helper_cost+=min_helper_distance*helper_weight

    #print("Helper cost:", helper_cost)

    return alpha * intro_cost - beta * inter_cost + gamma * helper_cost + delta * compact_cost




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
    Greedy placement:
    - Place the very first data qubit of the first process on phys 0.
    - For every next data qubit across all processes:
         choose the unused physical qubit
         that is closest to ANY already-used physical qubit.
    Produces a compact cluster-like initial layout.
    """
    total_data_qubits = sum(p.get_num_data_qubits() for p in process_list)
    if total_data_qubits > n_qubits:
        raise ValueError("Not enough physical qubits for all data qubits")

    mapping: Dict[int, tuple[int, int]] = {}

    # --- Step 1: place the very first data qubit onto physical qubit 0 ---
    mapping[0] = (process_list[0].get_process_id(), 0)

    used_phys = {0}
    remaining_phys = set(range(n_qubits)) - used_phys

    # Iterator for (pid, data_qubit)
    placement_list = []
    for proc in process_list:
        pid = proc.get_process_id()
        for dq in range(proc.get_num_data_qubits()):
            placement_list.append((pid, dq))

    # We already placed first one, so skip it
    placement_list = placement_list[1:]

    # --- Step 2: greedy expansion ---
    for pid, dq in placement_list:
        best_phys = None
        best_score = float("inf")

        for phys in remaining_phys:
            # distance to the closest used phys (cluster expansion)
            dist_to_cluster = min(distance[phys][u] for u in used_phys)
            if dist_to_cluster < best_score:
                best_score = dist_to_cluster
                best_phys = phys

        # Assign the chosen physical location
        mapping[best_phys] = (pid, dq)

        # Update sets
        used_phys.add(best_phys)
        remaining_phys.remove(best_phys)

    return mapping

def propose_neighbor(mapping: Dict[int, tuple[int, int]],
                     n_qubits: int,
                     move_prob: float = 0.3) -> Dict[int, tuple[int, int]]:
    """
    Given a mapping, return a new mapping by either:
    - Swapping two mapped physical qubits, or
    - Moving a data qubit to a helper location.
    """
    new_mapping = dict(mapping)  # shallow copy is enough

    used_phys = list(new_mapping.keys())
    all_phys = list(range(n_qubits))
    helper_phys = [p for p in all_phys if p not in used_phys]

    # If we have no helper qubits, we can only swap.
    if not helper_phys or random.random() > move_prob:
        # swap two mapped physical locations
        if len(used_phys) < 2:
            return new_mapping
        a, b = random.sample(used_phys, 2)
        new_mapping[a], new_mapping[b] = new_mapping[b], new_mapping[a]
    else:
        # move one data qubit to a helper physical qubit
        a = random.choice(used_phys)
        h = random.choice(helper_phys)
        new_mapping[h] = new_mapping[a]
        del new_mapping[a]

    return new_mapping



def iteratively_find_the_best_mapping_for_data(process_list: List[process],
                                      n_qubits: int,
                                      n_restarts: int = 5,
                                      steps_per_restart: int = 2000
                                      ) -> Dict[int, tuple[int, int]]:
    """
    Heuristic search for a good mapping using simulated annealing
    with multiple random restarts.

    Returns the best mapping found.
    """
    global_best_mapping = None
    global_best_cost = float("inf")

    for r in range(n_restarts):
        # 1) random initial mapping
        # current_mapping = random_initial_mapping(process_list, n_qubits)
        current_mapping = greedy_initial_mapping(process_list, n_qubits,hardware_distance_pair)
        current_cost = calculate_mapping_cost(process_list, current_mapping)

        # temperature schedule (very simple linear cooling)
        # scale the initial T with magnitude of the cost to get something reasonable
        T0 = max(1.0, abs(current_cost) * 0.1)

        for step in range(steps_per_restart):
            # temperature decreases over time
            t = step / max(1, steps_per_restart - 1)
            T = T0 * (1.0 - t) + 1e-3  # from T0 -> ~0

            # 2) propose a neighbor and compute its cost
            candidate_mapping = propose_neighbor(current_mapping, n_qubits)
            candidate_cost = calculate_mapping_cost(process_list, candidate_mapping)

            delta = candidate_cost - current_cost

            # 3) acceptance rule (simulated annealing)
            if delta < 0 or math.exp(-delta / T) > random.random():
                current_mapping = candidate_mapping
                current_cost = candidate_cost

                # track global best
                if current_cost < global_best_cost:
                    global_best_cost = current_cost
                    global_best_mapping = current_mapping

        print(f"[Restart {r}] best so far: {global_best_cost}")

    print("Final best cost:", global_best_cost)
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
    """
    start = time.time()
    pid = 0
    while time.time() - start < max_time:
        # Wait random time until next arrival
        wait = random.expovariate(arrival_rate)  # mean 1/lambda
        time.sleep(wait)

        # Generate a random process

        shots = random.choice(np.arange(500, 2500, 100))
        #shots = 1000
        #Generate a random process from benchmark suit
        benchmark_id = random.randint(0, len(benchmark_suit) - 1)
        proc = generate_process_from_benchmark(benchmark_id, pid, shots, share_qubit=share_qubit)
        print(f"[ARRIVAL] New process {benchmark_suit[benchmark_id]} arriving, pid: {pid}, shots: {shots}")
        scheduler.add_process(proc, source_id=benchmark_id)
        pid += 1

    print("[STOP] Finished generating processes.")
    



def generate_process_from_benchmark(benchmark_id: int, pid: int, shots: int, share_qubit=True) -> process:
    """
    Generate a process from the benchmark suit, given the pid and shots
    1. Load the benchmark data from the file
    2. Create a process instance
    3. Return the process instance
    """
    if not share_qubit:
        file_path = f"{benchmark_suit_file_root_no_helper}{benchmark_suit[benchmark_id]}"
    else:
        file_path = f"{benchmark_suit_file_root}{benchmark_suit[benchmark_id]}"
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


    haloScheduler_instance=haloScheduler(use_simulator=True)
    haloScheduler_instance.start()


    producer_thread = threading.Thread(
        target=random_arrival_generator,
        args=(haloScheduler_instance, 0.2, 20.0, True),
        daemon=False
    )
    producer_thread.start()


    simulation_time = 30  # seconds
    time.sleep(simulation_time)


    # Wait for producer to finish generating all processes
    producer_thread.join()


    haloScheduler_instance.stop()
    print("Simulation finished.")


    haloScheduler_instance.store_log("halo_scheduler_log.txt")

