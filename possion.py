# Find the optimal process batch (with two strategies and experiment)

from typing import Dict, List, Tuple, Callable, Any
import time
import threading
import queue
import random

HARDWARE_SHOTS_PER_SECOND = 100.0   # hypothetical hardware speed
HARDWARE_QUBITS = 133               # hypothetical hardware qubit count
GOLDEN_RATIO = (1 + 5 ** 0.5) / 2   # target-ish ratio between data qubits and all qubits


class process:
    def __init__(self, process_id: int, num_data_qubits: int,
                 num_helper_qubits: int, shots: int) -> None:
        self._process_id = process_id
        self._num_data_qubits = num_data_qubits
        self._num_helper_qubits = num_helper_qubits
        self._shots_total = shots           # original total shots
        self._remaining_shots = shots       # remaining shots

        # For metrics
        self.arrival_time: float | None = None
        self.first_start_time: float | None = None

    def get_process_id(self) -> int:
        return self._process_id
    
    def get_num_data_qubits(self) -> int:
        return self._num_data_qubits

    def get_num_helper_qubits(self) -> int:
        return self._num_helper_qubits

    def get_shots_total(self) -> int:
        return self._shots_total

    def get_remaining_shots(self) -> int:
        return self._remaining_shots

    def consume_shots(self, n: int) -> None:
        """Decrease remaining shots by n (clamped at 0)."""
        self._remaining_shots = max(0, self._remaining_shots - n)

    def __repr__(self) -> str:
        return (f"process(id={self._process_id}, "
                f"data={self._num_data_qubits}, helper={self._num_helper_qubits}, "
                f"remaining_shots={self._remaining_shots})")


# =========================
# Batch simulation function
# =========================
def OS_simulator(batch_entries: list[tuple[process, int]],
                 batch_shots: int,
                 shots_per_second: float) -> None:
    """
    Simulate executing a *batch* of processes in parallel on the hardware.

    Each entry in `batch_entries` is (process, replicas).
    All replicas run in parallel for `batch_shots` shots.

    CRUCIAL MODELING ASSUMPTION:
        - A "shot" is a global cycle.
        - Time per shot does NOT depend on how many processes are in the batch,
          as long as qubit capacity is not exceeded.
        - So wall-clock time for this batch is:

                exec_time = batch_shots / shots_per_second

          independent of total_replicas.
    """
    if not batch_entries or batch_shots <= 0:
        return

    total_replicas = sum(rep for _, rep in batch_entries)

    # For logging / utilization (not used in timing)
    used_qubits = 0
    for p, rep in batch_entries:
        q_per_replica = p.get_num_data_qubits() + p.get_num_helper_qubits()
        used_qubits += q_per_replica * rep

    exec_time = batch_shots / shots_per_second

    print(
        f"[OS] Starting batch: {len(batch_entries)} unique procs, "
        f"{total_replicas} replicas, batch_shots={batch_shots}, "
        f"used_qubits={used_qubits}, "
        f"exec_time={exec_time:.2f}s"
    )
    for p, rep in batch_entries:
        print(
            f"     - PID {p.get_process_id()} "
            f"replicas={rep}, remaining_shots={p.get_remaining_shots()}"
        )

    time.sleep(exec_time)

    print("[OS] Finished batch.")


# =========================
# ADVANCED batch selector
# =========================

def select_batch_advanced(candidates: List[process],
                          max_qubits: int) -> Tuple[List[Tuple[process, int]], int]:
    """
    Advanced selector with possible replication.

    Steps:
      1. Choose a base set of distinct processes that fit in `max_qubits`
         (data+helper) using a greedy heuristic.
      2. Set batch_shots = min remaining_shots over that base set.
      3. If there is still qubit capacity left, replicate processes in the batch
         (add more copies of the same process) to increase throughput, making sure:
            replicas * batch_shots <= remaining_shots(process)
            and total qubits used <= max_qubits.

    Returns:
        (batch_entries, batch_shots)
        where batch_entries is a list of (process, replicas).
    """
    if not candidates:
        return [], 0

    # Sort by remaining shots (largest first) so "long" jobs get scheduled first.
    sorted_candidates = sorted(
        candidates,
        key=lambda p: p.get_remaining_shots(),
        reverse=True,
    )

    base_batch: List[process] = []
    used_qubits = 0

    # Phase 1: choose base set of distinct processes
    for p in sorted_candidates:
        p_qubits = p.get_num_data_qubits() + p.get_num_helper_qubits()
        if p_qubits <= 0:
            continue

        if used_qubits + p_qubits > max_qubits:
            continue

        used_qubits += p_qubits
        base_batch.append(p)

    if not base_batch:
        return [], 0

    # Fixed batch shots for this batch
    batch_shots = min(p.get_remaining_shots() for p in base_batch)
    if batch_shots <= 0:
        return [], 0

    # Phase 2: replication
    batch_entries: List[Tuple[process, int]] = [(p, 1) for p in base_batch]
    used_qubits_now = used_qubits
    remaining_qubits = max_qubits - used_qubits_now

    # For replication, consider processes with large remaining_shots first
    base_sorted_for_rep = sorted(
        base_batch,
        key=lambda p: p.get_remaining_shots(),
        reverse=True,
    )

    # Quick lookup for current replicas count
    replicas_map = {id(p): 1 for p in base_batch}

    for p in base_sorted_for_rep:
        if remaining_qubits <= 0:
            break

        p_qubits = p.get_num_data_qubits() + p.get_num_helper_qubits()
        if p_qubits <= 0:
            continue

        max_replicas_for_p = p.get_remaining_shots() // batch_shots
        # We already have 1 replica in the batch
        max_extra_replicas = max_replicas_for_p - 1
        if max_extra_replicas <= 0:
            continue

        # Limited by qubit capacity
        max_by_qubits = remaining_qubits // p_qubits
        extra_replicas = min(max_extra_replicas, max_by_qubits)

        if extra_replicas <= 0:
            continue

        replicas_map[id(p)] += extra_replicas
        remaining_qubits -= extra_replicas * p_qubits

    # Build final batch_entries from replicas_map
    final_batch_entries: List[Tuple[process, int]] = []
    for p in base_batch:
        rep = replicas_map[id(p)]
        if rep > 0:
            final_batch_entries.append((p, rep))

    return final_batch_entries, batch_shots


def extract_batch_from_queue_advanced(proc_queue: "queue.Queue[process]",
                                      max_qubits: int) -> Tuple[List[Tuple[process, int]], int]:
    """
    Extract a batch of processes from the queue using `select_batch_advanced`,
    with possible replication.
    """
    drained: List[process] = []

    # Drain all available processes at this moment (nonblocking)
    while True:
        try:
            p = proc_queue.get_nowait()
            drained.append(p)
        except queue.Empty:
            break

    if not drained:
        return [], 0

    batch_entries, batch_shots = select_batch_advanced(drained, max_qubits)

    # Requeue the non-selected processes
    selected_ids = {id(p) for p, _ in batch_entries}
    for p in drained:
        if id(p) not in selected_ids:
            proc_queue.put(p)

    return batch_entries, batch_shots


# =========================
# NAIVE batch selector
# =========================

def extract_batch_from_queue_naive(proc_queue: "queue.Queue[process]",
                                   max_qubits: int) -> Tuple[List[Tuple[process, int]], int]:
    """
    Naive selector:
      - Take the next available process from the queue (FIFO).
      - Run it as the *entire* batch, with no replication.
      - Run it for all its remaining shots.

    We ignore max_qubits here except for a sanity check.
    """
    try:
        p = proc_queue.get_nowait()
    except queue.Empty:
        return [], 0

    required_qubits = p.get_num_data_qubits() + p.get_num_helper_qubits()
    if required_qubits > max_qubits:
        # In a real system we would handle this; here we just print and drop.
        print(f"[NAIVE] Process {p.get_process_id()} needs {required_qubits} qubits, "
              f"exceeds hardware capacity {max_qubits}. Dropping.")
        # Not requeueing; effectively rejecting this process
        return [], 0

    batch_shots = p.get_remaining_shots()
    if batch_shots <= 0:
        # Nothing to do; drop it silently
        return [], 0

    # Entire batch is just one replica of p
    return [(p, 1)], batch_shots


# =========================
# Simulator
# =========================

class Boisson_simulator:
    """
    The simulator for user behavior.
    Processes arrive randomly according to a Poisson process
    (i.e., exponential inter-arrival times).

    The batch selection strategy is given as a function:
        batch_extractor(proc_queue, max_qubits) -> (batch_entries, batch_shots)
    """

    def __init__(self,
                 processes: Dict[int, process],
                 shots: float,
                 batch_extractor: Callable[
                     ["queue.Queue[process]", int],
                     Tuple[List[Tuple[process, int]], int]
                 ],
                 name: str = "unknown") -> None:
        # Here we treat `shots` as the Poisson rate λ (arrivals per second)
        self._process_templates = processes
        self._lambda = float(shots)
        self._batch_extractor = batch_extractor
        self._name = name

        # Stats
        self.total_arrivals = 0
        self.total_completed = 0
        self.waiting_times: List[float] = []

        self._available_qubits = HARDWARE_QUBITS

    def run(self, maximum_time: float, seed: int | None = None) -> Dict[str, Any]:
        """
        Run the Boisson simulator for at most `maximum_time` *real* seconds.

        Two threads:
            - arrival_thread: generates new processes and puts them into a queue
            - hardware_thread: repeatedly extracts a batch from the queue,
                               runs it, updates remaining shots, and requeues
                               unfinished processes.
        Returns a dict with metrics:
            - throughput
            - avg_wait_time
            - total_completed
            - total_arrivals
            - sim_time
        """
        if seed is not None:
            random.seed(seed)

        proc_queue: "queue.Queue[process]" = queue.Queue()
        stop_event = threading.Event()

        SHOTS_PER_SECOND = HARDWARE_SHOTS_PER_SECOND
        QUBITS = HARDWARE_QUBITS

        # For assigning fresh process IDs
        next_proc_id = max(self._process_templates.keys(), default=0) + 1
        next_proc_id_lock = threading.Lock()

        sim_start_time = time.time()

        def arrival_thread():
            nonlocal next_proc_id
            start_time = time.time()

            while True:
                now = time.time()
                if now - start_time >= maximum_time:
                    print(f"[ARRIVAL-{self._name}] Max simulation time reached. Stop creating new processes.")
                    break

                # Exponential inter-arrival time with rate λ
                dt = random.expovariate(self._lambda)
                time.sleep(dt)

                # Pick a random template and clone it with a fresh ID
                template = random.choice(list(self._process_templates.values()))
                with next_proc_id_lock:
                    pid = next_proc_id
                    next_proc_id += 1

                new_proc = process(
                    process_id=pid,
                    num_data_qubits=template.get_num_data_qubits(),
                    num_helper_qubits=template.get_num_helper_qubits(),
                    shots=template.get_shots_total(),
                )
                new_proc.arrival_time = time.time()

                proc_queue.put(new_proc)
                self.total_arrivals += 1
                print(f"[ARRIVAL-{self._name}] New process arrived: {new_proc}")

            # Signal that no more processes will arrive
            stop_event.set()
            print(f"[ARRIVAL-{self._name}] Arrival thread finished.")

        def hardware_thread():
            while True:
                # If no more arrivals and queue is empty -> we're done
                if stop_event.is_set() and proc_queue.empty():
                    print(f"[HW-{self._name}] No more work and arrival stopped. Hardware thread exits.")
                    break

                # Extract a batch (may be empty) using chosen strategy
                batch_entries, batch_shots = self._batch_extractor(proc_queue, QUBITS)

                if not batch_entries or batch_shots <= 0:
                    # Nothing useful to run right now
                    time.sleep(0.1)
                    continue

                # Mark first start times
                now = time.time()
                for p, _ in batch_entries:
                    if p.first_start_time is None:
                        p.first_start_time = now

                OS_simulator(batch_entries, batch_shots, SHOTS_PER_SECOND)

                # After execution: update remaining shots and decide who is done
                for p, replicas in batch_entries:
                    consumed = replicas * batch_shots
                    p.consume_shots(consumed)
                    if p.get_remaining_shots() > 0:
                        # Still has shots to run, put back into the queue
                        proc_queue.put(p)
                    else:
                        # Process completed
                        self.total_completed += 1
                        if p.arrival_time is not None and p.first_start_time is not None:
                            wait = p.first_start_time - p.arrival_time
                            self.waiting_times.append(wait)
                        print(f"[HW-{self._name}] Process {p.get_process_id()} completed.")

        # Create and start the threads
        t_arrival = threading.Thread(target=arrival_thread, daemon=True)
        t_hardware = threading.Thread(target=hardware_thread, daemon=True)

        print(f"[MAIN-{self._name}] Starting Boisson simulator.")
        t_arrival.start()
        t_hardware.start()

        # Wait for both threads to finish
        t_arrival.join()
        t_hardware.join()

        sim_end_time = time.time()
        sim_time = sim_end_time - sim_start_time

        avg_wait_time = (sum(self.waiting_times) / len(self.waiting_times)
                         if self.waiting_times else 0.0)
        throughput = (self.total_completed / sim_time) if sim_time > 0 else 0.0

        print(f"[MAIN-{self._name}] Simulation finished.")
        print(f"  Total arrivals : {self.total_arrivals}")
        print(f"  Total completed: {self.total_completed}")
        print(f"  Remaining in queue: {proc_queue.qsize()}")
        print(f"  Simulation time: {sim_time:.3f}s")
        print(f"  Throughput: {throughput:.4f} processes/s")
        print(f"  Avg wait time: {avg_wait_time:.4f}s")

        return {
            "name": self._name,
            "throughput": throughput,
            "avg_wait_time": avg_wait_time,
            "total_completed": self.total_completed,
            "total_arrivals": self.total_arrivals,
            "sim_time": sim_time,
        }


# =========================
# Experiment function
# =========================

def run_experiment(maximum_time: float = 10.0,
                   poisson_rate: float = 1.0,
                   seed: int = 42) -> Dict[str, Dict[str, Any]]:
    """
    Run the simulator twice:
      1. With advanced batch selection (with replication).
      2. With naive batch selection (single process per batch).

    Compare:
      - Throughput
      - Average wait time

    Returns a dict:
      {
        "advanced": metrics_dict,
        "naive": metrics_dict
      }
    """
    # Example templates for different kinds of processes
    templates: Dict[int, process] = {
        # --- Small-ish processes (fast jobs, good for latency) ---
        1: process(1, num_data_qubits=20, num_helper_qubits=10, shots=80),
        2: process(2, num_data_qubits=24, num_helper_qubits=12, shots=100),
        3: process(3, num_data_qubits=28, num_helper_qubits=14, shots=120),

        # --- Medium processes (main workloads) ---
        4: process(4, num_data_qubits=32, num_helper_qubits=16, shots=150),
        5: process(5, num_data_qubits=36, num_helper_qubits=18, shots=180),
        6: process(6, num_data_qubits=40, num_helper_qubits=20, shots=200),
        7: process(7, num_data_qubits=45, num_helper_qubits=22, shots=220),

        # --- Large processes (heavy jobs, exercise scheduling) ---
        8: process(8, num_data_qubits=50, num_helper_qubits=25, shots=250),
        9: process(9, num_data_qubits=55, num_helper_qubits=28, shots=300),
        10: process(10, num_data_qubits=60, num_helper_qubits=30, shots=350),
    }

    # Run advanced
    sim_adv = Boisson_simulator(
        templates,
        shots=poisson_rate,
        batch_extractor=extract_batch_from_queue_advanced,
        name="advanced",
    )
    print("\n========== Running ADVANCED strategy ==========\n")
    metrics_adv = sim_adv.run(maximum_time=maximum_time, seed=seed)

    # Run naive with the same template *fresh* (we reuse the templates dict, but
    # each run creates new process instances for arrivals, so it's fine).
    sim_naive = Boisson_simulator(
        templates,
        shots=poisson_rate,
        batch_extractor=extract_batch_from_queue_naive,
        name="naive",
    )
    print("\n========== Running NAIVE strategy ==========\n")
    metrics_naive = sim_naive.run(maximum_time=maximum_time, seed=seed)

    print("\n========== Comparison ==========")
    print(f"Throughput (adv) : {metrics_adv['throughput']:.4f} "
          f"vs (naive): {metrics_naive['throughput']:.4f}")
    print(f"Avg wait (adv)   : {metrics_adv['avg_wait_time']:.4f}s "
          f"vs (naive): {metrics_naive['avg_wait_time']:.4f}s")

    return {"advanced": metrics_adv, "naive": metrics_naive}


if __name__ == "__main__":
    # Example: run experiment comparing advanced vs naive for 10 seconds
    results = run_experiment(maximum_time=10.0, poisson_rate=1.0, seed=123)
