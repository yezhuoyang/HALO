#Study the effect of mapping cost on the fidelity with concrete examples
from halo import *
from process import plot_process_schedule_on_10_qubit_hardware

from qiskit.qasm2 import dumps

from qiskit_ibm_runtime import QiskitRuntimeService





def load_ibm_api_key_from_file(filename: str) -> str:
    """
    Load the IBMQ API key from a file
    """
    with open(filename, "r") as f:
        api_key = f.read().strip()
    return api_key


APIKEY = load_ibm_api_key_from_file("apikey")






def get_result_counts_from_job(job_id: str) -> dict:
    pass
    #    service = QiskitRuntimeService(
    #         channel='ibm_quantum_platform',
    #         instance='crn:v1:bluemix:public:quantum-computing:us-east:a/9975179a4fd749d4b3b2e7f4f445effd:22466e10-e37c-4613-b5bb-a4fcd4ea5278::',
    #         token=APIKEY 
    #     )
    #     job = service.job('d4iubgk3tdfc73dmtclg')
    #     pub = job.result()[0]  # first (and only) PUB result
        
    #     counts = pub.join_data().get_counts()


    #     redistributed_result = scheduler._jobmanager.redistribute_job_result(measurement_to_process_map, counts)



    #     print("Received Counts:", counts)
    #     scheduler.update_process_queue(shots, redistributed_result)


benchmark_root_path="C://Users//yezhu//Documents//HALO//benchmark//multiXmedium//"

multix_medium_benchmark={
    0: "mcx_9_0",
    1: "mcx_9_1",
    2: "mcx_10_0",
    3: "mcx_10_1",
    4: "mcx_11_0",
    5: "mcx_11_1",
    6: "mcx_12_0",
    7: "mcx_12_1",
    8: "mcx_13_0",
    9: "mcx_13_1",
    10: "mcx_14_0",
    11: "mcx_14_1"
}

def test_scheduling():
    """
    A simple test function for the haloScheduler
    """
    process1 = generate_process_from_benchmark(benchmark_root_path=benchmark_root_path,benchmark_suit=multix_medium_benchmark,benchmark_id=0, pid=1, shots=1000, share_qubit=True)
    process2 = generate_process_from_benchmark(benchmark_root_path=benchmark_root_path,benchmark_suit=multix_medium_benchmark,benchmark_id=1, pid=2, shots=1000, share_qubit=True)
    process3 = generate_process_from_benchmark(benchmark_root_path=benchmark_root_path,benchmark_suit=multix_medium_benchmark,benchmark_id=2, pid=3, shots=1000, share_qubit=True)
    process4 = generate_process_from_benchmark(benchmark_root_path=benchmark_root_path,benchmark_suit=multix_medium_benchmark,benchmark_id=3, pid=4, shots=1000, share_qubit=True)
    process5 = generate_process_from_benchmark(benchmark_root_path=benchmark_root_path,benchmark_suit=multix_medium_benchmark,benchmark_id=4, pid=5, shots=1000, share_qubit=True)
    process6 = generate_process_from_benchmark(benchmark_root_path=benchmark_root_path,benchmark_suit=multix_medium_benchmark,benchmark_id=5, pid=6, shots=1000, share_qubit=True)
    process7 = generate_process_from_benchmark(benchmark_root_path=benchmark_root_path,benchmark_suit=multix_medium_benchmark,benchmark_id=6, pid=7, shots=1000, share_qubit=True)
    #process8 = generate_process_from_benchmark(benchmark_root_path=benchmark_root_path,benchmark_suit=multix_medium_benchmark,benchmark_id=7, pid=8, shots=1000, share_qubit=True)


    print("Start testing haloScheduler...")
    scheduler = haloScheduler(use_simulator=False)
    scheduler.add_process(process1,source_id=0)
    scheduler.add_process(process2,source_id=1)
    scheduler.add_process(process3,source_id=2)
    scheduler.add_process(process4,source_id=3)
    scheduler.add_process(process5,source_id=4)
    scheduler.add_process(process6,source_id=5)
    scheduler.add_process(process7,source_id=6)
    #scheduler.add_process(process8,source_id=0)
 

    shots, next_batch = scheduler.get_next_batch(force_return=True)


    L = scheduler.allocate_data_qubit(next_batch)

    # L={18: (5, 1), 114: (3, 0), 14: (5, 0), 2: (1, 0), 118: (6, 1), 
    #    0: (1, 1), 113: (4, 3), 129: (3, 2), 115: (3, 1), 33: (2, 1), 
    #    109: (4, 2), 37: (2, 3), 111: (6, 2), 130: (6, 0), 51: (2, 2), 
    #    11: (5, 2), 20: (1, 3), 120: (6, 3), 127: (4, 1), 32: (2, 0), 
    #    128: (4, 0), 3: (1, 2)

    plot_process_schedule_on_torino(
        torino_coupling_map(),
        next_batch,
        L,
        out_png="first_case_real_run.png",
    )


    utility, total_measurements, measurement_to_process_map, scheduled_instructions = scheduler.dynamic_helper_scheduling(L, next_batch)

    print("Qubit Utilization:", utility)
    result=scheduler._jobmanager.execute_on_hardware(shots,total_measurements,measurement_to_process_map,scheduled_instructions)
    scheduler.update_process_queue(shots, result)


    benchmark_id=0
    for proc in next_batch:
        benchmark_id=proc.get_process_id()-1
        ideal_result = load_ideal_count_output(benchmarktype.MULTI_CONTROLLED_X_MEDIUM,benchmark_id)
        print("Ideal Counts:", ideal_result)
        print("Received Counts:", proc.get_result_counts())
        fidelity=distribution_fidelity(ideal_result, proc.get_result_counts())
        print(f"Benchmark: {benchmark_id}")
        print(f"Fidelity: {fidelity:.10f}")







def test_sequential_scheduling():
    """
    A simple test function for the haloScheduler
    """
    process1 = generate_process_from_benchmark(benchmark_root_path=benchmark_root_path,benchmark_suit=multix_medium_benchmark,benchmark_id=0, pid=1, shots=1000, share_qubit=True)
    process2 = generate_process_from_benchmark(benchmark_root_path=benchmark_root_path,benchmark_suit=multix_medium_benchmark,benchmark_id=1, pid=2, shots=1000, share_qubit=True)
    process3 = generate_process_from_benchmark(benchmark_root_path=benchmark_root_path,benchmark_suit=multix_medium_benchmark,benchmark_id=2, pid=3, shots=1000, share_qubit=True)
    process4 = generate_process_from_benchmark(benchmark_root_path=benchmark_root_path,benchmark_suit=multix_medium_benchmark,benchmark_id=3, pid=4, shots=1000, share_qubit=True)
    process5 = generate_process_from_benchmark(benchmark_root_path=benchmark_root_path,benchmark_suit=multix_medium_benchmark,benchmark_id=4, pid=5, shots=1000, share_qubit=True)
    process6 = generate_process_from_benchmark(benchmark_root_path=benchmark_root_path,benchmark_suit=multix_medium_benchmark,benchmark_id=5, pid=6, shots=1000, share_qubit=True)
    process7 = generate_process_from_benchmark(benchmark_root_path=benchmark_root_path,benchmark_suit=multix_medium_benchmark,benchmark_id=6, pid=7, shots=1000, share_qubit=True)
    #process8 = generate_process_from_benchmark(benchmark_root_path=benchmark_root_path,benchmark_suit=multix_medium_benchmark,benchmark_id=7, pid=8, shots=1000, share_qubit=True)


    print("Start testing haloScheduler...")
    scheduler = haloScheduler(use_simulator=True)
    scheduler.add_process(process1,source_id=0)
    scheduler.add_process(process2,source_id=1)
    scheduler.add_process(process3,source_id=2)
    scheduler.add_process(process4,source_id=3)
    scheduler.add_process(process5,source_id=4)
    scheduler.add_process(process6,source_id=5)
    scheduler.add_process(process7,source_id=6)
    #scheduler.add_process(process8,source_id=0)
 
    for proc in [process1, process2, process3, process4, process5, process6, process7]:

        shots, next_batch = proc.get_remaining_shots(),[proc]


        L = scheduler.allocate_data_qubit(next_batch)

        # L={18: (5, 1), 114: (3, 0), 14: (5, 0), 2: (1, 0), 118: (6, 1), 
        #    0: (1, 1), 113: (4, 3), 129: (3, 2), 115: (3, 1), 33: (2, 1), 
        #    109: (4, 2), 37: (2, 3), 111: (6, 2), 130: (6, 0), 51: (2, 2), 
        #    11: (5, 2), 20: (1, 3), 120: (6, 3), 127: (4, 1), 32: (2, 0), 
        #    128: (4, 0), 3: (1, 2)

        plot_process_schedule_on_torino(
            torino_coupling_map(),
            next_batch,
            L,
            out_png="first_case_real_run.png",
        )


        total_measurements, measurement_to_process_map, scheduled_instructions = scheduler.dynamic_helper_scheduling(L, next_batch)

        
        result=scheduler._jobmanager.execute_on_hardware(shots,total_measurements,measurement_to_process_map,scheduled_instructions)
        scheduler.update_process_queue(shots, result)


    benchmark_id=0
    for proc in [process1, process2, process3, process4, process5, process6, process7]:
        benchmark_id=proc.get_process_id()-1
        ideal_result = load_ideal_count_output(benchmarktype.MULTI_CONTROLLED_X_MEDIUM,benchmark_id)
        print("Ideal Counts:", ideal_result)
        print("Received Counts:", proc.get_result_counts())
        fidelity=distribution_fidelity(ideal_result, proc.get_result_counts())
        print(f"Benchmark: {benchmark_id}")
        print(f"Fidelity: {fidelity:.10f}")


















if __name__ == "__main__":
    test_scheduling()
    #test_sequential_scheduling()