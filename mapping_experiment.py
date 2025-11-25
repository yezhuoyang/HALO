#Study the effect of mapping cost on the fidelity with concrete examples
from halo import *
from process import plot_process_schedule_on_10_qubit_hardware

from qiskit.qasm2 import dumps












def test_scheduling():
    """
    A simple test function for the haloScheduler
    """
    process1 = generate_process_from_benchmark(benchmark_id=0, pid=1, shots=1000, share_qubit=True)
    process2 = generate_process_from_benchmark(benchmark_id=1, pid=2, shots=1000, share_qubit=True)
    process3 = generate_process_from_benchmark(benchmark_id=2, pid=3, shots=1000, share_qubit=True)
    process4 = generate_process_from_benchmark(benchmark_id=3, pid=4, shots=1000, share_qubit=True)
    process5 = generate_process_from_benchmark(benchmark_id=4, pid=5, shots=1000, share_qubit=True)
    process6 = generate_process_from_benchmark(benchmark_id=5, pid=6, shots=1000, share_qubit=True)
    process7 = generate_process_from_benchmark(benchmark_id=6, pid=7, shots=1000, share_qubit=True)
    process8 = generate_process_from_benchmark(benchmark_id=0, pid=8, shots=1000, share_qubit=True)
    process9 = generate_process_from_benchmark(benchmark_id=1, pid=9, shots=1000, share_qubit=True)
    process10 = generate_process_from_benchmark(benchmark_id=2, pid=10, shots=1000, share_qubit=True)
    process11 = generate_process_from_benchmark(benchmark_id=3, pid=11, shots=1000, share_qubit=True)
    process12 = generate_process_from_benchmark(benchmark_id=4, pid=12, shots=1000, share_qubit=True)
    process13 = generate_process_from_benchmark(benchmark_id=5, pid=13, shots=1000, share_qubit=True)
    process14 = generate_process_from_benchmark(benchmark_id=6, pid=14, shots=1000, share_qubit=True)



    print("Start testing haloScheduler...")
    scheduler = haloScheduler(use_simulator=False)
    scheduler.add_process(process1,source_id=0)
    scheduler.add_process(process2,source_id=1)
    scheduler.add_process(process3,source_id=2)
    scheduler.add_process(process4,source_id=3)
    scheduler.add_process(process5,source_id=4)
    scheduler.add_process(process6,source_id=5)
    scheduler.add_process(process7,source_id=6)
    scheduler.add_process(process8,source_id=0)
    scheduler.add_process(process9,source_id=1)
    scheduler.add_process(process10,source_id=2)
    scheduler.add_process(process11,source_id=3)
    scheduler.add_process(process12,source_id=4)
    scheduler.add_process(process13,source_id=5)
    scheduler.add_process(process14,source_id=6)




    shots, next_batch = scheduler.get_next_batch(force_return=True)


    L = scheduler.allocate_data_qubit(next_batch)


    plot_process_schedule_on_torino(
        torino_coupling_map(),
        next_batch,
        L,
        out_png="first_case.png",
    )


    total_measurements, measurement_to_process_map, scheduled_instructions = scheduler.dynamic_helper_scheduling(L, next_batch)

    
    result=scheduler._jobmanager.execute_on_hardware(shots,total_measurements,measurement_to_process_map,scheduled_instructions)


    scheduler.update_process_queue(shots,result)


    benchmark_id=0
    for proc in next_batch:
        benchmark_id=(proc.get_process_id()-1)%7
        ideal_result = load_ideal_count_output(benchmark_id)
        fidelity=distribution_fidelity(ideal_result, proc.get_result_counts())
        print(f"Benchmark: {benchmark_id}")
        print(f"Fidelity: {fidelity}")









if __name__ == "__main__":
    test_scheduling()