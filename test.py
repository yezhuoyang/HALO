#Test the correctness of halo scheduling
from halo import *
from process import plot_process_schedule_on_10_qubit_hardware

from qiskit.qasm2 import dumps







def test_scheduling():
    """
    A simple test function for the haloScheduler
    """
    process1 = generate_process_from_benchmark(benchmark_id=0, pid=1, shots=500, share_qubit=True)
    process2 = generate_process_from_benchmark(benchmark_id=1, pid=2, shots=500, share_qubit=True)

    print("Start testing haloScheduler...")
    scheduler = haloScheduler(use_simulator=True)
    scheduler.add_process(process1,source_id=0)
    scheduler.add_process(process2,source_id=1)


    print("Start getting next batch...")
    shots, next_batch = scheduler.get_next_batch(force_return=True)


    L = scheduler.allocate_data_qubit(next_batch)


    plot_process_schedule_on_10_qubit_hardware(
        simple_10_qubit_coupling_map(),
        next_batch,
        L,
        out_png="best_10_qubit_mapping_6proc.png",
    )

    total_measurements, measurement_to_process_map, scheduled_instructions = scheduler.dynamic_helper_scheduling(L, next_batch)


    qiskit_circuit = construct_qiskit_circuit_for_hardware_instruction(total_measurements, scheduled_instructions)


    qiskit_circuit.draw("mpl").savefig("constructed_circuit.png")
    print(print(dumps(qiskit_circuit)))
    shots = 2000

    sim = AerSimulator()
    tqc = transpile(qiskit_circuit, sim)

    # Run with 1000 shots
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts(tqc)



    print("Simulated Counts:", counts)




    # print("Instructions:")
    # for inst in scheduled_instructions:
    #     print(inst)


    # qiskit_circuit = construct_qiskit_circuit_for_hardware_instruction(total_measurements, scheduled_instructions)


                # Step 4: Send the scheduled instructions to hardware
    # result=scheduler._jobmanager.execute_on_hardware(shots,total_measurements,measurement_to_process_map,scheduled_instructions)







    # scheduler.update_process_queue(shots,result)

    # ideal_result = load_ideal_count_output(0)
    
    # scheduler.update_process_queue(shots,result)
    # fidelity1=distribution_fidelity(ideal_result, process1.get_result_counts())
    # print("Fidelity for process 1:", fidelity1)

    # print("Process 1 result counts:", process1.get_result_counts())

    # print("Process 1 ideal result:", ideal_result)


    # ideal_result = load_ideal_count_output(1)
    # fidelity2=distribution_fidelity(ideal_result, process2.get_result_counts())
    # print("Fidelity for process 2:", fidelity2)



if __name__ == "__main__":
    test_scheduling()





