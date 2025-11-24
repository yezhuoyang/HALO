# HALO



The is the halo(Helper qubit aware layout optimzation) scheduler for IBM quantum computer. 


# How to run the experiment?

The run the experiment on the benchmark suit, directly run:

```python
py halo.py
```

Note that if you want to run experiment on a real quantum computer, please store your APIkey in the aipkey file.



# Run Halo scheduling


The following code do experiment of halo scheduling:


```python
if __name__ == "__main__":
    random.seed(42)
    haloScheduler_instance=haloScheduler(use_simulator=True)   #Change this parameter to False if you want to run on real quantum hardware
    haloScheduler_instance.start()

    producer_thread = threading.Thread(
        target=random_arrival_generator,
        args=(haloScheduler_instance, 0.3, 40.0,True),
        daemon=False
    )
    producer_thread.start()
    simulation_time = 40  # Total running time for simulation
    time.sleep(simulation_time)
    # Wait for producer to finish generating all processes
    producer_thread.join()
    haloScheduler_instance.stop()
    print("Simulation finished.")
    haloScheduler_instance.store_log("halo_scheduler_log_not_share.txt")
```



# Run naive sequential scheduling


# Run space-share scheduling without helper qubit sharing


# Run shot-unaware scheduling



# How to setup parameters?

The scheduling experiment has the following hpyer parameters:

1. The rate of bossion distribution
2. The total running time.
3. The parameter is the data qubit layout cost. 
4. The maximal ratio of data qubit to total number of qubits.


# Figure of merit

We measure the following metrics to evaluate the performance:

1. The average accuracy of all quantum processes
2. The average waiting time for all quantum process
3. The throughput of the sheduler





# Scheduling log


After experiments finish, you will see a halo_scheduler_log.txt file,


# Result visualization


You can visualize the result in visulization.py file.






