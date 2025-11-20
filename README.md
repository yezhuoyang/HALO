# HALO



The is the halo(Helper qubit aware layout optimzation) scheduler for IBM quantum computer. 


# How to run the experiment?

The run the experiment on the benchmark suit, directly run:

```python
py halo.py
```

Note that if you want to run experiment on a real quantum computer, please store your APIkey in the aipkey file.


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






