from typing import Dict
from matplotlib import pyplot as plt
import numpy as np
from qiskit.transpiler import CouplingMap
from qiskit.providers.fake_provider import GenericBackendV2  # lives here

def torino_coupling_map():
    COUPLING = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], # The first long row
        [0,15], [15,19], [4,16], [16,23], [8,17], [17,27], [12,18], [18,31], # Short row 1
        [19,20], [20,21], [21,22], [22,23], [23,24], [24,25], [25,26], [26,27], [27,28], [28,29], [29,30], [30,31], [31,32], [32,33], # The second long row
        [21,34], [34,40], [25,35], [35,44], [29,36], [36,48], [33,37], [37,52], # Short row 2
        [38,39], [39,40], [40,41], [41,42], [42,43], [43,44], [44,45], [45,46], [46,47], [47,48], [48,49], [49,50], [50,51], [51,52], # The third long row
        [38,53], [53,57], [42,54], [54,61], [46,55], [55,65], [50,56], [56,69], # Short row 3
        [57,58], [58,59], [59,60], [60,61], [61,62], [62,63], [63,64], [64,65], [65,66], [66,67], [67,68], [68,69], [69,70], [70,71], # The forth long row
        [59,72], [72,78], [63,73], [73,82], [67,74], [74,86], [71,75], [75,90], # Short row 4
        [76,77], [77,78], [78,79], [79,80], [80,81], [81,82], [82,83], [83,84], [84,85], [85,86], [86,87], [87,88], [88,89], [89,90], # The fifth long row
        [76,91], [91,95], [80,92], [92,99], [84,93], [93,103], [88,94], [94,107], # Short row 5
        [95,96], [96,97], [97,98], [98,99], [99,100], [100,101], [101,102], [102,103], [103,104], [104,105], [105,106], [106,107], [107,108], [108,109], # The sixth long row
        [97,110], [110,116], [101,111], [111,120], [105,112], [112,124],[109,113], [113,128], # Short row 6
        [114,115], [115,116], [116,117], [117,118], [118,119], [119,120], [120,121], [121,122], [122,123], [123,124], [124,125], [125,126], [126,127], [127,128], # The seventh long row
        [114,129], [118, 130], [122,131], [126,132]  # Short row 7
    ]
    return COUPLING


def simple_10_qubit_coupling_map():
    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], [0,5], [1,6], [2,7], [3,8], [4,9],[5,6], [6,7],[7,8],[8,9]]  # linear chain
    return COUPLING


def simple_20_qubit_coupling_map():
    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], 
                [0,5], [1,6], [2,7], [3,8], [4,9],
                [5,6], [6,7],[7,8],[8,9],
                [5,10], [6, 11], [7, 12], [8, 13], [9, 14],
                [10,11],[11,12], [12,13], [13,14], 
                [10,15], [11,16], [12,17], [13,18], [14,19],
                [15,16], [16,17], [17,18], [18,19]]  
    return COUPLING

def get_10_qubit_hardware_coords() -> list[tuple[float, float]]:
    edge_length = 1
    coords = [ ]
    for i in range(10):
        if i<5:
            coords.append( (float(i*edge_length), 0.0) )
        else:
            coords.append( (float((i-5)*edge_length), -edge_length))
    return coords


def torino_qubit_coords() -> list[tuple[float, float]]:
    coords = [(0.0, 0.0)] * 133

    # Long rows: each has 16 nodes, at x=0..15
    long_starts = [0, 19, 38, 57, 76, 95, 114]
    for r, start in enumerate(long_starts):
        y = -2.0 * r
        for k in range(15):
            coords[start + k] = (float(k), y)

    # Short rows: each has 4 nodes, alternating column anchors
    short_starts = [15, 34, 53, 72, 91, 110, 129]

    anchors_odd  = [0, 4, 8, 12]  # short rows 1,3,5,7
    anchors_even = [2, 6, 10, 14]   # short rows 2,4,6
    for s, start in enumerate(short_starts):
        y = -(2.0 * s + 1.0)
        xs = anchors_odd if (s % 2 == 0) else anchors_even
        for j, x in enumerate(xs):
            coords[start + j] = (float(x), y)

    return coords



def construct_fake_ibm_torino():
    NUM_QUBITS = 133


    # Directed edges (bidirectional 0<->1 and 1->2)
    COUPLING = [
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], # The first long row
        [0,15], [15,19], [4,16], [16,23], [8,17], [17,27], [12,18], [18,31], # Short row 1
        [19,20], [20,21], [21,22], [22,23], [23,24], [24,25], [25,26], [26,27], [27,28], [28,29], [29,30], [30,31], [31,32], [32,33], # The second long row
        [21,34], [34,40], [25,35], [35,44], [29,36], [36,48], [33,37], [37,52], # Short row 2
        [38,39], [39,40], [40,41], [41,42], [42,43], [43,44], [44,45], [45,46], [46,47], [47,48], [48,49], [49,50], [50,51], [51,52], # The third long row
        [38,53], [53,57], [42,54], [54,61], [46,55], [55,65], [50,56], [56,69], # Short row 3
        [57,58], [58,59], [59,60], [60,61], [61,62], [62,63], [63,64], [64,65], [65,66], [66,67], [67,68], [68,69], [69,70], [70,71], # The forth long row
        [59,72], [72,78], [63,73], [73,82], [67,74], [74,86], [71,75], [75,90], # Short row 4
        [76,77], [77,78], [78,79], [79,80], [80,81], [81,82], [82,83], [83,84], [84,85], [85,86], [86,87], [87,88], [88,89], [89,90], # The fifth long row
        [76,91], [91,95], [80,92], [92,99], [84,93], [93,103], [88,94], [94,107], # Short row 5
        [95,96], [96,97], [97,98], [98,99], [99,100], [100,101], [101,102], [102,103], [103,104], [104,105], [105,106], [106,107], [107,108], [108,109], # The sixth long row
        [97,110], [110,116], [101,111], [111,120], [105,112], [112,124],[109,113], [113,128], # Short row 6
        [114,115], [115,116], [116,117], [117,118], [118,119], [119,120], [120,121], [121,122], [122,123], [123,124], [124,125], [125,126], [126,127], [127,128], # The seventh long row
        [114,129], [118, 130], [122,131], [126,132]  # Short row 7
    ]

    BASIS = ["cz","id","rx","rz","rzz","sx","x"]  # add more *only* if truly native

    backend = GenericBackendV2(
        num_qubits=NUM_QUBITS,
        basis_gates=BASIS,         # optional
        coupling_map=COUPLING,     # strongly recommended
        control_flow=True,        # set True if you want dynamic circuits            
        seed=1234,                 # reproducible auto-generated props
        noise_info=True            # attach plausible noise/durations
    )

    return backend



def construct_10_qubit_hardware():
    NUM_QUBITS = 10
    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], [0,5], [1,6], [2,7], [3,8], [4,9],[5,6], [6,7],[7,8],[8,9]]  # linear chain
    BASIS = ["cx", "id", "rz", "sx", "x"]  # add more *only* if truly native

    backend = GenericBackendV2(
        num_qubits=NUM_QUBITS,
        basis_gates=BASIS,         # optional
        coupling_map=COUPLING,     # strongly recommended
        control_flow=True,        # set True if you want dynamic circuits            
        seed=1234,                 # reproducible auto-generated props
        noise_info=True            # attach plausible noise/durations
    )

    return backend    



def construct_20_qubit_hardware():
    NUM_QUBITS = 20
    COUPLING = [[0, 1], [1, 2], [2, 3], [3, 4], 
                [0,5], [1,6], [2,7], [3,8], [4,9],
                [5,6], [6,7],[7,8],[8,9],
                [5,10], [6, 11], [7, 12], [8, 13], [9, 14],
                [10,11],[11,12], [12,13], [13,14], 
                [10,15], [11,16], [12,17], [13,18], [14,19],
                [15,16], [16,17], [17,18], [18,19]]  
    BASIS = ["cx", "id", "rz", "sx", "x"]  # add more *only* if truly native

    backend = GenericBackendV2(
        num_qubits=NUM_QUBITS,
        basis_gates=BASIS,         # optional
        coupling_map=COUPLING,     # strongly recommended
        control_flow=True,        # set True if you want dynamic circuits            
        seed=1234,                 # reproducible auto-generated props
        noise_info=True            # attach plausible noise/durations
    )

    return backend    


def construct_30_qubit_hardware():
    NUM_QUBITS = 30
    BASIS = ["cx", "id", "rz", "sx", "x"]  # add more *only* if truly native

    # We'll arrange the qubits in a 5x6 grid:
    # row 0:  0  1  2  3  4  5
    # row 1:  6  7  8  9 10 11
    # row 2: 12 13 14 15 16 17
    # row 3: 18 19 20 21 22 23
    # row 4: 24 25 26 27 28 29
    #
    # Couplings: all horizontal neighbors and all vertical neighbors.
    COUPLING = [
        # Row 0 horizontal
        [0,1], [1,2], [2,3], [3,4], [4,5],
        # Row 1 horizontal
        [6,7], [7,8], [8,9], [9,10], [10,11],
        # Row 2 horizontal
        [12,13], [13,14], [14,15], [15,16], [16,17],
        # Row 3 horizontal
        [18,19], [19,20], [20,21], [21,22], [22,23],
        # Row 4 horizontal
        [24,25], [25,26], [26,27], [27,28], [28,29],

        # Vertical edges
        [0,6], [1,7], [2,8], [3,9], [4,10], [5,11],
        [6,12], [7,13], [8,14], [9,15], [10,16], [11,17],
        [12,18], [13,19], [14,20], [15,21], [16,22], [17,23],
        [18,24], [19,25], [20,26], [21,27], [22,28], [23,29],
    ]
    backend = GenericBackendV2(
        num_qubits=NUM_QUBITS,
        basis_gates=BASIS,     # optional
        coupling_map=COUPLING, # strongly recommended
        control_flow=True,     # set True if you want dynamic circuits
        seed=1234,             # reproducible auto-generated props
        noise_info=True        # attach plausible noise/durations
    )

    return backend




def simple_30_qubit_coupling_map():
    COUPLING = [
        # Row 0 horizontal
        [0,1], [1,2], [2,3], [3,4], [4,5],
        # Row 1 horizontal
        [6,7], [7,8], [8,9], [9,10], [10,11],
        # Row 2 horizontal
        [12,13], [13,14], [14,15], [15,16], [16,17],
        # Row 3 horizontal
        [18,19], [19,20], [20,21], [21,22], [22,23],
        # Row 4 horizontal
        [24,25], [25,26], [26,27], [27,28], [28,29],

        # Vertical edges
        [0,6], [1,7], [2,8], [3,9], [4,10], [5,11],
        [6,12], [7,13], [8,14], [9,15], [10,16], [11,17],
        [12,18], [13,19], [14,20], [15,21], [16,22], [17,23],
        [18,24], [19,25], [20,26], [21,27], [22,28], [23,29],
    ]
    return COUPLING



def get_20_qubit_hardware_coords() -> list[tuple[float, float]]:
    edge_length = 1
    coords = [ ]
    for i in range(20):
        if i<5:
            coords.append( (float(i*edge_length), 0.0) )
        elif i<10:
            coords.append( (float((i-5)*edge_length), -edge_length))
        elif i<15:
            coords.append( (float((i-10)*edge_length), -2*edge_length))
        else:
            coords.append( (float((i-15)*edge_length), -3*edge_length))
    return coords