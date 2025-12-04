from enum import Enum
import random
from typing import List, Tuple, Optional
import qiskit.qasm2
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator  # <- use AerSimulator (Qiskit 2.x)
import pickle


"""
instruction.py

This module defines the atomic structure of a user process: the Instruction class
Each instruction has a type (e.g., H, CNOT, MEASURE), a list of target qubits, and a timestamp.

Classes:
    Instruction -- Represents a quantum gate operation or measurement

Example:
    >>> inst = Instruction(type=Instype.H, qubit="q1")
    >>> print(inst)
    H gate on qubit 0 at time 5
"""

class Instype(Enum):
    H = 1
    X = 2
    Y = 3
    Z = 3
    T = 4
    Tdg = 5
    U = 6
    S = 7
    Sdg = 8
    SX = 9
    RZ = 10
    RX = 11
    RY = 12
    U3 = 13
    Toffoli = 14
    CNOT = 15
    CH = 16
    SWAP = 17
    CSWAP = 18
    CP = 19
    RESET = 20
    MEASURE = 21
    RELEASE = 22



def get_gate_type_name(type: Instype) -> str:
    """
    Get the name of the gate type.
    
    Args:
        type (Instype): The type of the instruction.
    
    Returns:
        str: The name of the instruction type.
    """
    match type:
        case Instype.H:
            return "H"
        case Instype.X:
            return "X"  
        case Instype.Y:
            return "Y"
        case Instype.Z:
            return "Z"
        case Instype.T:
            return "T"
        case Instype.U:
            return "U"
        case Instype.Tdg:
            return "Tdg"
        case Instype.S:
            return "S"
        case Instype.Sdg:
            return "Sdg"
        case Instype.SX:
            return "SX"
        case Instype.RZ:
            return "RZ"
        case Instype.RX:
            return "RX"
        case Instype.RY:
            return "RY"
        case Instype.U3:
            return "U3"
        case Instype.Toffoli:
            return "Toffoli"
        case Instype.CNOT:
            return "CNOT"
        case Instype.SWAP:
            return "SWAP"
        case Instype.CSWAP:
            return "CSWAP"
        case Instype.CP:   
            return "CP"
        case Instype.CH:
            return "CH"
        case Instype.RESET:
            return "RESET"
        case Instype.MEASURE:
            return "MEASURE"
        case Instype.RELEASE:
            return "RELEASE"
        case _:
            raise ValueError("Unknown instruction type")




class instruction:

    """
    Initialize a new instruction.

    qubit address are represented as string.

    For data qubit, the convention for data qubit k is qk,
    for helper qubit, the convention of helper qubit k is sk
    """
    def __init__(self, type:Instype, qubitaddress:List[str],classical_address: int=None, reset_address: int=None, params: List[float]=None) -> None:
        self._type=type
        self._qubitaddress=qubitaddress
        self._scheduled_mapped_address={} # This is the physical address after scheduling
        self._params=params if params is not None else [] # The rotation angles for rotation gates, for example, for RZ(0.2*pi), params=[0.2]
        if self._type==Instype.MEASURE:
            if classical_address is None:
                raise ValueError("Classical address must be provided for MEASURE instruction.")
            self._classical_address=classical_address
            self._scheduled_classical_address=None
        if self._type==Instype.RESET:
            if reset_address is None:
                raise ValueError("Reset address must be provided for RESET instruction.")
            self._reset_address=reset_address
        self._helper_qubit_count=0
        for addr in qubitaddress:
            if addr.startswith('s'):
                self._helper_qubit_count+=1

        self._processID=-1



    def get_classical_address(self) -> int:
        """
        Get the classical address associated with the instruction.
        
        Returns:
            int: The classical address for the instruction.
        """
        if self._type != Instype.MEASURE:
            raise ValueError("Classical address is only applicable for MEASURE instructions.")
        return self._classical_address



    def set_scheduled_classical_address(self, classical_address: int):
        """
        Set the scheduled classical address associated with the instruction.
        
        Args:
            classical_address (int): The scheduled classical address to set.
        """
        if self._type != Instype.MEASURE:
            raise ValueError("Scheduled classical address is only applicable for MEASURE instructions.")
        self._scheduled_classical_address = classical_address


    def get_scheduled_classical_address(self) -> int:
        """
        Get the scheduled classical address associated with the instruction.
        
        Returns:
            int: The scheduled classical address for the instruction.
        """
        if self._type != Instype.MEASURE:
            raise ValueError("Scheduled classical address is only applicable for MEASURE instructions.")
        return self._scheduled_classical_address



    def set_processID(self, processID: int):
        """
        Set the process ID associated with the instruction.
        
        Args:
            processID (int): The process ID to set.
        """
        self._processID=processID


    def get_processID(self) -> int:
        """
        Get the process ID associated with the instruction.
        
        Returns:
            int: The process ID for the instruction.
        """
        return self._processID



    def get_reset_address(self) -> int:
        """
        Get the reset address associated with the instruction.
        
        Returns:
            int: The reset address for the instruction.
        """
        if self._type != Instype.RESET:
            raise ValueError("Reset address is only applicable for RESET instructions.")
        return self._reset_address



    def get_helper_qubit_count(self) -> int:
        """
        Get the number of helper qubits involved in the instruction.
        
        Returns:
            int: The number of helper qubits.
        """
        return self._helper_qubit_count



    def get_all_helper_qubit_addresses(self) -> List[str]:
        """
        Get the addresses of all helper qubits involved in the instruction.
        
        Returns:
            List[str]: A list of helper qubit addresses.
        """
        helper_qubit_addresses = []
        for addr in self._qubitaddress:
            if addr.startswith('s'):
                helper_qubit_addresses.append(addr)
        return helper_qubit_addresses



    def get_all_data_qubit_addresses(self) -> List[str]:
        """
        Get the addresses of all data qubits involved in the instruction.
        
        Returns:
            List[str]: A list of data qubit addresses.
        """
        data_qubit_addresses = []
        for addr in self._qubitaddress:
            if addr.startswith('q'):
                data_qubit_addresses.append(addr)
        return data_qubit_addresses



    def is_release_helper_qubit(self) -> bool:
        """
        Check if the instruction is a release operation.
        
        Returns:
            bool: True if the instruction is a release, False otherwise.
        """
        return self._type == Instype.RELEASE



    def is_system_call(self) -> bool:
        """
        Check if the instruction is a system call.
        
        Returns:
            bool: True if the instruction is a system call, False otherwise.
        """
        return self._type in {Instype.RELEASE}



    def is_two_qubit_gate(self) -> bool:
        """
        Check if the instruction is a two-qubit gate.
        
        Returns:
            bool: True if the instruction is a two-qubit gate, False otherwise.
        """
        return self._type in {Instype.CNOT, Instype.CH, Instype.SWAP, Instype.CP}


    def get_params(self) -> List[float]:
        """
        Return the parameters associated with the instruction.
        
        Returns:
            List[float]: The list of parameters for the instruction.
        """
        return self._params
    

    def get_classical_address(self) -> int:
        """
        Get the classical address associated with the instruction.
        
        Returns:
            int: The classical address for the instruction.
        """
        if self._type != Instype.MEASURE:
            raise ValueError("Classical address is only applicable for MEASURE instructions.")
        return self._classical_address


    def reset_mapping(self):
        self._scheduled_mapped_address={} # This is the physical address after scheduling
        self._scheduled_classical_address=None

    def is_reset(self) -> bool:
        """
        Check if the instruction is a reset operation.
        
        Returns:
            bool: True if the instruction is a reset, False otherwise.
        """
        return self._type == Instype.RESET



    def is_measurement(self) -> bool:
        """
        Check if the instruction is a measurement operation.
        
        Returns:
            bool: True if the instruction is a measurement, False otherwise.
        """
        return self._type == Instype.MEASURE



    def is_scheduled(self) -> bool:
        """
        Check if the instruction has been scheduled (i.e., mapped to physical addresses).
        
        Returns:
            bool: True if the instruction is scheduled, False otherwise.
        """
        return len(self._scheduled_mapped_address) == len(self._qubitaddress)



    def set_scheduled_mapped_address(self, virtualaddress: str, physicaladdress: int):
        """
        Set the scheduled mapped address for a given qubit address.
        
        Args:
            qubitaddress (virtualAddress): The virtual address of the qubit.
            physicaladdress (int): The physical address to which the qubit is mapped.
        """
        self._scheduled_mapped_address[virtualaddress] = physicaladdress


    def get_scheduled_mapped_address(self, virtualAddress: str) -> int:
        """
        Get the scheduled mapped addresses for the instruction.
        
        Returns:
            dict: A dictionary mapping virtual qubit addresses to physical addresses.
        """
        return self._scheduled_mapped_address[virtualAddress]


    def get_processID(self) -> int:
        """
        Get the process ID associated with the instruction.
        """
        return self._processID


    def get_type(self) -> Instype:
        """
        Get the type of the instruction.
        """
        return self._type

    def get_qubitaddress(self) -> List[str]:
        """
        Get the list of qubit addresses associated with the instruction.
        """
        return self._qubitaddress


    def __str__(self) -> str:
        outputstr="P("+str(self._processID)+"): "
        match self._type:
            case Instype.H:
                outputstr+="H"
            case Instype.X:
                outputstr+="X"
            case Instype.Y:
                outputstr+="Y"
            case Instype.Z:
                outputstr+="Z"
            case Instype.T:
                outputstr+="T"
            case Instype.Tdg:
                outputstr+="Tdg"
            case Instype.S:
                outputstr+="S"
            case Instype.Sdg:
                outputstr+="Sdg"
            case Instype.SX:
                outputstr+="SX"
            case Instype.RZ:
                outputstr+="RZ("+str(self._params[0])+"*pi)"
            case Instype.RX:
                outputstr+="RX("+str(self._params[0])+"*pi)"
            case Instype.RY:
                outputstr+="RY("+str(self._params[0])+"*pi)"
            case Instype.U3:
                outputstr+="U3("+str(self._params[0])+"*pi, "+str(self._params[1])+"*pi, "+str(self._params[2])+"*pi)"
            case Instype.U:
                outputstr+="U("+str(self._params[0])+"*pi, "+str(self._params[1])+"*pi, "+str(self._params[2])+"*pi)"
            case Instype.Toffoli:
                outputstr+="Toffoli"    
            case Instype.CNOT:
                outputstr+="CNOT"
            case Instype.CH:
                outputstr+="CH"
            case Instype.SWAP:
                outputstr+="SWAP"
            case Instype.CSWAP:
                outputstr+="CSWAP"
            case Instype.CP:
                outputstr+="CP("+str(self._params[0])+"*pi)"
            case Instype.RESET:
                outputstr+="RESET"+" qubit("+str(self._reset_address)+")"
                return outputstr
            case Instype.MEASURE:
                outputstr+="c" + str(self._classical_address) + "=MEASURE"
            case Instype.RELEASE:
                outputstr+="RELEASE"
        outputstr+=" qubit("
        for i, addr in  enumerate(self._qubitaddress):
            outputstr+=addr
            if addr in self._scheduled_mapped_address:
                outputstr+="->"+str(self._scheduled_mapped_address[addr])
            if i!=len(self._qubitaddress)-1:
                outputstr+=", "
        outputstr+=")"
        return outputstr

    def __repr__(self):
        return self.__str__()





def construct_qiskit_circuit(num_data_qubit: int, num_syndrome_qubit: int, num_classical_bits: int, instruction_list: List[instruction]) -> QuantumCircuit:
    """
    Construct a qiskit circuit from the instruction list.
    Also help to visualize the circuit.
    """
    dataqubit = QuantumRegister(num_data_qubit, "q")

    # Second part: 2 qubits named 's'
    syndromequbit = QuantumRegister(num_syndrome_qubit, "s")

    # Classical registers (optional, if you want measurements)
    classicalbits = ClassicalRegister(num_classical_bits, "c")

    # Combine them into one circuit
    qiskit_circuit = QuantumCircuit(dataqubit,  syndromequbit, classicalbits)

    for inst in instruction_list:
        if inst.is_system_call():
            continue
        addresses = inst.get_qubitaddress()
        qiskitaddress=[]
        for addr in addresses:
            if addr.startswith('s'):
                print(addr)
                qiskitaddress.append(syndromequbit[int(addr[1:])])
            else:
                qiskitaddress.append(dataqubit[int(addr[1:])])
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
                qiskit_circuit.u(params[0], params[1], params[2], qiskitaddress[0])
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
                qiskit_circuit.reset(qiskitaddress[0])
            case Instype.RELEASE:
                qiskit_circuit.reset(qiskitaddress[0])
            case Instype.MEASURE:
                classical_address=inst.get_classical_address()
                qiskit_circuit.measure(qiskitaddress[0], classical_address)

    return qiskit_circuit





import re


_param_list_re = re.compile(r"\(([^)]*)\)")

def _parse_float_list(s: str) -> List[float]:
    """
    Parse a comma-separated list of floats inside parentheses.
    Accepts whitespace; returns [] if s is empty/only spaces.
    """
    s = s.strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    return [float(p) for p in parts if p]


# def _grab_params(text_after_gate: str) -> Tuple[List[float], str]:
#     """
#     Given '(<params>) rest', return (params_list, rest_without_paren_prefix).
#     If no parentheses, returns ([], original_text).
#     """
#     m = _param_list_re.match(text_after_gate.strip())
#     if not m:
#         return [], text_after_gate.strip()
#     params = _parse_float_list(m.group(1))
#     # remove the '(...)' prefix from the remaining text
#     remainder = text_after_gate.strip()[m.end():].strip()
#     return params, remainder


def _grab_params(param_str: str) -> Tuple[List[float], str]:
    """
    Extracts numbers from parentheses and returns the remaining string.
    Input: "(1.23, 4.56) q0" -> Returns: ([1.23, 4.56], " q0")
    Input: " q0"             -> Returns: ([], " q0")
    """
    param_str = param_str.strip()
    if not param_str.startswith("("):
        return [], param_str
    
    # Find the closing parenthesis
    try:
        end_idx = param_str.index(")")
    except ValueError:
        raise ValueError(f"Missing closing parenthesis in parameters: {param_str}")

    content = param_str[1:end_idx]
    remainder = param_str[end_idx+1:]
    
    # Parse floats (handles '1.23, 4.56')
    # Note: For production, use a safe eval or math expression parser here
    try:
        params = [float(x.strip()) for x in content.split(",") if x.strip()]
    except ValueError:
        # Fallback for simple symbolic math like "3.14" if pure float fails
        # or raise error
        raise ValueError(f"Could not parse parameters: {content}")
        
    return params, remainder

def parse_program_from_string(qasm_code: str) -> Tuple[List[instruction], int, int, int]:
    # Normalize lines
    # Normalize lines, skip empty and comment lines
    raw_lines = [
        ln.strip()
        for ln in qasm_code.strip().splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]

    # Header Regex
    alloc_data_re = re.compile(r"^q\s*=\s*alloc_data\(\s*(\d+)\s*\)\s*$", re.IGNORECASE)
    alloc_syn_re  = re.compile(r"^s\s*=\s*alloc_helper\(\s*(\d+)\s*\)\s*$", re.IGNORECASE)
    set_shot_re   = re.compile(r"^set_shot\(\s*(\d+)\s*\)\s*$", re.IGNORECASE)
    
    # Special Instruction Regex
    measure_re = re.compile(r"^c(\d+)\s*=\s*MEASURE\s+([qs]\d+)", re.IGNORECASE)
    dealloc_s_re = re.compile(r"^deallocate_helper\(\s*(s\d+)\s*\)", re.IGNORECASE)
    dealloc_q_re = re.compile(r"^deallocate_data\(\s*q\s*\)", re.IGNORECASE)

    # Gate Registry: Name -> (Instype, expected_param_count, expected_qubit_count)
    GATE_MAP = {
        # No Params
        "h": (Instype.H, 0, 1), "x": (Instype.X, 0, 1), "y": (Instype.Y, 0, 1), "z": (Instype.Z, 0, 1),
        "t": (Instype.T, 0, 1), "tdg": (Instype.Tdg, 0, 1), "s": (Instype.S, 0, 1), "sdg": (Instype.Sdg, 0, 1),
        "sx": (Instype.SX, 0, 1), "reset": (Instype.RESET, 0, 1),
        "cnot": (Instype.CNOT, 0, 2), "cx": (Instype.CNOT, 0, 2), "ch": (Instype.CH, 0, 2), "swap": (Instype.SWAP, 0, 2),
        "ccx": (Instype.Toffoli, 0, 3), "toffoli": (Instype.Toffoli, 0, 3), "cswap": (Instype.CSWAP, 0, 3),
        
        # Params
        "rx": (Instype.RX, 1, 1), "ry": (Instype.RY, 1, 1), "rz": (Instype.RZ, 1, 1),
        "u": (Instype.U, 3, 1), "u3": (Instype.U3, 3, 1),
        "cp": (Instype.CP, 1, 2), "cu1": (Instype.CP, 1, 2)
    }

    data_n = 0
    syn_n = 0
    measure_n = 0
    inst_list: List[instruction] = []

    for ln in raw_lines:
        # --- 1. Header Handling ---
        if m := alloc_data_re.match(ln):
            data_n = int(m.group(1))
            continue
        if m := alloc_syn_re.match(ln):
            syn_n = int(m.group(1))
            continue
        if m := set_shot_re.match(ln):
            continue
        
        # --- 2. Special Instructions ---
        if m := measure_re.match(ln):
            c_idx = int(m.group(1))
            q_target = m.group(2)
            measure_n += 1
            inst_list.append(instruction(Instype.MEASURE, [q_target], classical_address=c_idx))
            continue
            
        if m := dealloc_s_re.match(ln):
            inst_list.append(instruction(Instype.RELEASE, [m.group(1)]))
            continue
        
        if dealloc_q_re.match(ln):
            continue

        # --- 3. Robust Gate Parsing ---
        
        # Split first token (gate) from the rest
        parts = ln.split(None, 1)
        if not parts: continue
        
        raw_gate_token = parts[0]
        
        # Logic Fix: Handle "CP(1.23)" where param is attached to name
        if "(" in raw_gate_token:
            split_idx = raw_gate_token.find("(")
            gate_name = raw_gate_token[:split_idx]  # Extract "CP"
            # Reconstruct the "rest of line" to include the params we just sliced off
            attached_params = raw_gate_token[split_idx:]
            rest_of_line = attached_params + (" " + parts[1] if len(parts) > 1 else "")
        else:
            gate_name = raw_gate_token
            rest_of_line = parts[1] if len(parts) > 1 else ""

        gate_name = gate_name.lower()

        if gate_name not in GATE_MAP:
            raise ValueError(f"Unknown gate: {gate_name} in line '{ln}'")

        inst_type, expected_params, expected_qubits = GATE_MAP[gate_name]
        
        params = []
        qubits = []

        # --- Strategy A: Parenthesis Parsing ---
        # Checks if parentheses exist in the arguments part
        if "(" in rest_of_line:
            start = rest_of_line.find("(")
            end = rest_of_line.rfind(")")
            if start != -1 and end != -1:
                p_str = rest_of_line[start+1:end]
                q_str = rest_of_line[end+1:]
                
                if p_str.strip():
                    params = [float(x.strip()) for x in p_str.split(",")]
                qubits = [q.strip() for q in q_str.split(",") if q.strip()]

        # --- Strategy B: Linear Parsing ---
        # Handles "CP 1.25, q0" or "CP 1.25 q0"
        else:
            normalized_args = rest_of_line.replace(",", " ")
            tokens = [t.strip() for t in normalized_args.split()]
            
            for token in tokens:
                try:
                    val = float(token)
                    params.append(val)
                except ValueError:
                    qubits.append(token)

        # --- Validation ---
        if len(params) != expected_params:
            raise ValueError(f"Gate {gate_name.upper()} expects {expected_params} params, got {len(params)} in '{ln}'")

        if len(qubits) != expected_qubits:
             raise ValueError(f"Gate {gate_name.upper()} expects {expected_qubits} qubits, got {len(qubits)} in '{ln}'")

        inst_list.append(instruction(inst_type, qubits, params=(params if params else None)))

    return inst_list, data_n, syn_n, measure_n


# def parse_program_from_string(qasm_code: str) -> Tuple[List[instruction], int, int, int]:
#     """
#     Parse the custom process program DSL into a `process` object.
#     Return (inst_list, data_n, syn_n,measure_n).

#     Expects header lines:
#         q = alloc_data(N)
#         s = alloc_helper(M)
#         set_shot(S)

#     Supports gates like:
#         H q0
#         CNOT q0, q2
#         RX(1.234) q3
#         U(theta, phi, lam) q1
#         ...
#         c0 = MEASURE q2
#         deallocate_data(q)
#         deallocate_helper(s)
#     """
#     # normalize and split lines
#     raw_lines = [ln.strip() for ln in qasm_code.strip().splitlines() if ln.strip()]

#     # ---- Pass 1: read header (alloc & shots) ----
#     data_n = 0
#     syn_n = 0
#     measure_n = 0

#     alloc_data_re = re.compile(r"^q\s*=\s*alloc_data\(\s*(\d+)\s*\)\s*$", re.IGNORECASE)
#     alloc_syn_re  = re.compile(r"^s\s*=\s*alloc_helper\(\s*(\d+)\s*\)\s*$", re.IGNORECASE)
#     set_shot_re   = re.compile(r"^set_shot\(\s*(\d+)\s*\)\s*$", re.IGNORECASE)

#     # We’ll keep the non-header lines to parse as instructions
#     instr_lines: List[str] = []

#     for ln in raw_lines:
#         if m := alloc_data_re.match(ln):
#             data_n = int(m.group(1))
#             continue
#         if m := alloc_syn_re.match(ln):
#             syn_n = int(m.group(1))
#             continue
#         if m := set_shot_re.match(ln):
#             shots = int(m.group(1))
#             continue
#         # not a header line → it’s an instruction or dealloc
#         instr_lines.append(ln)


#     inst_list=[]


#     # ---- Parsers for instruction lines ----
#     # Simple 1-qubit no-parameter gates
#     oneq_no_param = {
#         "h": Instype.H,
#         "x": Instype.X,
#         "y": Instype.Y,
#         "z": Instype.Z,
#         "t": Instype.T,
#         "tdg": Instype.Tdg,
#         "s": Instype.S,
#         "sdg": Instype.Sdg,
#         "sx": Instype.SX,
#         "reset": Instype.RESET,
#     }

#     # Two-qubit gates without params
#     twoq_no_param = {
#         "cnot": Instype.CNOT,
#         "cx": Instype.CNOT,   # alias
#         "ch": Instype.CH,
#         "swap": Instype.SWAP,
#     }

#     # Three-qubit no-param
#     threeq_no_param = {
#         "cswap": Instype.CSWAP,
#         "toffoli": Instype.Toffoli,
#         "ccx": Instype.Toffoli,  # alias
#     }

#     # Helpers
#     def add_oneq_gate(kind: Instype, addr_tok: str, params: Optional[List[float]] = None):
#         if params:
#             inst_list.append(instruction(kind, [addr_tok], params=params))
#         else:
#             inst_list.append(instruction(kind, [addr_tok]))

#     def add_twoq_gate(kind: Instype, tok_a: str, tok_b: str, params: Optional[List[float]] = None):
#         if params:
#             inst_list.append(instruction(kind, [tok_a, tok_b], params=params))
#         else:
#             inst_list.append(instruction(kind, [tok_a, tok_b]))

#     def add_threeq_gate(kind: Instype, tok_a: str, tok_b: str, tok_c: str):
#         inst_list.append(instruction(kind, [tok_a, tok_b, tok_c]))

#     # Regexes for instruction shapes
#     measure_re = re.compile(r"^c\s*(\d+)\s*=\s*MEASURE\s+([qs]\d+)\s*$", re.IGNORECASE)
#     # dealloc lines (we'll synthesize syscalls ourselves, but allow them to appear)
#     dealloc_q_re = re.compile(r"^deallocate_data\s*\(\s*q\s*\)\s*$", re.IGNORECASE)
#     dealloc_s_re = re.compile(
#         r"^\s*deallocate_helper\s*\(\s*(s\d+)\s*\)\s*$",
#         re.IGNORECASE
#     )

#     for ln in instr_lines:
#         # Skip optional trailing dealloc directives; we'll add syscalls at the end.
#         if dealloc_q_re.match(ln):
#             continue

#         # Measurement: "c0 = MEASURE q2"
#         m = measure_re.match(ln)
#         if m:
#             measure_n += 1
#             cidx = int(m.group(1))
#             qtok = m.group(2)
#             inst_list.append(instruction(Instype.MEASURE, [qtok], classical_address=cidx))
#             continue

#         # deallocate_helper(s0)
#         mdealloc_s = dealloc_s_re.match(ln)
#         if mdealloc_s:
#             sreg = mdealloc_s.group(1)
#             # synthesize RELEASE instructions for all helper qubits
#             inst_list.append(instruction(Instype.RELEASE, [sreg]))
#             continue


#         # Tokenize: first word is gate mnemonic (maybe with params), the rest are args
#         # We'll manually pull params when present (RX/RY/RZ/U/CU1).
#         # Examples:
#         #   "H q0"
#         #   "RX(1.23) q1"
#         #   "U(θ,φ,λ) q3"
#         #   "CNOT q0, q2"
#         #   "CU1(π/2) q0, q1"
#         parts = ln.split(None, 1)
#         if not parts:
#             continue
#         gate_full = parts[0].strip()
#         rest = parts[1].strip() if len(parts) > 1 else ""
#         gate = gate_full.lower()

#         # Parameterized single-qubit
#         if gate.startswith("rx"):
#             params, rest2 = _grab_params(rest if gate == "rx" else gate_full[2:] + rest)
#             target = rest2
#             add_oneq_gate(Instype.RX, target, params=params)
#             continue

#         if gate.startswith("ry"):
#             params, rest2 = _grab_params(rest if gate == "ry" else gate_full[2:] + rest)
#             target = rest2
#             add_oneq_gate(Instype.RY, target, params=params)
#             continue

#         if gate.startswith("rz"):
#             params, rest2 = _grab_params(rest if gate == "rz" else gate_full[2:] + rest)
#             target = rest2
#             add_oneq_gate(Instype.RZ, target, params=params)
#             continue

#         if gate.startswith("u3"):  # treat like U
#             params, rest2 = _grab_params(rest if gate == "u3" else gate_full[2:] + rest)
#             target = rest2
#             if len(params) != 3:
#                 raise ValueError(f"U3 expects 3 parameters, got {params}")
#             add_oneq_gate(Instype.U3, target, params=params)
#             continue

#         if gate.startswith("u(") or gate == "u":
#             # handle forms like "U( ... ) q0" or weird tokenization
#             params, rest2 = _grab_params(ln[1:] if gate_full.lower().startswith("u(") else rest)
#             target = rest2
#             if len(params) != 3:
#                 raise ValueError(f"U expects 3 parameters, got {params}")
#             add_oneq_gate(Instype.U, target, params=params)
#             continue

#         if gate.startswith("cu1") or gate.startswith("cp"):
#             # Decide how to build the string we feed into _grab_params
#             # Case A: token is just "cu1" or "cp" and params are in `rest`,
#             #         e.g. "CP (theta) q0, q1"
#             if gate in ("cu1", "cp"):
#                 param_src = rest
#             else:
#                 # Case B: token already includes the "(", e.g. "CP(theta)"
#                 #         so slice off the name and keep from "(" onward.
#                 # length of name: 3 for CU1, 2 for CP
#                 name_len = 3 if gate.startswith("cu1") else 2
#                 param_src = gate_full[name_len:] + rest
#                 # e.g. gate_full = "CP(0.5)"  -> gate_full[2:] = "(0.5)"
#                 #      param_src = "(0.5)" + "q0, q1" -> "(0.5)q0, q1"

#             params, rest2 = _grab_params(param_src)

#             if len(params) != 1:
#                 print(param_src)
#                 raise ValueError(f"CU1/CP expects 1 parameter, got {params}")

#             # rest2 should now look like "q0, q1"
#             toks = [t.strip() for t in rest2.split(",")]
#             if len(toks) != 2:
#                 raise ValueError(f"CU1/CP expects two qubit args, got '{rest2}'")
#             add_twoq_gate(Instype.CP, toks[0], toks[1], params=params)
#             continue

#         # No-parameter one-qubit?
#         if gate in oneq_no_param:
#             # rest should be like "q0"
#             add_oneq_gate(oneq_no_param[gate], rest)
#             continue

#         # Two-qubit no-param?
#         if gate in twoq_no_param:
#             toks = [t.strip() for t in rest.split(",")]
#             if len(toks) != 2:
#                 raise ValueError(f"{gate.upper()} expects two qubit args, got '{rest}'")
#             add_twoq_gate(twoq_no_param[gate], toks[0], toks[1])
#             continue

#         # Three-qubit no-param?
#         if gate in threeq_no_param:
#             toks = [t.strip() for t in rest.split(",")]
#             if len(toks) != 3:
#                 raise ValueError(f"{gate.upper()} expects three qubit args, got '{rest}'")
#             add_threeq_gate(threeq_no_param[gate], toks[0], toks[1], toks[2])
#             continue

#         raise ValueError(f"Unsupported or malformed instruction line: '{ln}'")


#     return inst_list, data_n, syn_n, measure_n    




def parse_program_from_file(file_path: str) -> Tuple[List[instruction], int, int, int]:
    """
    Parse the custom process program DSL from a file into a `process` object.
    Return (inst_list, data_n, syn_n, measure_n).
    """
    with open(file_path, 'r') as file:
        qasm_code = file.read()
    return parse_program_from_string(qasm_code)





def toffoli_decomposition(qubit1: str, qubit2: str, qubit3: str) -> Tuple[List[instruction], str]:
    """
    We construct the Toffoli gate by decomposing it into a series of CNOT and single-qubit gates:
    

    CCX(a, b, c) = H(c) CNOT(b, c) Tdg(c) CNOT(a, c) T(c) CNOT(b, c) Tdg(c) CNOT(a, c) T(b) T(c) H(c) CNOT(a, b) T(a) Tdg(b) CNOT(a, b) 
    """
    program_string = ""
    inst_list = []
    inst_list.append(instruction(Instype.H, [qubit3]))
    program_string += f"H {qubit3}\n"
    inst_list.append(instruction(Instype.CNOT, [qubit2, qubit3]))
    program_string += f"CNOT {qubit2}, {qubit3}\n"
    inst_list.append(instruction(Instype.Tdg, [qubit3]))
    program_string += f"Tdg {qubit3}\n"
    inst_list.append(instruction(Instype.CNOT, [qubit1, qubit3]))
    program_string += f"CNOT {qubit1}, {qubit3}\n"
    inst_list.append(instruction(Instype.T, [qubit3]))
    program_string += f"T {qubit3}\n"
    inst_list.append(instruction(Instype.CNOT, [qubit2, qubit3]))
    program_string += f"CNOT {qubit2}, {qubit3}\n"
    inst_list.append(instruction(Instype.Tdg, [qubit3]))
    program_string += f"Tdg {qubit3}\n"
    inst_list.append(instruction(Instype.CNOT, [qubit1, qubit3]))
    program_string += f"CNOT {qubit1}, {qubit3}\n"
    inst_list.append(instruction(Instype.T, [qubit2]))
    program_string += f"T {qubit2}\n"
    inst_list.append(instruction(Instype.T, [qubit3]))
    program_string += f"T {qubit3}\n"
    inst_list.append(instruction(Instype.H, [qubit3]))
    program_string += f"H {qubit3}\n"
    inst_list.append(instruction(Instype.CNOT, [qubit1, qubit2]))
    program_string += f"CNOT {qubit1}, {qubit2}\n"
    inst_list.append(instruction(Instype.T, [qubit1]))
    program_string += f"T {qubit1}\n"
    inst_list.append(instruction(Instype.Tdg, [qubit2]))
    program_string += f"Tdg {qubit2}\n"
    inst_list.append(instruction(Instype.CNOT, [qubit1, qubit2]))
    program_string += f"CNOT {qubit1}, {qubit2}\n"
    return inst_list, program_string    







def generate_random_boolean_function(num_vars: int, num_gates: int, seed: int=None) -> Tuple[List[instruction], str]:
    pass





def generate_LCU_program(num_qubits: int, ham_str_list:List[str],num_terms: int, seed: int=None) -> Tuple[List[instruction], str]:
    pass










def generate_program_without_helper_qubit(data_n, syn_n, original_inst_list: List[instruction]) -> str:
    """
    Generate a new program without any helper qubits.
    All helper qubits are re-labeled as data qubits.

    
    For example, input program:

    q = alloc_data(3)
    s = alloc_helper(2)
    set_shot(1000)
    H s0
    CNOT q0, s0
    CNOT q1, s0
    H s0
    c0 = MEASURE s0
    deallocate_helper(s0)
    H s1
    CNOT q1, s1
    CNOT q2, s1
    H s1
    c1 = MEASURE s1
    deallocate_helper(s1)
    deallocate_data(q)

    Output program:

    q = alloc_data(5)
    set_shot(1000)
    H q3
    CNOT q0, q3
    CNOT q1, q3
    H q3
    c0 = MEASURE q3
    H q4
    CNOT q1, q4
    CNOT q2, q4
    H q4
    c1 = MEASURE q4
    deallocate_data(q)

    """
    output_program="q = alloc_data("+str(data_n+syn_n)+")\n"

    for inst in  original_inst_list:
        if inst.is_system_call():
            continue
        addresses = inst.get_qubitaddress()
        qiskitaddress=[]
        for addr in addresses:
            if addr.startswith('s'):
                qiskitaddress.append('q'+str(int(addr[1:])+data_n))
            else:
                qiskitaddress.append(addr)
        match inst.get_type():
            case Instype.H:
                output_program+="H "+qiskitaddress[0]+"\n"
            case Instype.X:
                output_program+="X "+qiskitaddress[0]+"\n"
            case Instype.Y:
                output_program+="Y "+qiskitaddress[0]+"\n"
            case Instype.Z:
                output_program+="Z "+qiskitaddress[0]+"\n"
            case Instype.T:
                output_program+="T "+qiskitaddress[0]+"\n"
            case Instype.Tdg:
                output_program+="Tdg "+qiskitaddress[0]+"\n"
            case Instype.S:
                output_program+="S "+qiskitaddress[0]+"\n"
            case Instype.Sdg:
                output_program+="Sdg "+qiskitaddress[0]+"\n"
            case Instype.SX:
                output_program+="SX "+qiskitaddress[0]+"\n"
            case Instype.RZ:
                params=inst.get_params()
                output_program+="RZ("+str(params[0])+") "+qiskitaddress[0]+"\n"
            case Instype.RX:
                params=inst.get_params()
                output_program+="RX("+str(params[0])+") "+qiskitaddress[0]+"\n"
            case Instype.RY:
                params=inst.get_params()
                output_program+="RY("+str(params[0])+") "+qiskitaddress[0]+"\n"
            case Instype.U3:
                params=inst.get_params()
                output_program+="U3("+str(params[0])+", "+str(params[1])+", "+str(params[2])+") "+qiskitaddress[0]+"\n"
            case Instype.U:
                params=inst.get_params()
                output_program+="U("+str(params[0])+", "+str(params[1])+", "+str(params[2])+") "+qiskitaddress[0]+"\n"
            case Instype.Toffoli:
                output_program+="Toffoli "+qiskitaddress[0]+", "+qiskitaddress[1]+", "+qiskitaddress[2]+"\n"
            case Instype.CNOT:
                output_program+="CNOT "+qiskitaddress[0]+", "+qiskitaddress[1]+"\n"
            case Instype.CH:
                output_program+="CH "+qiskitaddress[0]+", "+qiskitaddress[1]+"\n"
            case Instype.SWAP:
                output_program+="SWAP "+qiskitaddress[0]+", "+qiskitaddress[1]+"\n"
            case Instype.CSWAP:
                output_program+="CSWAP "+qiskitaddress[0]+", "+qiskitaddress[1]+", "+qiskitaddress[2]+"\n"
            case Instype.CP:
                params=inst.get_params()
                output_program+="CP "+str(params[0])+", "+qiskitaddress[0]+", "+qiskitaddress[1]+"\n"
            case Instype.RESET:
                output_program+="RESET "+qiskitaddress[0]+"\n"
            case Instype.RELEASE:
                continue
            case Instype.MEASURE:
                classical_address=inst.get_classical_address()
                output_program+="c"+str(classical_address)+" = MEASURE "+qiskitaddress[0]+"\n"

    output_program+="deallocate_data(q)\n"
    return output_program











classical_logic_small_benchmark={
    0: "varnum_3_d_3_0",
    1: "varnum_3_d_3_1",
    2: "varnum_3_d_4_0",
    3: "varnum_3_d_4_1",
    4: "varnum_4_d_2_0",
    5: "varnum_4_d_2_1",
    6: "varnum_4_d_3_0",
    7: "varnum_4_d_3_1",
    8: "varnum_4_d_4_0",
    9: "varnum_4_d_4_1"
}




classical_logic_medium_benchmark = {
    0:  "varnum_7_d_5_0",
    1:  "varnum_7_d_5_1",
    2:  "varnum_8_d_4_0",
    3:  "varnum_8_d_5_0",
    4:  "varnum_9_d_4_0",
    5:  "varnum_10_d_3_0",
    6:  "varnum_10_d_3_1",
    7:  "varnum_10_d_4_0",
    8:  "varnum_11_d_4_0",
    9:  "varnum_11_d_4_1",
    10: "varnum_12_d_4_0",
    11: "varnum_12_d_4_1",
}



def rewrite_benchmark_program(benchmarkname: str,benchmark_suit: dict):
    """
    Rewrite the benchmark program with the given ID to remove helper qubits.
    Store the file to the benchmarkdata directory with the same name.
    """

    for benchmark_id in benchmark_suit.keys():
        file_path = f"C:\\Users\\yezhu\\Documents\\HALO\\benchmark\\{benchmarkname}\\{benchmark_suit[benchmark_id]}"

        inst_list, data_n, syn_n, measure_n = parse_program_from_file(file_path)

        output_program = generate_program_without_helper_qubit(data_n, syn_n, inst_list)

        output_file_path = f"C:\\Users\\yezhu\\Documents\\HALO\\benchmark\\{benchmarkname}\\nohelper\\{benchmark_suit[benchmark_id]}"

        with open(output_file_path, "w") as f:
            f.write(output_program)





from pathlib import Path


def simulate_benchmark_program(source_file_folder: str, is_clifford: bool=False):
    """
    For all the benchmark programs in the given folder,

    Store the simulation result in the same folder with the name 'source_file_name'_counts.pkl
    """
    
    folder = Path(source_file_folder)
    parent = folder.parent
    folder_name = folder.name   
    shots = 2000

    # Construct destination folder:
    result_folder = parent / "result2000shots" / folder_name
    result_folder.mkdir(parents=True, exist_ok=True)

    for program_path in folder.glob('*'):
        if program_path.is_file():
            inst_list, data_n, syn_n, measure_n = parse_program_from_file(str(program_path))

            print(f"Simulating program: {program_path.stem} with data_n={data_n}, syn_n={syn_n}, measure_n={measure_n}")

            qiskit_circuit = construct_qiskit_circuit(data_n, syn_n, measure_n, inst_list)

            if is_clifford:
                sim = AerSimulator(method="stabilizer")
            else:
                sim = AerSimulator()
            tqc = transpile(qiskit_circuit, sim)

            # Run with 1000 shots
            result = sim.run(tqc, shots=shots).result()
            counts = result.get_counts(tqc)


            print(f"Result: {counts}")
            output_file_path = result_folder/ f"{program_path.stem}_counts.pkl"

            with open(output_file_path, "wb") as f:
                pickle.dump(counts, f)






def test_no_helper_program(folder_path: str):
    """
    Test the program without helper qubits.
    Simulate these program, verify the correctness by comparing the result with the expected result.
    """

    file_path = "C:\\Users\\yezhu\\Documents\\HALO\\benchmark\\syndrome_extraction_surface_n4_nohelper"


    inst_list, data_n, syn_n, measure_n = parse_program_from_file(file_path)



    qiskit_circuit = construct_qiskit_circuit(data_n, syn_n, measure_n, inst_list)


    shots = 2000

    sim = AerSimulator()
    tqc = transpile(qiskit_circuit, sim)

    # Run with 1000 shots
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts(tqc)

    print(counts)















if __name__ == "__main__":

    # for i in range(2,8):
    #     generate_mcx_benchmark_small(i)


    # for i in range(9,15):
    #     generate_mcx_benchmark_medium(i)


    
    benchmarkname="arithmedium"
    rewrite_benchmark_program(benchmarkname, classical_logic_medium_benchmark)



    # simulate_benchmark_program("C:\\Users\\yezhu\\Documents\\HALO\\benchmark\\arithsmall")

    # simulate_benchmark_program("C:\\Users\\yezhu\\Documents\\HALO\\benchmark\\arithmedium")




    #generate_qec_benchmark_medium()
    # generate_random_benchmark_small(data_n=4, syn_n=4, gate_count=10, label=0, seed=42)
    # generate_random_benchmark_small(data_n=4, syn_n=4, gate_count=15, label=0, seed=42)
    # generate_random_benchmark_small(data_n=5, syn_n=5, gate_count=15, label=0, seed=42)
    # generate_random_benchmark_small(data_n=5, syn_n=5, gate_count=20, label=0, seed=42)
    # generate_random_benchmark_small(data_n=5, syn_n=4, gate_count=15, label=0, seed=42)
    # generate_random_benchmark_small(data_n=6, syn_n=6, gate_count=15, label=0, seed=42)
    # generate_random_benchmark_small(data_n=6, syn_n=6, gate_count=18, label=0, seed=42)
    # generate_random_benchmark_small(data_n=7, syn_n=7, gate_count=25, label=0, seed=42)
    # generate_random_benchmark_small(data_n=7, syn_n=7, gate_count=10, label=0, seed=42)
    # generate_random_benchmark_small(data_n=8, syn_n=8, gate_count=22, label=0, seed=42)
    # generate_random_benchmark_small(data_n=9, syn_n=8, gate_count=20, label=0, seed=42)



    # generate_random_benchmark_medium(data_n=15, syn_n=15, gate_count=30, label=0, seed=42)
    # generate_random_benchmark_medium(data_n=15, syn_n=15, gate_count=40, label=0, seed=42)
    # generate_random_benchmark_medium(data_n=10, syn_n=20, gate_count=45, label=0, seed=42)
    # generate_random_benchmark_medium(data_n=20, syn_n=10, gate_count=50, label=0, seed=42)
    # generate_random_benchmark_medium(data_n=14, syn_n=15, gate_count=45, label=0, seed=42)
    # generate_random_benchmark_medium(data_n=12, syn_n=18, gate_count=58, label=0, seed=42)
    # generate_random_benchmark_medium(data_n=11, syn_n=19, gate_count=48, label=0, seed=42)
    # generate_random_benchmark_medium(data_n=10, syn_n=20, gate_count=55, label=0, seed=42)
    # generate_random_benchmark_medium(data_n=15, syn_n=12, gate_count=40, label=0, seed=42)
    # generate_random_benchmark_medium(data_n=13, syn_n=17, gate_count=52, label=0, seed=42)
    # generate_random_benchmark_medium(data_n=17, syn_n=13, gate_count=50, label=0, seed=42)

    #rewrite_benchmark_program()

    # inst_list, prog_str=generate_random_program(data_n=5, syn_n=5, gate_count=20, seed=42)
    # for inst in inst_list:
    #     print(inst)

    # print("----------------------------------------------------------------------------")
    # print(prog_str)

    # new_inst_list, data_n, syn_n, measure_n = parse_program_from_string(prog_str)


    # for inst in new_inst_list:
    #     print(inst)

    # inst_list, program_string = generate_stabilizer_measurement_circuit(5, ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"], round=2)

    # print(program_string)

    # inst_list, program_string = generate_multi_controlled_x(4, decompose_toffoli=True)

    # print(program_string)



#    file_path = "C:\\Users\\yezhu\\Documents\\HALO\\benchmark\\randomsmall\\data_4_syn_4_gc_10_0"

#    inst_list, data_n, syn_n, measure_n = parse_program_from_file(file_path)



#    print(f"data_n={data_n}, syn_n={syn_n}, measure_n={measure_n}")
   
#    print("Original Program:")
#    for inst in inst_list:
#     print(inst)


#    output_program = generate_program_without_helper_qubit(data_n, syn_n, inst_list)


#    print(output_program)




#    qiskit_circuit = construct_qiskit_circuit(data_n, syn_n, measure_n, inst_list)


#    shots = 2000

#    sim = AerSimulator()
#    tqc = transpile(qiskit_circuit, sim)

#    # Run with 1000 shots
#    result = sim.run(tqc, shots=shots).result()
#    counts = result.get_counts(tqc)

#    with open("C://Users//yezhu//Documents//HALO//benchmark//result2000shots//syndrome_extraction_surface_n4_counts.pkl", "wb") as f:
#      pickle.dump(counts, f)


   #print(simulate_result)
   #parse_qasm_instruction(0,qasm_code)
