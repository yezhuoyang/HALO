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
                qiskit_circuit.reset(qiskitaddress[0])
            case Instype.RELEASE:
                qiskit_circuit.reset(qiskitaddress[0])
            case Instype.MEASURE:
                scheduled_classical_address=inst.get_scheduled_classical_address()
                qiskit_circuit.measure(qiskitaddress[0], scheduled_classical_address)

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


def _grab_params(text_after_gate: str) -> Tuple[List[float], str]:
    """
    Given '(<params>) rest', return (params_list, rest_without_paren_prefix).
    If no parentheses, returns ([], original_text).
    """
    m = _param_list_re.match(text_after_gate.strip())
    if not m:
        return [], text_after_gate.strip()
    params = _parse_float_list(m.group(1))
    # remove the '(...)' prefix from the remaining text
    remainder = text_after_gate.strip()[m.end():].strip()
    return params, remainder



def parse_program_from_string(qasm_code: str) -> Tuple[List[instruction], int, int, int]:
    """
    Parse the custom process program DSL into a `process` object.
    Return (inst_list, data_n, syn_n,measure_n).

    Expects header lines:
        q = alloc_data(N)
        s = alloc_helper(M)
        set_shot(S)

    Supports gates like:
        H q0
        CNOT q0, q2
        RX(1.234) q3
        U(theta, phi, lam) q1
        ...
        c0 = MEASURE q2
        deallocate_data(q)
        deallocate_helper(s)
    """
    # normalize and split lines
    raw_lines = [ln.strip() for ln in qasm_code.strip().splitlines() if ln.strip()]

    # ---- Pass 1: read header (alloc & shots) ----
    data_n = 0
    syn_n = 0
    measure_n = 0

    alloc_data_re = re.compile(r"^q\s*=\s*alloc_data\(\s*(\d+)\s*\)\s*$", re.IGNORECASE)
    alloc_syn_re  = re.compile(r"^s\s*=\s*alloc_helper\(\s*(\d+)\s*\)\s*$", re.IGNORECASE)
    set_shot_re   = re.compile(r"^set_shot\(\s*(\d+)\s*\)\s*$", re.IGNORECASE)

    # We’ll keep the non-header lines to parse as instructions
    instr_lines: List[str] = []

    for ln in raw_lines:
        if m := alloc_data_re.match(ln):
            data_n = int(m.group(1))
            continue
        if m := alloc_syn_re.match(ln):
            syn_n = int(m.group(1))
            continue
        if m := set_shot_re.match(ln):
            shots = int(m.group(1))
            continue
        # not a header line → it’s an instruction or dealloc
        instr_lines.append(ln)


    inst_list=[]


    # ---- Parsers for instruction lines ----
    # Simple 1-qubit no-parameter gates
    oneq_no_param = {
        "h": Instype.H,
        "x": Instype.X,
        "y": Instype.Y,
        "z": Instype.Z,
        "t": Instype.T,
        "tdg": Instype.Tdg,
        "s": Instype.S,
        "sdg": Instype.Sdg,
        "sx": Instype.SX,
        "reset": Instype.RESET,
    }

    # Two-qubit gates without params
    twoq_no_param = {
        "cnot": Instype.CNOT,
        "cx": Instype.CNOT,   # alias
        "ch": Instype.CH,
        "swap": Instype.SWAP,
    }

    # Three-qubit no-param
    threeq_no_param = {
        "cswap": Instype.CSWAP,
        "toffoli": Instype.Toffoli,
        "ccx": Instype.Toffoli,  # alias
    }

    # Helpers
    def add_oneq_gate(kind: Instype, addr_tok: str, params: Optional[List[float]] = None):
        if params:
            inst_list.append(instruction(kind, [addr_tok], params=params))
        else:
            inst_list.append(instruction(kind, [addr_tok]))

    def add_twoq_gate(kind: Instype, tok_a: str, tok_b: str, params: Optional[List[float]] = None):
        if params:
            inst_list.append(instruction(kind, [tok_a, tok_b], params=params))
        else:
            inst_list.append(instruction(kind, [tok_a, tok_b]))

    def add_threeq_gate(kind: Instype, tok_a: str, tok_b: str, tok_c: str):
        inst_list.append(instruction(kind, [tok_a, tok_b, tok_c]))

    # Regexes for instruction shapes
    measure_re = re.compile(r"^c\s*(\d+)\s*=\s*MEASURE\s+([qs]\d+)\s*$", re.IGNORECASE)
    # dealloc lines (we'll synthesize syscalls ourselves, but allow them to appear)
    dealloc_q_re = re.compile(r"^deallocate_data\s*\(\s*q\s*\)\s*$", re.IGNORECASE)
    dealloc_s_re = re.compile(
        r"^\s*deallocate_helper\s*\(\s*(s\d+)\s*\)\s*$",
        re.IGNORECASE
    )

    for ln in instr_lines:
        # Skip optional trailing dealloc directives; we'll add syscalls at the end.
        if dealloc_q_re.match(ln):
            continue

        # Measurement: "c0 = MEASURE q2"
        m = measure_re.match(ln)
        if m:
            measure_n += 1
            cidx = int(m.group(1))
            qtok = m.group(2)
            inst_list.append(instruction(Instype.MEASURE, [qtok], classical_address=cidx))
            continue

        # deallocate_helper(s0)
        mdealloc_s = dealloc_s_re.match(ln)
        if mdealloc_s:
            sreg = mdealloc_s.group(1)
            # synthesize RELEASE instructions for all helper qubits
            inst_list.append(instruction(Instype.RELEASE, [sreg]))
            continue


        # Tokenize: first word is gate mnemonic (maybe with params), the rest are args
        # We'll manually pull params when present (RX/RY/RZ/U/CU1).
        # Examples:
        #   "H q0"
        #   "RX(1.23) q1"
        #   "U(θ,φ,λ) q3"
        #   "CNOT q0, q2"
        #   "CU1(π/2) q0, q1"
        parts = ln.split(None, 1)
        if not parts:
            continue
        gate_full = parts[0].strip()
        rest = parts[1].strip() if len(parts) > 1 else ""
        gate = gate_full.lower()

        # Parameterized single-qubit
        if gate.startswith("rx"):
            params, rest2 = _grab_params(rest if gate == "rx" else gate_full[2:] + rest)
            target = rest2
            add_oneq_gate(Instype.RX, target, params=params)
            continue

        if gate.startswith("ry"):
            params, rest2 = _grab_params(rest if gate == "ry" else gate_full[2:] + rest)
            target = rest2
            add_oneq_gate(Instype.RY, target, params=params)
            continue

        if gate.startswith("rz"):
            params, rest2 = _grab_params(rest if gate == "rz" else gate_full[2:] + rest)
            target = rest2
            add_oneq_gate(Instype.RZ, target, params=params)
            continue

        if gate.startswith("u3"):  # treat like U
            params, rest2 = _grab_params(rest if gate == "u3" else gate_full[2:] + rest)
            target = rest2
            if len(params) != 3:
                raise ValueError(f"U3 expects 3 parameters, got {params}")
            add_oneq_gate(Instype.U3, target, params=params)
            continue

        if gate.startswith("u(") or gate == "u":
            # handle forms like "U( ... ) q0" or weird tokenization
            params, rest2 = _grab_params(ln[1:] if gate_full.lower().startswith("u(") else rest)
            target = rest2
            if len(params) != 3:
                raise ValueError(f"U expects 3 parameters, got {params}")
            add_oneq_gate(Instype.U, target, params=params)
            continue

        if gate.startswith("cu1") or gate.startswith("cp"):
            # Decide how to build the string we feed into _grab_params
            # Case A: token is just "cu1" or "cp" and params are in `rest`,
            #         e.g. "CP (theta) q0, q1"
            if gate in ("cu1", "cp"):
                param_src = rest
            else:
                # Case B: token already includes the "(", e.g. "CP(theta)"
                #         so slice off the name and keep from "(" onward.
                # length of name: 3 for CU1, 2 for CP
                name_len = 3 if gate.startswith("cu1") else 2
                param_src = gate_full[name_len:] + rest
                # e.g. gate_full = "CP(0.5)"  -> gate_full[2:] = "(0.5)"
                #      param_src = "(0.5)" + "q0, q1" -> "(0.5)q0, q1"

            params, rest2 = _grab_params(param_src)

            if len(params) != 1:
                raise ValueError(f"CU1/CP expects 1 parameter, got {params}")

            # rest2 should now look like "q0, q1"
            toks = [t.strip() for t in rest2.split(",")]
            if len(toks) != 2:
                raise ValueError(f"CU1/CP expects two qubit args, got '{rest2}'")
            add_twoq_gate(Instype.CP, toks[0], toks[1], params=params)
            continue

        # No-parameter one-qubit?
        if gate in oneq_no_param:
            # rest should be like "q0"
            add_oneq_gate(oneq_no_param[gate], rest)
            continue

        # Two-qubit no-param?
        if gate in twoq_no_param:
            toks = [t.strip() for t in rest.split(",")]
            if len(toks) != 2:
                raise ValueError(f"{gate.upper()} expects two qubit args, got '{rest}'")
            add_twoq_gate(twoq_no_param[gate], toks[0], toks[1])
            continue

        # Three-qubit no-param?
        if gate in threeq_no_param:
            toks = [t.strip() for t in rest.split(",")]
            if len(toks) != 3:
                raise ValueError(f"{gate.upper()} expects three qubit args, got '{rest}'")
            add_threeq_gate(threeq_no_param[gate], toks[0], toks[1], toks[2])
            continue

        raise ValueError(f"Unsupported or malformed instruction line: '{ln}'")


    return inst_list, data_n, syn_n, measure_n    




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



def generate_multi_controlled_x(num_controls: int, decompose_toffoli: bool) -> Tuple[List[instruction], str]:
    """
    Generate a multi-controlled X gate using ancilla qubits.
    Returns a list of instructions implementing the multi-controlled X gate.
    """
    program_string = ""
    inst_list = []
    program_string += f"q = alloc_data({num_controls+1})\n"
    program_string += f"s = alloc_helper({num_controls-1})\n"

    if decompose_toffoli:
        toffoli_insts, toffoli_str = toffoli_decomposition(f"q0", f"q1", f"s0")
        inst_list.extend(toffoli_insts)
        program_string += toffoli_str
    else:
        inst_list.append(instruction(Instype.Toffoli, [f"q0", f"q1", f"s0"]))
        program_string += f"Toffoli q0, q1, s0\n"

    for i in range(2, num_controls):
        if decompose_toffoli:
            toffoli_insts, toffoli_str = toffoli_decomposition(f"q{i}", f"s{i-2}", f"s{i-1}")
            inst_list.extend(toffoli_insts)
            program_string += toffoli_str
        else:
            inst_list.append(instruction(Instype.Toffoli, [f"q{i}", f"s{i-2}", f"s{i-1}"]))
            program_string += f"Toffoli q{i}, s{i-2}, s{i-1}\n"

    inst_list.append(instruction(Instype.CNOT, [f"s{num_controls-2}", f"q{num_controls}"]))
    program_string += f"CNOT s{num_controls-2}, q{num_controls}\n"

    #Uncompute
    for i in range(num_controls-1, 1, -1):
        if decompose_toffoli:
            toffoli_insts, toffoli_str = toffoli_decomposition(f"q{i}", f"s{i-2}", f"s{i-1}")
            inst_list.extend(toffoli_insts)
            program_string += toffoli_str
        else:
            inst_list.append(instruction(Instype.Toffoli, [f"q{i}", f"s{i-2}", f"s{i-1}"]))
            program_string += f"Toffoli q{i}, s{i-2}, s{i-1}\n"
        #Release helper qubit
        inst_list.append(instruction(Instype.RELEASE, [f"s{i-1}"]))
        program_string += f"deallocate_helper(s{i-1})\n"

    if decompose_toffoli:
        toffoli_insts, toffoli_str = toffoli_decomposition(f"q0", f"q1", f"s0")
        inst_list.extend(toffoli_insts)
        program_string += toffoli_str
    else:
        inst_list.append(instruction(Instype.Toffoli, [f"q0", f"q1", f"s0"]))
        program_string += f"Toffoli q0, q1, s0\n"
    #Release helper qubit
    inst_list.append(instruction(Instype.RELEASE, [f"s0"]))
    program_string += f"deallocate_helper(s0)\n"


    program_string += f"deallocate_data(q)\n"
    return inst_list, program_string




def generate_stabilizer_measurement_circuit(n_data:int,stabilizers:List[str], round: int) -> Tuple[List[instruction], str]:
    """
    Provided a list of stabilizers in string format (e.g., "XZZXI"),
    generate the corresponding stabilizer measurement circuit which measure round `round`.
    """
    program_string = ""
    inst_list = []
    program_string += f"q = alloc_data({n_data})\n"
    program_string += f"s = alloc_helper({round*len(stabilizers)})\n"

    for stab_index, stabilizer in enumerate(stabilizers):
        ancilla_qubit = f"s{stab_index + round*len(stabilizers)}"


        inst_list.append(instruction(Instype.H, [ancilla_qubit]))
        program_string += f"H {ancilla_qubit}\n"

        for data_index, pauli in enumerate(stabilizer):
            data_qubit = f"q{data_index}"
            match pauli:
                case 'X':
                    inst_list.append(instruction(Instype.CNOT, [data_qubit, ancilla_qubit]))
                    program_string += f"CNOT {ancilla_qubit}, {data_qubit}\n"
                case 'Z':
                    inst_list.append(instruction(Instype.H, [data_qubit]))
                    program_string += f"H {data_qubit}\n"
                    inst_list.append(instruction(Instype.CNOT, [data_qubit, ancilla_qubit]))
                    program_string += f"CNOT {ancilla_qubit}, {data_qubit}\n"
                    inst_list.append(instruction(Instype.H, [data_qubit]))
                    program_string += f"H {data_qubit}\n"
                case 'I':
                    continue
                case _:
                    raise ValueError(f"Invalid Pauli operator '{pauli}' in stabilizer '{stabilizer}'")

        # Final Hadamard on ancilla qubit before measurement
        inst_list.append(instruction(Instype.H, [ancilla_qubit]))
        program_string += f"H {ancilla_qubit}\n"

        # Measure the ancilla qubit
        classical_address = stab_index + round * len(stabilizers)
        inst_list.append(instruction(Instype.MEASURE, [ancilla_qubit], classical_address=classical_address))
        program_string += f"c{classical_address} = MEASURE {ancilla_qubit}\n"


        #Release the used ancilla qubit
        inst_list.append(instruction(Instype.RELEASE, [ancilla_qubit]))
        program_string += f"deallocate_helper({ancilla_qubit})\n"

    program_string += f"deallocate_data(q)\n"
    return inst_list, program_string






def generate_random_boolean_function(num_vars: int, num_gates: int, seed: int=None) -> Tuple[List[instruction], str]:
    pass







def generate_random_program(data_n: int, syn_n: int, gate_count: int,seed: int=None) -> Tuple[List[instruction], str]:
    """
    Generate a completely random program with the given parameters.
    """
    if seed is not None:
        random.seed(seed)

    program_string = ""
    inst_list = []
    program_string += f"q = alloc_data({data_n})\n"
    program_string += f"s = alloc_helper({syn_n})\n"



    """
    We maintain a set of active helper qubits.
    When we release a helper qubit, we remove it from the set.
    """
    active_helper_qubits = set()
    for i in range(syn_n):
        active_helper_qubits.add(f"s{i}")
    active_data_qubits = set()
    for i in range(data_n):
        active_data_qubits.add(f"q{i}")

    forbidden = {Instype.RESET,
                Instype.CSWAP, Instype.Toffoli, Instype.RELEASE}
    allowed = [g for g in Instype if g not in forbidden]



    #Randomize the location of release instructions
    release_locations = random.sample(range(gate_count)[2:], k=syn_n)
    current_gate_count = 0

    current_measure_index = 0
    for _ in range(gate_count):
        # Randomly choose gate type and qubits

        if current_gate_count in release_locations and active_helper_qubits:
            random_qubit = random.choice(list(active_helper_qubits))
            inst = instruction(Instype.MEASURE, [random_qubit], classical_address=current_measure_index)
            inst_list.append(inst)
            program_string += f"c{current_measure_index} = MEASURE {random_qubit}\n"
            current_measure_index += 1
            inst = instruction(Instype.RELEASE, [random_qubit])
            active_helper_qubits.remove(random_qubit)
            program_string += f"deallocate_helper({random_qubit})\n"
            inst_list.append(inst)
            current_gate_count += 1
            continue

        gate_type = random.choice(allowed)
        """
        Randomly generate two qubit,
        either qk or sk, k is the qubit index
        We can only choose from active helper qubits
        """
        if active_data_qubits==set() and active_helper_qubits==set():
            break


        match gate_type:
            case Instype.H | Instype.X | Instype.Y | Instype.Z | Instype.T | Instype.Tdg | Instype.S | Instype.Sdg | Instype.SX:
                random_qubit = random.choice(list(active_data_qubits)+list(active_helper_qubits))
                inst = instruction(gate_type, [random_qubit])          
                inst_list.append(inst)      
                program_string += f"{get_gate_type_name(gate_type)} {random_qubit}\n"
            case Instype.RZ | Instype.RX | Instype.RY:
                random_qubit = random.choice(list(active_data_qubits)+list(active_helper_qubits))
                angle = random.uniform(0, 2 * 3.141592653589793)
                inst = instruction(gate_type, [random_qubit], params=[angle])
                inst_list.append(inst)
                program_string += f"{get_gate_type_name(gate_type)}({angle}) {random_qubit} \n"
            case Instype.U3 | Instype.U:
                random_qubit = random.choice(list(active_data_qubits)+list(active_helper_qubits))
                angles = [random.uniform(0, 2 * 3.141592653589793) for _ in range(3)]
                inst = instruction(gate_type, [random_qubit], params=angles)
                inst_list.append(inst)
                program_string += f"{get_gate_type_name(gate_type)}({angles[0]}, {angles[1]}, {angles[2]}) {random_qubit} \n"
            case Instype.CNOT | Instype.CH | Instype.SWAP:
                qubits = random.sample(list(active_data_qubits)+list(active_helper_qubits), k=2)
                inst = instruction(gate_type, qubits)
                inst_list.append(inst)
                program_string += f"{get_gate_type_name(gate_type)} {qubits[0]}, {qubits[1]}\n"
            case Instype.CP:
                qubits = random.sample(list(active_data_qubits)+list(active_helper_qubits), k=2)
                angle = random.uniform(0, 2 * 3.141592653589793)
                inst = instruction(gate_type, qubits, params=[angle])
                inst_list.append(inst)
                program_string += f"{get_gate_type_name(gate_type)}({angle}) {qubits[0]}, {qubits[1]}\n"
            case Instype.MEASURE:
                random_qubit = random.choice(list(active_data_qubits)+list(active_helper_qubits))
                inst = instruction(gate_type, [random_qubit], classical_address=current_measure_index)
                inst_list.append(inst)
                program_string += f"c{current_measure_index} = MEASURE {random_qubit}\n"
                current_measure_index += 1
                # Remove the measured qubit from active sets
                if random_qubit.startswith('q'):
                    active_data_qubits.remove(random_qubit)
                else:
                    active_helper_qubits.remove(random_qubit)
                    #Add a release instruction
                    inst = instruction(Instype.RELEASE, [random_qubit])
                    inst_list.append(inst)
                    program_string += f"deallocate_helper({random_qubit})\n"


        current_gate_count += 1

    program_string += "deallocate_data(q)\n"
    return inst_list, program_string



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















benchmark_suit={
    0:"cat_state_prep_n4",
    1:"cat_state_verification_n4",
    2:"repetition_code_distance3_n3",
    3:"shor_parity_measurement_n4",
    4:"shor_stabilizer_XZZX_n3",
    5:"shor_stabilizer_ZZZZ_n4",
    6:"syndrome_extraction_surface_n4"
}



def rewrite_benchmark_program():
    """
    Rewrite the benchmark program with the given ID to remove helper qubits.
    Store the file to the benchmarkdata directory with the same name.
    """

    for benchmark_id in benchmark_suit.keys():
        file_path = f"C:\\Users\\yezhu\\Documents\\HALO\\benchmark\\{benchmark_suit[benchmark_id]}"

        inst_list, data_n, syn_n, measure_n = parse_program_from_file(file_path)

        output_program = generate_program_without_helper_qubit(data_n, syn_n, inst_list)

        output_file_path = f"C:\\Users\\yezhu\\Documents\\HALO\\benchmarkdata\\{benchmark_suit[benchmark_id]}"

        with open(output_file_path, "w") as f:
            f.write(output_program)





if __name__ == "__main__":
    #rewrite_benchmark_program()

    # inst_list, prog_str=generate_random_program(data_n=5, syn_n=5, gate_count=20, seed=42)
    # for inst in inst_list:
    #     print(inst)

    # print("----------------------------------------------------------------------------")
    # print(prog_str)

    # new_inst_list, data_n, syn_n, measure_n = parse_program_from_string(prog_str)


    # for inst in new_inst_list:
    #     print(inst)

    inst_list, program_string = generate_stabilizer_measurement_circuit(5, ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"], round=2)

    print(program_string)

    # inst_list, program_string = generate_multi_controlled_x(4, decompose_toffoli=True)

    # print(program_string)



#    file_path = "C:\\Users\\yezhu\\Documents\\HALO\\benchmark\\syndrome_extraction_surface_n4"




#    inst_list, data_n, syn_n, measure_n = parse_program_from_file(file_path)



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
