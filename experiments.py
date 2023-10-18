import numpy as np
import qiskit.quantum_info as qi
import random

from qiskit.quantum_info.operators import Operator
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_experiments.library import LocalReadoutError
from qiskit_aer.noise import (
    NoiseModel, 
    QuantumError, 
    ReadoutError,
    pauli_error, 
    depolarizing_error, 
    thermal_relaxation_error
)
from qiskit import *


I = np.identity(2)
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
Paulis = {0:I,1:Z,2:X,3:X@Z}

def counts2probs(p,shots):
    return {k: v/shots for k,v in p.items()}

def R2P(x):
    return abs(x), np.angle(x)

def experiment1(gate):

    """ Measurements in the Z basis 
    
    input: a unitary gate to be estimated
    output: quantum circuit
    
    """
    
    circ = QuantumCircuit()
    qr = QuantumRegister(3)
    circ.add_register( qr )

    circ.barrier()
    for i in range(2):
        circ.h(i+1)
    circ.barrier()

    circ.cx(2,0)
    circ.x(1)
    circ.cz(1,0)
    circ.x(1)

    circ.barrier()

    circ.append(gate, [0])
    circ.barrier()

    circ.cz(1,0)
    circ.x(2)
    circ.cx(2,0)
    circ.x(2)

    circ.barrier()

    for i in range(2):
        circ.h(i+1)

    circ.barrier()
    circ.measure_all()

    return circ


def experiment2(gate):

    """ Measurements in the X basis 
    
    input: a unitary gate to be estimated
    output: quantum circuit
    
    """
    
    circ = QuantumCircuit()
    qr = QuantumRegister(3)
    circ.add_register( qr )

    circ.barrier()
    for i in range(2):
        circ.h(i+1)
    circ.barrier()

    circ.cx(2,0)
    circ.x(1)
    circ.cz(1,0)
    circ.x(1)

    circ.barrier()
    circ.append(gate, [0])
    circ.barrier()

    circ.cz(1,0)
    circ.x(2)
    circ.cx(2,0)
    circ.x(2)

    circ.barrier()

    for i in range(2):
        circ.h(i+1)

    circ.barrier()
    circ.x(1)
    circ.cx(1,0)
    circ.x(1)
    circ.cz(2,0)

    circ.barrier()

    for i in range(2):
        circ.h(i+1)

    circ.measure_all()

    return circ 


def experiment3(gate):

    """ Measurements in the Y basis 
    
    input: a unitary gate to be estimated
    output: quantum circuit
    
    """

    circ = QuantumCircuit()
    qr = QuantumRegister(3)
    circ.add_register( qr )

    circ.barrier()
    for i in range(2):
        circ.h(i+1)
    circ.barrier()

    circ.cx(2,0)
    circ.x(1)
    circ.cz(1,0)
    circ.x(1)

    circ.barrier()
    circ.append(gate, [0])
    circ.barrier()

    circ.cz(1,0)
    circ.x(2)
    circ.cx(2,0)
    circ.x(2)

    circ.barrier()

    for i in range(2):
        circ.h(i+1)

    circ.barrier()
    circ.x(1)
    circ.cx(1,0)
    circ.x(1)
    circ.cz(2,0)

    φ0,φ1,φ2,φ3 = 0, np.pi/2, np.pi/2, np.pi

    circ.barrier()
    circ.p(φ1,1)
    circ.p(φ2,2)


    circ.barrier()

    for i in range(2):
        circ.h(i+1)

    circ.measure_all()

    return circ


def UnitaryEstimation(p0,p1,p2): 
    
    """
    Funtion to estimate a Unitary Transformation 
    
    input: Probabilities distributions obtained from the measurements (experiments1, experiments2 and experiments3)
    output: estimated unitary transformation
    """

    for pi in [p0,p1,p2]:
        for l in ['00', '01', '10', '11']:
            if l not in pi.keys():
                pi[l] = 0

    probs1 = np.array([p1['00'],p1['10'],p1['01'],p1['11']])
    probs2 = np.array([p2['00'],p2['10'],p2['01'],p2['11']])
    radius_r = np.sqrt(np.array([p0['10'],p0['00'],p0['11'],p0['01']]))
    
    if radius_r[0]==1.:
        resulting_U=Paulis[0]
    elif radius_r[1]==1.:
        resulting_U=Paulis[1]
    elif radius_r[2]==1.:
        resulting_U=Paulis[2]
    elif radius_r[3]==1.:
        resulting_U=Paulis[3]
    
    elif radius_r[1]==0. and radius_r[2]==0.:
        y1=1
        x2=1
        y3=np.sign(1-4*probs2[0])
        if y3 == 0.0:
            y3 = random.choice([-1,1])

        delta1 = np.arcsin(y1)
        delta2 = np.arccos(x2) 
        delta3 = np.arcsin(y3)

        resulting_phases  = np.array([0, delta1,delta1+delta2,delta1+delta2+delta3])

        resulting_U = np.zeros((2,2), dtype = complex)

        for i,(rr,aa) in enumerate(zip(radius_r,resulting_phases)):
            resulting_U += rr*np.exp(1j*aa)*Paulis[i]
        
    elif radius_r[0]==0. and radius_r[3]==0.:
        x2 = np.sign(probs1[0] + probs1[3] - (probs2[0] + probs2[3]))
        if x2 == 0.0:
            x2 = random.choice([-1,1])
        y1=1
        y3=1
        
        delta1 = np.arcsin(y1)
        delta2 = np.arccos(x2) 
        delta3 = np.arcsin(y3)

        resulting_phases  = np.array([0, delta1,delta1+delta2,delta1+delta2+delta3])

        resulting_U = np.zeros((2,2), dtype = complex)

        for i,(rr,aa) in enumerate(zip(radius_r,resulting_phases)):
            resulting_U += rr*np.exp(1j*aa)*Paulis[i]
        
    else:
        x2 = np.sign(probs1[0] + probs1[3] - (probs2[0] + probs2[3]))
        if x2 == 0.0:
            x2 = random.choice([-1,1])

        y1y3 = np.sign( (1 -(probs1[0] + probs1[3] + probs2[0] + probs2[3] ))/x2  )
            
        if y1y3 < 0:
            if ( np.abs(radius_r[1]*radius_r[3] - radius_r[0]*radius_r[2]) 
                < np.abs(radius_r[2]*radius_r[3]-radius_r[0]*radius_r[1]) ):
                y3=np.sign((probs2[2]-probs2[3]-radius_r[1]*radius_r[2]*x2-radius_r[0]*radius_r[3]*x2*y1y3)/
                        (radius_r[2]*radius_r[3]-radius_r[0]*radius_r[1]))
            
            elif np.abs(radius_r[1]*radius_r[3] - radius_r[0]*radius_r[2])==0.:
                y3=np.sign((probs2[2]-probs2[3]-radius_r[1]*radius_r[2]*x2-radius_r[0]*radius_r[3]*x2*y1y3))  
            else:
                y3 = np.sign( (probs2[2] + probs2[3] - 0.5)/( x2*(radius_r[1]*radius_r[3] - radius_r[0]*radius_r[2]) ) )
            y1 = -y3

        if y1y3 >= 0:
            if ( np.abs(radius_r[1]*radius_r[3] + radius_r[0]*radius_r[2])
                <np.abs(radius_r[2]*radius_r[3]+radius_r[0]*radius_r[1]) ):
                y3=np.sign(probs2[2]-probs2[3]-radius_r[1]*radius_r[2]*x2-radius_r[0]*radius_r[3]*x2*y1y3)
            else:
                y3 = np.sign( (probs2[2] + probs2[3] - 0.5)/( x2 ) )
            y1 = y3
        
        delta1 = np.arcsin(y1)
        delta2 = np.arccos(x2) 
        delta3 = np.arcsin(y3)

        resulting_phases  = np.array([0, delta1,delta1+delta2,delta1+delta2+delta3])

        resulting_U = np.zeros((2,2), dtype = complex)

        for i,(rr,aa) in enumerate(zip(radius_r,resulting_phases)):
            resulting_U += rr*np.exp(1j*aa)*Paulis[i]
        
    return resulting_U
    

def miro_circuits(gate):
    
    circ1 = experiment1(gate)
    circ2 = experiment2(gate)
    circ3 = experiment3(gate)
    
    return [circ1, circ2, circ3]


def qiskit_circ(gate):
    
    qr = QuantumRegister(1)
    circ = QuantumCircuit(qr)
    circ.append(gate, [0])
    
    return circ


def run_experiments(gate, backend, num_of_shots = 100, rout_mitigation = True):
    
    """
        Function to estimate a two-dimensional unitary gate.
        In this function measurements in the Z, X and Y pauli basis are performed in 
        experiments1, experiments2 and experiments3. Then, the probabilities obtained 
        from experiments are passed to "UnitaryEstimation" estimate "gate". 
        
        Inputs: A "gate" ( a qiskit.extensions.unitary.UnitaryGate instance) to be estimated 
        Return: Probabilities distributions p0,p1 and p2 obtained from experiments1, experiments and experiments3.
    """
    
    list_of_circs = miro_circuits(gate)
    qubits = [0,1,2]
    run_circs = backend.run(list_of_circs, shots=num_of_shots)
    
    if rout_mitigation:
        exp = LocalReadoutError(qubits)
        result = exp.run(backend)
        mitigator = result.analysis_results(0).value
        experimental_probs = []
        for circ in run_circs.result().get_counts():
            experimental_probs.append(mitigator.quasi_probabilities(circ).nearest_probability_distribution())
        experimental_probs = [probs.binary_probabilities() for probs in experimental_probs]
    else:
        experimental_probs = run_circs.result().get_counts()
        experimental_probs = [counts2probs(p, num_of_shots) for p in experimental_probs]

    probs_for_estimation = []
    for probs in experimental_probs:
        for l in ['000', '001', '010', '011','100', '101', '110', '111']:
            if l not in probs.keys():
                probs[l] = 0
        probs2qubits = {}
        probs2qubits['00'] = probs['000'] + probs['001']
        probs2qubits['01'] = probs['010'] + probs['011']
        probs2qubits['10'] = probs['100'] + probs['101']
        probs2qubits['11'] = probs['110'] + probs['111']
        probs_for_estimation.append(probs2qubits)

    p0, p1, p2 = probs_for_estimation
    
    return p0, p1, p2


def CustomNoiseModel(Rel_Ts,Errores,Times):
    
    
    """ This function is used to customize the noise model """
    
    T1s=Rel_Ts[0]
    T2s=Rel_Ts[1]
    sx_errors=Errores[0]
    x_errors=Errores[1]
    cxs=Errores[2]
    readout_errors=Errores[3]
    cx_34=cxs[0]
    cx_13=cxs[1]
    cx_12=cxs[2]
    cx_01=cxs[3]
    time_id=Times[0]
    time_sx=Times[1]
    time_x=Times[2]
    time_cx=Times[3]
    time_reset=Times[4]
    time_measure=Times[5]
    
    #Create an ideal NoiseModel
    our_noise_model = NoiseModel()

    #Add depolarizing error for 'sx' and 'x'
    errors = [depolarizing_error(x_errors[i], 1) for i in range(5)]
    for i in range(5):
        our_noise_model.add_quantum_error(errors[i], ['sx', 'x'], [i])

    #Add depolarizing error for 'cx'
    error=depolarizing_error(cx_34, 2)
    our_noise_model.add_quantum_error(error, ['cx'], [3,4])

    error=depolarizing_error(cx_34, 2)
    our_noise_model.add_quantum_error(error, ['cx'], [4,3])

    error=depolarizing_error(cx_13, 2)
    our_noise_model.add_quantum_error(error, ['cx'], [1,3])

    error=depolarizing_error(cx_13, 2)
    our_noise_model.add_quantum_error(error, ['cx'], [3,1])

    error=depolarizing_error(cx_12, 2)
    our_noise_model.add_quantum_error(error, ['cx'], [1,2])

    error=depolarizing_error(cx_12, 2)
    our_noise_model.add_quantum_error(error, ['cx'], [2,1])

    error=depolarizing_error(cx_01, 2)
    our_noise_model.add_quantum_error(error, ['cx'], [0,1])

    error=depolarizing_error(cx_01, 2)
    our_noise_model.add_quantum_error(error, ['cx'], [1,0])


    #Add bit flip error for measurements
    error_meas = [pauli_error([('X',readout_errors[i]), ('I', 1 - readout_errors[i])]) for i in range(5)]
    for i in range(5):
        our_noise_model.add_quantum_error(error_meas[i], ['measure'], [i])

    #Add thermal relaxation
    # QuantumError objects
    errors_reset = [thermal_relaxation_error(t1, t2, time_reset)
                    for t1, t2 in zip(T1s, T2s)]
    errors_measure = [thermal_relaxation_error(t1, t2, time_measure)
                    for t1, t2 in zip(T1s, T2s)]
    errors_id  = [thermal_relaxation_error(t1, t2, time_id)
                for t1, t2 in zip(T1s, T2s)]
    errors_sx  = [thermal_relaxation_error(t1, t2, time_sx)
                for t1, t2 in zip(T1s, T2s)]
    errors_x  = [thermal_relaxation_error(t1, t2, time_x)
                for t1, t2 in zip(T1s, T2s)]
    errors_cx = [[thermal_relaxation_error(t1a, t2a, time_cx).expand(
                thermal_relaxation_error(t1b, t2b, time_cx))
                for t1a, t2a in zip(T1s, T2s)]
                for t1b, t2b in zip(T1s, T2s)]

    our_noise_model.add_quantum_error(errors_cx[3][4], "cx", [3, 4])
    our_noise_model.add_quantum_error(errors_cx[1][3], "cx", [1, 3])
    our_noise_model.add_quantum_error(errors_cx[1][2], "cx", [1, 2])
    our_noise_model.add_quantum_error(errors_cx[0][1], "cx", [0, 1])


    for j in range(5):
        our_noise_model.add_quantum_error(errors_reset[j], "reset", [j])
        our_noise_model.add_quantum_error(errors_measure[j], "measure", [j])
        our_noise_model.add_quantum_error(errors_id[j], "id", [j])
        our_noise_model.add_quantum_error(errors_sx[j], "sx", [j])
        our_noise_model.add_quantum_error(errors_x[j], "x", [j])
    
    return our_noise_model


