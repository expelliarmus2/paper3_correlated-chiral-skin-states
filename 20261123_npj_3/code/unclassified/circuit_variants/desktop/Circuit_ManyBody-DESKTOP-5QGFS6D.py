"""


GPU势能在linux上运行。
https://qiskit.github.io/qiskit-aer/tutorials/1_aersimulator.html#GPU-Simulation

Q: 直接从算符映射为矩阵，结果和哈密顿量的不一致，应该如何求解？
因为比特没有考虑交换不同比特的负号，所以需要进行Jordan-Wigner变换，
a_i = \Pi_{m<i} Z_{m} a_i。也就是如果前面有占据的话，那么应该出现一个负号。
但是这一点在量子线路中会出现很长的链，如何处理？？
例 a_i^{d} a_j -> a_i^{d} \Pi_{i=<m<j} a_j 


(1<<4)+1才是17，去掉括号是32

运算顺序，可以知道加法运算顺序更高。
**	幂
~	按位“取反”
*、/、%、//	乘、除、取模、取整除
+、-	加、减
>>、<<	右移、左移
&	按位“与”
^、|	按位“异或”、按位“或”
<=、<、>、>=	比较运算符
==、!=	等于、不等于
=、%=、/=、//=、-=、+=、*=、**=	赋值运算符
is、is not	身份运算符
in、not in	成员运算符
and or not	逻辑运算符

"""
import numpy as np
import time
from qiskit import QuantumCircuit, Aer, execute
from qiskit.quantum_info import Operator
from scipy.linalg import expm
from qiskit.quantum_info import Operator, Statevector
import matplotlib.pyplot as plt

from qiskit import Aer, transpile, QuantumCircuit
from qiskit.providers.aer import AerSimulator
from MeasureNonhermitianSkin.Experimental.ManyBodyQWZ import ParticleConservedSystem
from ComputationalPhysics.Package.BitOperation import ReadBit

num_x = 2
num_y = 4
sig_m = np.array([[0, 1.],[0,0]], dtype=np.complex128)
sig_p = np.array([[0, 0.],[1.,0]], dtype=np.complex128) # A->B
sig_I = np.identity(2, dtype=np.complex128)
sig_Z = np.array([[1., 0],[0,-1.]], dtype=np.complex128)

def simulate_qwz_evolution(num_x, num_y, time_steps, hamiltonian):
    """
    Simulates the time evolution of the QWZ model using Qiskit.

    Args:
        num_x (int): Number of unit cells in the x direction.
        num_y (int): Number of unit cells in the y direction.
        time_steps (int): Number of time steps for the evolution.
        hamiltonian (np.ndarray): The Hamiltonian matrix of the system.

    Returns:
        list: List of quantum states at each time step.
    """
    num_qubits = int(np.log2(hamiltonian.shape[0]))
    qc = QuantumCircuit(num_qubits)

    # Initial state (can be modified as needed)
    initial_state = np.zeros(hamiltonian.shape[0])
    initial_state[0] = 1

    # Convert Hamiltonian to a unitary operator for time evolution
    dt = 1.0 / time_steps
    evolution_operator = Operator(expm(-1j * hamiltonian * dt))

    # Simulate time evolution
    backend = Aer.get_backend('statevector_simulator')
    states = [initial_state]
    for _ in range(time_steps):
        qc.unitary(evolution_operator, range(num_qubits), label="Time Evolution")
        job = execute(qc, backend, initial_statevector=states[-1])
        result = job.result()
        states.append(result.get_statevector())

    return states

def Circuit(t_list):
    # 这里也要注意非厄米出现负号，只要涉及到两个粒子就需要考虑了？？但是如何放进去呢？
    u = -1.5 
    num_x = 2
    num_y = 3
    gamma = 0.2
    total_mode = 2*num_x*num_y
    total_bit = total_mode + 4
    t = t_list[1]

    # diagonal term.
    H0 = np.array([[0, 0],[0, u]], dtype=np.complex128)
    U0 = expm(-1j*H0*t) # A site
    U0_dagger = expm(1j*H0*t) # B site

    # y direction.
    Hx = np.zeros((4, 4), dtype=np.complex128)
    Hx[1, 2] = 1/2
    Hx[2, 1] = 1/2
    Ux = expm(-1j * Hx * t)
    Ux_dagger = expm(1j * Hx * t)

    HxAA_y = np.kron(sig_p, np.kron(sig_Z, sig_m))/2
    UxAA_y = expm(-1j * (HxAA_y+HxAA_y.T) * t)
    UxAA_y_dagger = expm(1j * (HxAA_y+HxAA_y.T) * t)

    HxAB_y = np.kron(sig_p, np.kron(sig_Z, np.kron(sig_Z, sig_m)))/2
    UxAB_y_dagger = expm(1j * (HxAB_y+HxAB_y.T) * t)

    # x direction hopping.
    Z_string = np.array([1.], dtype=np.complex128)
    for i in range(2*num_y-2):
        Z_string = np.kron(Z_string, sig_Z)

    HyBA_x = 1j * np.kron(sig_p, np.kron(Z_string, sig_m))/2 # 小到大系数为i
    UyBA_x = expm(-1j * (HyBA_x+np.conj(HyBA_x).T) * t)

    HyAB_x = 1j * np.kron(sig_p, np.kron(sig_Z, np.kron(Z_string, np.kron(sig_Z, sig_m))))/2 # 小到大系数为i
    UyAB_x = expm(-1j * (HyAB_x+np.conj(HyAB_x).T) * t)

    HxAA_x = np.kron(sig_p, np.kron(Z_string, np.kron(sig_Z, sig_m)))/2 # 小到大系数为i
    UxAA_x = expm(-1j * (HxAA_x+np.conj(HxAA_x).T) * t)
    UxAA_x_dagger = expm(1j * (HxAA_x+np.conj(HxAA_x).T) * t)

    # measurements 
    alpha = np.exp(-t*gamma)
    beta = np.sqrt(1-alpha**2)
    UM = np.identity(2**2, dtype=np.complex128)
    UM[1:3, 1:3] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128)
    # 增益耗散取决于辅助比特制备在什么量子态。

    qc = QuantumCircuit(total_bit)
    # onsite
    for i in range(0, total_mode, 2):
        qc.unitary(Operator(U0), [i], label='U0')
    for i in range(1, total_mode, 2):
        qc.unitary(Operator(U0_dagger), [i], label='U0_d')
    # x direction.
    for i in range(0, total_mode, 2):
        # A->A in x direction
        if i+2*num_y<total_mode:
            qc.unitary(Operator(UxAA_x), list(range(i, i+2*num_y+1)), label='Ux')
            qc.unitary(Operator(UyAB_x), list(range(i, i+2*num_y+2)), label='Uy') # 也可能是dagger
    for i in range(1, total_mode, 2):
        # B->B in x direction
        if i+2*num_y<total_mode:
            qc.unitary(Operator(UxAA_x_dagger), list(range(i, i+2*num_y+1)), label='Ux_d')
            qc.unitary(Operator(UyBA_x), list(range(i, i+2*num_y)), label='Uy_d')
    # y direction
    for i in range(0, total_mode, 2):
        if i not in list(range(2*num_y-2,total_mode,2*num_y)):
            qc.unitary(Operator(UxAA_y), [i, i+1, i+2], label='Ux')
            qc.unitary(Operator(UxAB_y_dagger), [i, i+1, i+2, i+3], label='Ux_d') # 也可能是dagger
    for i in range(1, total_mode, 2):
        if i not in list(range(2*num_y-1,total_mode,2*num_y)):
            qc.unitary(Operator(UxAA_y_dagger), [i, i+1, i+2], label='Ux_d')
            qc.unitary(Operator(Ux), [i, i+1], label='Ux') # 也可能是dagger # Ux是对的，刚好没有Jordan变换的负号。
    
    dissi_site = [0,1,2*num_y, 2*num_y+1]
    qm = QuantumCircuit(total_bit)
    for i in range(len(dissi_site)):
        qm.unitary(Operator(UM), [dissi_site[i], i+2*num_x*num_y])

    q_flip = QuantumCircuit(total_bit)
    q_flip.x([2*num_x*num_y, 2*num_x*num_y+1, 2*num_x*num_y+2, 2*num_x*num_y+3])
    init_qc = QuantumCircuit(2*num_x*num_y)
    init_qc.x([0,4])
    state = Statevector(init_qc)
    state = np.array(state)

    from MeasureNonhermitianSkin.Experimental.ManyBodyQWZ import ParticleConservedSystem

    QWZ2 = ParticleConservedSystem(num_x, num_y, 2)
    QWZ2.add_onsite_term(-1.5)
    QWZ2.add_NH(gamma)
    QWZ2.add_xhopping_term()
    QWZ2.add_yhopping_term()

    ham2 = QWZ2.hamiltonian
    basis = QWZ2.basis
    print(ham2)
    initial_pick = np.array([state[basis[i]] for i in range(len(basis))])
    print('Norm', np.sum(np.abs(initial_pick)**2))
    record_comp = np.zeros((2, len(basis), len(t_list)), dtype=np.complex128)
    record_comp[0,:,0] = initial_pick
    record_comp[1,:,0] = initial_pick  # digital
    state = state[-2**(total_mode):]
    
    for i in range(len(t_list)-1):
        state = Statevector(state)
        full_circuit = QuantumCircuit(total_bit)
        full_circuit.initialize(state, list(range(0,total_mode,1)))  # 设定初态
        # full_circuit.initialize(state, list(range(total_mode-1, -1, -1)))  # 设定初态 这一个肯定不对，会出现震荡，也就是交换两次才回来。
        full_circuit = full_circuit.compose(q_flip)
        full_circuit = full_circuit.compose(qc)
        full_circuit = full_circuit.compose(qm)
        initial_state = Statevector(full_circuit)
        state = np.array(initial_state)[-2**(total_mode):]
        print('norm 0', np.linalg.norm(state))
        state = state / np.linalg.norm(state)
        state_pick = np.array([state[basis[i]] for i in range(len(basis))])
        record_comp[0,:,i+1] = state_pick
        record_comp[1,:,i+1] = expm(-1j*ham2*t_list[i+1]) @ record_comp[1,:,0]
        record_comp[1,:,i+1] = record_comp[1,:,i+1] / np.linalg.norm(record_comp[1,:,i+1])
    
    record_comp = np.abs(record_comp)**2
    #record_comp = record_comp[:,::2,:] + record_comp[:,1::2,:]
    print(record_comp[:,:,-1])
    fig, ax = plt.subplots(1,2,sharex=True, sharey=True)
    for i in range(len(basis)//2):
        ax[0].plot(t_list, record_comp[0, i], 'o-')
        ax[1].plot(t_list, record_comp[1, i], 'o-')
    plt.show()


def Circuit2(t_list):
    # 这里也要注意非厄米出现负号，只要涉及到两个粒子就需要考虑了？？但是如何放进去呢？
    u = -1.5 
    num_x = 2
    num_y = 4
    gamma = 0.2
    total_mode = 2*num_x*num_y
    total_bit = total_mode + 1
    t = t_list[1]

    # diagonal term.
    H0 = np.array([[0, 0],[0, u]], dtype=np.complex128)
    U0 = expm(-1j*H0*t) # A site
    U0_dagger = expm(1j*H0*t) # B site

    # y direction.
    Hx = np.zeros((4, 4), dtype=np.complex128)
    Hx[1, 2] = 1/2
    Hx[2, 1] = 1/2
    Ux = expm(-1j * Hx * t)
    Ux_dagger = expm(1j * Hx * t)

    HxAA_y = np.kron(sig_p, np.kron(sig_Z, sig_m))/2
    UxAA_y = expm(-1j * (HxAA_y+HxAA_y.T) * t)
    UxAA_y_dagger = expm(1j * (HxAA_y+HxAA_y.T) * t)

    HxAB_y = np.kron(sig_p, np.kron(sig_Z, np.kron(sig_Z, sig_m)))/2
    UxAB_y_dagger = expm(1j * (HxAB_y+HxAB_y.T) * t)

    # x direction hopping.
    Z_string = np.array([1.], dtype=np.complex128)
    for i in range(2*num_y-2):
        Z_string = np.kron(Z_string, sig_Z)

    HyBA_x = 1j * np.kron(sig_p, np.kron(Z_string, sig_m))/2 # 小到大系数为i
    UyBA_x = expm(-1j * (HyBA_x+np.conj(HyBA_x).T) * t)

    HyAB_x = 1j * np.kron(sig_p, np.kron(sig_Z, np.kron(Z_string, np.kron(sig_Z, sig_m))))/2 # 小到大系数为i
    UyAB_x = expm(-1j * (HyAB_x+np.conj(HyAB_x).T) * t)

    HxAA_x = np.kron(sig_p, np.kron(Z_string, np.kron(sig_Z, sig_m)))/2 # 小到大系数为i
    UxAA_x = expm(-1j * (HxAA_x+np.conj(HxAA_x).T) * t)
    UxAA_x_dagger = expm(1j * (HxAA_x+np.conj(HxAA_x).T) * t)

    # measurements 
    alpha = np.exp(-t*gamma)
    beta = np.sqrt(1-alpha**2)
    UM = np.identity(2**2, dtype=np.complex128)
    UM[1:3, 1:3] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128)
    # 增益耗散取决于辅助比特制备在什么量子态。

    qc = QuantumCircuit(total_bit)
    # onsite
    for i in range(0, total_mode, 2):
        qc.unitary(Operator(U0), [i], label='U0')
    for i in range(1, total_mode, 2):
        qc.unitary(Operator(U0_dagger), [i], label='U0_d')
    # x direction.
    for i in range(0, total_mode, 2):
        # A->A in x direction
        if i+2*num_y<total_mode:
            qc.unitary(Operator(UxAA_x), list(range(i, i+2*num_y+1)), label='Ux')
            qc.unitary(Operator(UyAB_x), list(range(i, i+2*num_y+2)), label='Uy') # 也可能是dagger
    for i in range(1, total_mode, 2):
        # B->B in x direction
        if i+2*num_y<total_mode:
            qc.unitary(Operator(UxAA_x_dagger), list(range(i, i+2*num_y+1)), label='Ux_d')
            qc.unitary(Operator(UyBA_x), list(range(i, i+2*num_y)), label='Uy_d')
    # y direction
    for i in range(0, total_mode, 2):
        if i not in list(range(2*num_y-2,total_mode,2*num_y)):
            qc.unitary(Operator(UxAA_y), [i, i+1, i+2], label='Ux')
            qc.unitary(Operator(UxAB_y_dagger), [i, i+1, i+2, i+3], label='Ux_d') # 也可能是dagger
    for i in range(1, total_mode, 2):
        if i not in list(range(2*num_y-1,total_mode,2*num_y)):
            qc.unitary(Operator(UxAA_y_dagger), [i, i+1, i+2], label='Ux_d')
            qc.unitary(Operator(Ux), [i, i+1], label='Ux') # 也可能是dagger # Ux是对的，刚好没有Jordan变换的负号。
    
    dissi_site = [0,1,2*num_y, 2*num_y+1]
    # qm = QuantumCircuit(total_bit)
    # for i in range(len(dissi_site)):
    #     qm.unitary(Operator(UM), [dissi_site[i], i+2*num_x*num_y])

    #q_flip = QuantumCircuit(total_bit)
    #q_flip.x([2*num_x*num_y, 2*num_x*num_y+1, 2*num_x*num_y+2, 2*num_x*num_y+3])

    from MeasureNonhermitianSkin.Experimental.ManyBodyQWZ import ParticleConservedSystem

    QWZ2 = ParticleConservedSystem(num_x, num_y, 2)
    QWZ2.add_onsite_term(-1.5)
    QWZ2.add_NH(gamma)
    QWZ2.add_xhopping_term()
    QWZ2.add_yhopping_term()

    ham2 = QWZ2.hamiltonian
    basis = QWZ2.basis
    print(ham2)

    init_qc = QuantumCircuit(2*num_x*num_y)
    init_qc.x([0,4])
    state = Statevector(init_qc)
    state = np.array(state)
    initial_pick = np.array([state[basis[i]] for i in range(len(basis))])
    print('Norm', np.sum(np.abs(initial_pick)**2))
    record_comp = np.zeros((2, len(basis), len(t_list)), dtype=np.complex128)
    record_comp[0,:,0] = initial_pick
    record_comp[1,:,0] = initial_pick  # digital
    state = state[-2**(total_mode):]

    for i in range(len(t_list)-1):
        state = Statevector(state)
        full_circuit = QuantumCircuit(total_bit)
        full_circuit.initialize(state, list(range(0,total_mode,1)))  # 设定初态
        # full_circuit.initialize(state, list(range(total_mode-1, -1, -1)))  # 设定初态 这一个肯定不对，会出现震荡，也就是交换两次才回来。
        #full_circuit = full_circuit.compose(q_flip)
        full_circuit = full_circuit.compose(qc)
        initial_state = Statevector(full_circuit)
        state = np.array(initial_state)[:2**total_mode]
        norm = 1
        for j in range(len(dissi_site)):
            state = Statevector(state)
            full_circuit = QuantumCircuit(total_bit)
            full_circuit.initialize(state, list(range(0,total_mode,1)))
            full_circuit.x([total_bit-1])
            full_circuit.unitary(Operator(UM), [dissi_site[j], total_bit-1])
            state = Statevector(full_circuit)
            state = np.array(state)[-2**(total_mode):]
            norm *= np.linalg.norm(state)
            state = state / np.linalg.norm(state)
        print('norm 0', norm)
        state_pick = np.array([state[basis[i]] for i in range(len(basis))])
        record_comp[0,:,i+1] = state_pick
        record_comp[1,:,i+1] = expm(-1j*ham2*t_list[i+1]) @ record_comp[1,:,0]
        record_comp[1,:,i+1] = record_comp[1,:,i+1] / np.linalg.norm(record_comp[1,:,i+1])
    
    record_comp = np.abs(record_comp)**2
    #record_comp = record_comp[:,::2,:] + record_comp[:,1::2,:]
    print(record_comp[:,:,-1])
    fig, ax = plt.subplots(1,2,sharex=True, sharey=True)
    for i in range(len(basis)//2):
        ax[0].plot(t_list, record_comp[0, i], 'o-')
        ax[1].plot(t_list, record_comp[1, i], 'o-')
    plt.show()


def Circuit3(t_list):
    # 这里也要注意非厄米出现负号，只要涉及到两个粒子就需要考虑了？？但是如何放进去呢？
    u = -1.5 
    num_x = 2
    num_y = 4
    gamma = 0.2
    total_mode = 2*num_x*num_y
    total_bit = total_mode + 1
    t = t_list[1]

    # diagonal term.
    H0 = np.array([[0, 0],[0, u]], dtype=np.complex128)
    U0 = expm(-1j*H0*t) # A site
    U0_dagger = expm(1j*H0*t) # B site

    # y direction.
    Hx = np.zeros((4, 4), dtype=np.complex128)
    Hx[1, 2] = 1/2
    Hx[2, 1] = 1/2
    Ux = expm(-1j * Hx * t)
    Ux_dagger = expm(1j * Hx * t)

    HxAA_y = np.kron(sig_p, np.kron(sig_Z, sig_m))/2
    UxAA_y = expm(-1j * (HxAA_y+HxAA_y.T) * t)
    UxAA_y_dagger = expm(1j * (HxAA_y+HxAA_y.T) * t)

    HxAB_y = np.kron(sig_p, np.kron(sig_Z, np.kron(sig_Z, sig_m)))/2
    UxAB_y_dagger = expm(1j * (HxAB_y+HxAB_y.T) * t)

    # x direction hopping.
    Z_string = np.array([1.], dtype=np.complex128)
    for i in range(2*num_y-2):
        Z_string = np.kron(Z_string, sig_Z)

    HyBA_x = 1j * np.kron(sig_p, np.kron(Z_string, sig_m))/2 # 小到大系数为i
    UyBA_x = expm(-1j * (HyBA_x+np.conj(HyBA_x).T) * t)

    HyAB_x = 1j * np.kron(sig_p, np.kron(sig_Z, np.kron(Z_string, np.kron(sig_Z, sig_m))))/2 # 小到大系数为i
    UyAB_x = expm(-1j * (HyAB_x+np.conj(HyAB_x).T) * t)

    HxAA_x = np.kron(sig_p, np.kron(Z_string, np.kron(sig_Z, sig_m)))/2 # 小到大系数为i
    UxAA_x = expm(-1j * (HxAA_x+np.conj(HxAA_x).T) * t)
    UxAA_x_dagger = expm(1j * (HxAA_x+np.conj(HxAA_x).T) * t)

    # 加入相互作用
    U = 0.4  # 粒子数守恒的相互作用。
    H_inter1 = np.zeros((4,4), dtype=np.complex128)
    H_inter1[3,3] = U
    U_inter = expm(-1j * H_inter1 * t)

    # X: (U n_{x,y, A}+n_{x,y, B}) * (U n_{x+1,y, A}+n_{x+1,y, B}). Y同样。

    # measurements 
    alpha = np.exp(-t*gamma)
    beta = np.sqrt(1-alpha**2)
    UM = np.identity(2**2, dtype=np.complex128)
    UM[1:3, 1:3] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128)
    # 增益耗散取决于辅助比特制备在什么量子态。

    qc = QuantumCircuit(total_mode)
    #qc.x([0,4])
    # onsite
    for i in range(0, total_mode, 2):
        qc.unitary(Operator(U0), [i], label='U0')
    for i in range(1, total_mode, 2):
        qc.unitary(Operator(U0_dagger), [i], label='U0_d')
    # x direction.
    for i in range(0, total_mode, 2):
        # A->A in x direction
        if i+2*num_y<total_mode:
            qc.unitary(Operator(UxAA_x), list(range(i, i+2*num_y+1)), label='Ux')
            qc.unitary(Operator(UyAB_x), list(range(i, i+2*num_y+2)), label='Uy') # 也可能是dagger
    for i in range(1, total_mode, 2):
        # B->B in x direction
        if i+2*num_y<total_mode:
            qc.unitary(Operator(UxAA_x_dagger), list(range(i, i+2*num_y+1)), label='Ux_d')
            qc.unitary(Operator(UyBA_x), list(range(i, i+2*num_y)), label='Uy_d')
    # y direction
    for i in range(0, total_mode, 2):
        if i not in list(range(2*num_y-2,total_mode,2*num_y)):
            qc.unitary(Operator(UxAA_y), [i, i+1, i+2], label='Ux')
            qc.unitary(Operator(UxAB_y_dagger), [i, i+1, i+2, i+3], label='Ux_d') # 也可能是dagger
    for i in range(1, total_mode, 2):
        if i not in list(range(2*num_y-1,total_mode,2*num_y)):
            qc.unitary(Operator(UxAA_y_dagger), [i, i+1, i+2], label='Ux_d')
            qc.unitary(Operator(Ux), [i, i+1], label='Ux') # 也可能是dagger # Ux是对的，刚好没有Jordan变换的负号。
    
    # # interaction 1
    # for i in range(0, total_mode, 2):
    #     qc.unitary(Operator(U_inter), [i, i+1], label='UI')
    
    # interaction 2
    for i in range(0, total_mode, 2):
        if i not in list(range(2*num_y-2,total_mode,2*num_y)):
            qc.unitary(Operator(U_inter), [i, i+2], label='UI')
            qc.unitary(Operator(U_inter), [i, i+3], label='UI')
    for i in range(1, total_mode, 2):
        if i not in list(range(2*num_y-1,total_mode,2*num_y)):
            qc.unitary(Operator(U_inter), [i, i+1], label='UI')
            qc.unitary(Operator(U_inter), [i, i+2], label='UI')
    for i in range(0, total_mode, 2*num_y):
        # A->A in x direction
        distance = 2*num_y
        if i+2*num_y<total_mode:
            qc.unitary(Operator(U_inter), [i, i+distance], label='UI')
            qc.unitary(Operator(U_inter), [i, i+distance+1], label='UI')
            qc.unitary(Operator(U_inter), [i+1, i+distance-1], label='UI')
            qc.unitary(Operator(U_inter), [i+1, i+distance], label='UI')

    # 加入相互作用
    #qc.save_state()

    # qm = QuantumCircuit(total_bit)
    # for i in range(len(dissi_site)):
    #     qm.unitary(Operator(UM), [dissi_site[i], i+2*num_x*num_y])

    #q_flip = QuantumCircuit(total_bit)
    #q_flip.x([2*num_x*num_y, 2*num_x*num_y+1, 2*num_x*num_y+2, 2*num_x*num_y+3])

    QWZ2 = ParticleConservedSystem(num_x, num_y, 2)
    QWZ2.add_onsite_term(-1.5)
    QWZ2.add_NH(gamma)
    QWZ2.add_xhopping_term()
    QWZ2.add_yhopping_term()
    QWZ2.add_interaction_inter(U)

    ham2 = QWZ2.hamiltonian
    basis = QWZ2.basis
    #print(ham2)


    # simulator = AerSimulator(
    #     method='matrix_product_state',
    # )
    # 使用Aer的GPU加速模拟器（如果可用）
    simulator = AerSimulator(
        method='statevector',
        device='GPU' if 'GPU' in AerSimulator().available_devices() else 'CPU',
        max_parallel_threads=1,
    )
    compiled_qc = transpile(qc, simulator)

    # device='GPU' if AerSimulator().available_devices(True)['name'] == 'GPU' else 'CPU', # optimization_level=2,
    init_state = np.zeros(2**total_bit, dtype=np.complex128)
    #init_state[(1<<4)+1] = 1.0  # 对应x([0,4])的激发态
    init_state[(1<<5)+1] = 1.0  # 对应x([0,4])的激发态
    initial_pick = init_state[basis].copy()
    print('Initial Norm:', np.linalg.norm(initial_pick))

    record_comp = np.zeros((2, len(basis), len(t_list)), dtype=np.complex128)
    record_comp[0,:,0] = initial_pick
    record_comp[1,:,0] = initial_pick  # digital

    # dissi_site = [0,1,2*num_y, 2*num_y+1]
    #dissi_site = [0,1,2*num_y, 2*num_y+1]
    dissi_site = sorted([i for i in range(0, total_mode, 2*num_y)] + [i for i in range(1, total_mode, 2*num_y)])
    # 预编译耗散操作电路模板
    dissi_circuits = []
    for site in dissi_site:
        dissi_qc = QuantumCircuit(total_bit)
        #dissi_qc.x(total_bit-1)
        dissi_qc.unitary(Operator(UM), [site, total_bit-1])
        #dissi_qc.save_state()
        # dissi_circuits.append(transpile(dissi_qc, simulator))
        dissi_circuits.append(dissi_qc)

    # record_comp = np.zeros((2, len(basis), len(t_list)), dtype=np.complex128)
    # record_comp[0,:,0] = initial_pick
    # record_comp[1,:,0] = initial_pick  # digital
    # state = state[-2**(total_mode):]

    current_state = init_state.copy()

    #print(compiled_qc.draw())
    for i in range(len(t_list)-1):
        print(i)
        # --------------------------
        # 主演化过程
        # --------------------------
        # 直接使用模拟器进行状态演化
        #print(current_state[:2**total_mode])
        full_circuit = QuantumCircuit(total_mode)
        # full_circuit.initialize(current_state[:2**total_mode], list(range(0,total_mode,1)))
        full_circuit.set_statevector(current_state[:2**total_mode])
        full_circuit = full_circuit.compose(qc)
        full_circuit.save_state()
        compiled_qc = transpile(full_circuit, simulator)
        #print(compiled_qc.draw())
        evolved = simulator.run(
            compiled_qc,
            shots=1,
            initial_statevector=current_state[:2**total_mode],
            memory=True
        ).result().data()['statevector']   # get_statevector() # data['statevector'] 
        # memory=True
        # --------------------------
        # 耗散过程优化
        # --------------------------
        for j, dissi_qc in enumerate(dissi_circuits):
            # 合并耗散操作和测量
            full_state = np.kron(np.array([0,1], dtype=np.complex128),evolved)  # 添加辅助比特
            full_circuit = QuantumCircuit(total_bit)
            # full_circuit.initialize(full_state, list(range(0,total_bit,1)))
            full_circuit.set_statevector(full_state) # 这个更快！！是对全部初始化。
            # full_circuit.compose(dissi_qc)
            full_circuit = full_circuit.compose(dissi_qc)
            full_circuit.save_state()
            compiled_qc1 = transpile(full_circuit, simulator)
            result = simulator.run(
                compiled_qc1,
                shots=1,
                memory=True
            ).result().data()['statevector'] #.result().get_statevector()
            # initial_statevector=full_state,
            
            # 直接操作状态向量（避免多次初始化）
            result = np.array(result)
            evolved = result[-2**total_mode:]
            norm = np.linalg.norm(evolved)
            #print('temp norm 2', np.linalg.norm(evolved))
            evolved /= norm
        
        current_state = evolved
        print('temp norm 2', np.linalg.norm(current_state))
        # --------------------------
        # 记录结果
        # --------------------------
        record_comp[0,:,i+1] = evolved[basis]

        record_comp[1,:,i+1] = expm(-1j*ham2*t_list[i+1]) @ initial_pick
        record_comp[1,:,i+1] /= np.linalg.norm(record_comp[1,:,i+1])
    
    record_comp = np.abs(record_comp)**2
    # 这种pick方法是不对的。
    fig, ax = plt.subplots(1,2,sharex=True, sharey=True)
    for i in range(len(basis)//2):
        ax[0].plot(t_list, record_comp[0, i], 'o-')
        ax[1].plot(t_list, record_comp[1, i], 'o-')
    plt.show()


class CircuitSystem:
    def __init__(self, num_xmodes: int, num_ymodes: int, total_particles: int, t_list: list, gamma:float=0.2, u:float=-1.5,U:float=0.1):
        self.num_x = num_xmodes
        self.num_y = num_ymodes
        self.gamma = gamma
        self.u = u
        self.U = U
        self.total_particles = total_particles
        self.sublat = 2 # 一个元胞中包含A，B两个原子。
        self.total_mode = 2*self.num_x*self.num_y
        self.total_bit = self.total_mode + 1
        self.t = t_list[1]

        self.qc = QuantumCircuit(self.total_mode)  # Hermitian evolution.
        self.dissi_circuits = []
    
    def add_HermtianCircuit(self):
        # diagonal term.
        H0 = np.array([[0, 0],[0, self.u]], dtype=np.complex128)
        U0 = expm(-1j*H0*self.t) # A site
        U0_dagger = expm(1j*H0*self.t) # B site

        # y direction.
        Hx = np.zeros((4, 4), dtype=np.complex128)
        Hx[1, 2] = 1/2
        Hx[2, 1] = 1/2
        Ux = expm(-1j * Hx * self.t)
        Ux_dagger = expm(1j * Hx * self.t)

        HxAA_y = np.kron(sig_p, np.kron(sig_Z, sig_m))/2
        UxAA_y = expm(-1j * (HxAA_y+HxAA_y.T) * self.t)
        UxAA_y_dagger = expm(1j * (HxAA_y+HxAA_y.T) * self.t)

        HxAB_y = np.kron(sig_p, np.kron(sig_Z, np.kron(sig_Z, sig_m)))/2
        UxAB_y_dagger = expm(1j * (HxAB_y+HxAB_y.T) * self.t)

        # x direction hopping.
        Z_string = np.array([1.], dtype=np.complex128)
        for i in range(2*self.num_y-2):
            Z_string = np.kron(Z_string, sig_Z)

        HyBA_x = 1j * np.kron(sig_p, np.kron(Z_string, sig_m))/2 # 小到大系数为i
        UyBA_x = expm(-1j * (HyBA_x+np.conj(HyBA_x).T) * self.t)

        HyAB_x = 1j * np.kron(sig_p, np.kron(sig_Z, np.kron(Z_string, np.kron(sig_Z, sig_m))))/2 # 小到大系数为i
        UyAB_x = expm(-1j * (HyAB_x+np.conj(HyAB_x).T) * self.t)

        HxAA_x = np.kron(sig_p, np.kron(Z_string, np.kron(sig_Z, sig_m)))/2 # 小到大系数为i
        UxAA_x = expm(-1j * (HxAA_x+np.conj(HxAA_x).T) * self.t)
        UxAA_x_dagger = expm(1j * (HxAA_x+np.conj(HxAA_x).T) * self.t)

        # onsite
        for i in range(0, self.total_mode, 2):
            self.qc.unitary(Operator(U0), [i], label='U0')
        for i in range(1, self.total_mode, 2):
            self.qc.unitary(Operator(U0_dagger), [i], label='U0_d')
        # x direction.
        for i in range(0, self.total_mode, 2):
            # A->A in x direction
            if i+2*self.num_y<self.total_mode:
                self.qc.unitary(Operator(UxAA_x), list(range(i, i+2*self.num_y+1)), label='Ux')
                self.qc.unitary(Operator(UyAB_x), list(range(i, i+2*self.num_y+2)), label='Uy') # 也可能是dagger
        for i in range(1, self.total_mode, 2):
            # B->B in x direction
            if i+2*self.num_y<self.total_mode:
                self.qc.unitary(Operator(UxAA_x_dagger), list(range(i, i+2*self.num_y+1)), label='Ux_d')
                self.qc.unitary(Operator(UyBA_x), list(range(i, i+2*self.num_y)), label='Uy')
        # y direction
        for i in range(0, self.total_mode, 2):
            if i not in list(range(2*self.num_y-2,self.total_mode,2*self.num_y)):
                #print(i,self.num_x, self.num_y, self.total_mode)
                self.qc.unitary(Operator(UxAA_y), [i, i+1, i+2], label='Ux')
                self.qc.unitary(Operator(UxAB_y_dagger), [i, i+1, i+2, i+3], label='Ux_d') # 也可能是dagger
        for i in range(1, self.total_mode, 2):
            if i not in list(range(2*self.num_y-1,self.total_mode,2*self.num_y)):
                self.qc.unitary(Operator(UxAA_y_dagger), [i, i+1, i+2], label='Ux_d')
                self.qc.unitary(Operator(Ux), [i, i+1], label='Ux') # 也可能是dagger # Ux是对的，刚好没有Jordan变换的负号。

    def nonHermitian_Circuit(self):
        # measurements 
        alpha = np.exp(-self.t*self.gamma)
        beta = np.sqrt(1-alpha**2)
        UM = np.identity(2**2, dtype=np.complex128)
        UM[1:3, 1:3] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128)
        # 增益耗散取决于辅助比特制备在什么量子态。
        
        dissi_site = sorted([i for i in range(0, self.total_mode, 2*self.num_y)] + [i for i in range(1, self.total_mode, 2*self.num_y)])
        # 预编译耗散操作电路模板
        for site in dissi_site:
            dissi_qc = QuantumCircuit(self.total_bit)
            #dissi_qc.x(total_bit-1)
            dissi_qc.unitary(Operator(UM), [site, self.total_bit-1])
            #dissi_qc.save_state()
            # dissi_circuits.append(transpile(dissi_qc, simulator))
            self.dissi_circuits.append(dissi_qc)

    def add_interaction_intra(self):
        # 加入相互作用
        #U = 0.4  # 粒子数守恒的相互作用。
        print(self.U)
        H_inter1 = np.zeros((4,4), dtype=np.complex128)
        H_inter1[3,3] = self.U
        U_inter = expm(-1j * H_inter1 * self.t)

        for i in range(0, self.total_mode, 2):
            self.qc.unitary(Operator(U_inter), [i, i+1], label='UI')
    
    def add_interaction_inter(self):
        print(self.U)
        H_inter1 = np.zeros((4,4), dtype=np.complex128)
        H_inter1[3,3] = self.U
        U_inter = expm(-1j * H_inter1 * self.t)
        # interaction 2
        for i in range(0, self.total_mode, 2):
            if i not in list(range(2*self.num_y-2,self.total_mode,2*self.num_y)):
                self.qc.unitary(Operator(U_inter), [i, i+2], label='UI')
                self.qc.unitary(Operator(U_inter), [i, i+3], label='UI')
                self.qc.unitary(Operator(U_inter), [i+1, i+2], label='UI')
                self.qc.unitary(Operator(U_inter), [i+1, i+3], label='UI')
        for i in range(0, self.total_mode, 2):
            # A->A in x direction
            distance = 2*self.num_y
            if i+distance<self.total_mode:
                self.qc.unitary(Operator(U_inter), [i, i+distance], label='UI')
                self.qc.unitary(Operator(U_inter), [i, i+distance+1], label='UI')
                self.qc.unitary(Operator(U_inter), [i+1, i+distance], label='UI')
                self.qc.unitary(Operator(U_inter), [i+1, i+distance+1], label='UI')

    def evolve(self, t_list, init_state_record=[0,1]): # 默认在(1,1)的元胞激发两个粒子。
        QWZ2 = ParticleConservedSystem(self.num_x, self.num_y, self.total_particles)
        QWZ2.add_onsite_term(self.u)
        QWZ2.add_xhopping_term()
        QWZ2.add_yhopping_term()
        QWZ2.add_interaction_inter(self.U)
        QWZ2.add_NH(self.gamma)
        print(self.qc)

        prob_list = np.zeros(len(t_list))
        prob_list[0] = 1.

        ham2 = QWZ2.hamiltonian
        self.basis = QWZ2.basis

        print('The system size:', ham2.shape, self.total_particles)

        simulator = AerSimulator(
            method='statevector',
            device='GPU' if 'GPU' in AerSimulator().available_devices() else 'CPU',
            max_parallel_threads=1,
        )# 'matrix_product_state', 'statevector',
        compiled_qc1 = transpile(self.qc, simulator)

        # device='GPU' if AerSimulator().available_devices(True)['name'] == 'GPU' else 'CPU', # optimization_level=2,
        init_state = np.zeros(2**self.total_bit, dtype=np.complex128)
        #init_state[(1<<4)+1] = 1.0  # 对应x([0,4])的激发态
        # init_state[(1<<2)+1] = 1.0  # 对应x([0,4])的激发态
        init_label = 0
        for idx in init_state_record:
            init_label += 1<<idx
        init_state[init_label] = 1.0
        initial_pick = init_state[self.basis].copy()
        print('Initial Norm:', np.linalg.norm(initial_pick))

        record_comp = np.zeros((2, len(self.basis), len(t_list)), dtype=np.complex128)
        record_comp[0,:,0] = initial_pick
        record_comp[1,:,0] = initial_pick  # digital

        # record_comp = np.zeros((2, len(basis), len(t_list)), dtype=np.complex128)
        # record_comp[0,:,0] = initial_pick
        # record_comp[1,:,0] = initial_pick  # digital
        # state = state[-2**(self.total_mode):]

        current_state = init_state.copy()

        #print(compiled_qc.draw())
        for i in range(len(t_list)-1):
            print(i)
            # --------------------------
            # 主演化过程
            # --------------------------
            # 直接使用模拟器进行状态演化
            #print(current_state[:2**self.total_mode])
            full_circuit = QuantumCircuit(self.total_mode)
            # full_circuit.initialize(current_state[:2**self.total_mode], list(range(0,self.total_mode,1)))
            full_circuit.set_statevector(current_state[:2**self.total_mode])
            full_circuit = full_circuit.compose(self.qc)
            # full_circuit = full_circuit.compose(compiled_qc1)
            full_circuit.save_state()
            compiled_qc = transpile(full_circuit, simulator)
            #print(compiled_qc.draw())
            evolved = simulator.run(
                compiled_qc,
                shots=1,
                initial_statevector=current_state[:2**self.total_mode],
                memory=True
            ).result().data()['statevector']   # get_statevector() # data['statevector'] 
            # memory=True
            # --------------------------
            # 耗散过程优化
            # --------------------------
            # 如果列表为空不会执行。
            temp_prob = 1.
            for j, dissi_qc in enumerate(self.dissi_circuits):
                # 合并耗散操作和测量
                # print('j', j)
                full_state = np.kron(np.array([0,1], dtype=np.complex128),evolved)  # 添加辅助比特
                full_circuit = QuantumCircuit(self.total_bit)
                # full_circuit.initialize(full_state, list(range(0,self.total_bit,1)))
                full_circuit.set_statevector(full_state) # 这个更快！！是对全部初始化。
                # full_circuit.compose(dissi_qc)
                full_circuit = full_circuit.compose(dissi_qc)
                full_circuit.save_state()
                compiled_qc1 = transpile(full_circuit, simulator)
                result = simulator.run(
                    compiled_qc1,
                    shots=1,
                    memory=True
                ).result().data()['statevector'] #.result().get_statevector()
                # initial_statevector=full_state,
                
                # 直接操作状态向量（避免多次初始化）
                result = np.array(result)
                evolved = result[-2**self.total_mode:]
                norm = np.linalg.norm(evolved)
                #print('temp norm 2', np.linalg.norm(evolved))
                evolved /= norm
                temp_prob *= norm**2
            
            prob_list[i+1] = prob_list[i] * temp_prob
            
            result = np.array(evolved)
            evolved = result[-2**self.total_mode:]
            norm = np.linalg.norm(evolved)
            #print('temp norm 2', np.linalg.norm(evolved))
            evolved /= norm
            
            current_state = evolved
            print('temp norm 2', np.linalg.norm(current_state))
            # --------------------------
            # 记录结果
            # --------------------------
            # print(self.basis)
            record_comp[0,:,i+1] = evolved[self.basis]

            record_comp[1,:,i+1] = expm(-1j*ham2*t_list[i+1]) @ initial_pick
            record_comp[1,:,i+1] /= np.linalg.norm(record_comp[1,:,i+1])
        
        popu_list = np.zeros((2, self.total_mode, len(t_list)))
        for i in range(len(t_list)):
            popu_list[0,:,i] = self.Calcu_Occup(record_comp[0,:,i])
            popu_list[1,:,i] = self.Calcu_Occup(record_comp[1,:,i])
        record_comp = np.abs(record_comp)**2
        return record_comp, popu_list, prob_list
    
    def Calcu_Occup(self, eigen_state):
        # calculate the n for a given state
        #assert len(eigen_state) == self.dimension
        population = np.zeros(self.total_mode)
        for idx, state in enumerate(self.basis):
            #print(eigen_state[idx]**2, np.array([ReadBit(state, i) for i in range(self.total_mode)]))
            population += np.abs(eigen_state[idx])**2 * np.array([ReadBit(state, i) for i in range(self.total_mode)])
        return population

class CircuitSystem2:
    def __init__(self, num_xmodes: int, num_ymodes: int, total_particles: int, t_list: list, gamma:float=0.2, u:float=-1.5,U:float=0.1):
        self.num_x = num_xmodes
        self.num_y = num_ymodes
        self.gamma = gamma
        self.u = u
        self.U = U
        self.total_particles = total_particles
        self.sublat = 2 # 一个元胞中包含A，B两个原子。
        self.total_mode = 2*self.num_x*self.num_y
        self.total_bit = self.total_mode + 1
        self.t = t_list[1]

        self.qc = QuantumCircuit(self.total_mode)  # Hermitian evolution.
        self.dissi_circuits = []
    
    def add_HermtianCircuit(self):
        # diagonal term.
        H0 = np.array([[0, 0],[0, self.u]], dtype=np.complex128)
        U0 = expm(-1j*H0*self.t) # A site
        U0_dagger = expm(1j*H0*self.t) # B site

        # y direction.
        Hx = np.zeros((4, 4), dtype=np.complex128)
        Hx[1, 2] = 1/2
        Hx[2, 1] = 1/2
        Ux = expm(-1j * Hx * self.t)
        Ux_dagger = expm(1j * Hx * self.t)

        HxAA_y = np.kron(sig_p, np.kron(sig_Z, sig_m))/2
        UxAA_y = expm(-1j * (HxAA_y+HxAA_y.T) * self.t)
        UxAA_y_dagger = expm(1j * (HxAA_y+HxAA_y.T) * self.t)

        HxAB_y = np.kron(sig_p, np.kron(sig_Z, np.kron(sig_Z, sig_m)))/2
        UxAB_y_dagger = expm(1j * (HxAB_y+HxAB_y.T) * self.t)

        # x direction hopping.
        Z_string = np.array([1.], dtype=np.complex128)
        for i in range(2*self.num_y-2):
            Z_string = np.kron(Z_string, sig_Z)

        HyBA_x = 1j * np.kron(sig_p, np.kron(Z_string, sig_m))/2 # 小到大系数为i
        UyBA_x = expm(-1j * (HyBA_x+np.conj(HyBA_x).T) * self.t)

        HyAB_x = 1j * np.kron(sig_p, np.kron(sig_Z, np.kron(Z_string, np.kron(sig_Z, sig_m))))/2 # 小到大系数为i
        UyAB_x = expm(-1j * (HyAB_x+np.conj(HyAB_x).T) * self.t)

        HxAA_x = np.kron(sig_p, np.kron(Z_string, np.kron(sig_Z, sig_m)))/2 # 小到大系数为i
        UxAA_x = expm(-1j * (HxAA_x+np.conj(HxAA_x).T) * self.t)
        UxAA_x_dagger = expm(1j * (HxAA_x+np.conj(HxAA_x).T) * self.t)

        # onsite
        for i in range(0, self.total_mode, 2):
            self.qc.unitary(Operator(U0), [i], label='U0')
        for i in range(1, self.total_mode, 2):
            self.qc.unitary(Operator(U0_dagger), [i], label='U0_d')
        # x direction.
        for i in range(0, self.total_mode, 2):
            # A->A in x direction
            if i+2*self.num_y<self.total_mode:
                self.qc.unitary(Operator(UxAA_x), list(range(i, i+2*self.num_y+1)), label='Ux')
                self.qc.unitary(Operator(UyAB_x), list(range(i, i+2*self.num_y+2)), label='Uy') # 也可能是dagger
        for i in range(1, self.total_mode, 2):
            # B->B in x direction
            if i+2*self.num_y<self.total_mode:
                self.qc.unitary(Operator(UxAA_x_dagger), list(range(i, i+2*self.num_y+1)), label='Ux_d')
                self.qc.unitary(Operator(UyBA_x), list(range(i, i+2*self.num_y)), label='Uy')
        # y direction
        for i in range(0, self.total_mode, 2):
            if i not in list(range(2*self.num_y-2,self.total_mode,2*self.num_y)):
                #print(i,self.num_x, self.num_y, self.total_mode)
                self.qc.unitary(Operator(UxAA_y), [i, i+1, i+2], label='Ux')
                self.qc.unitary(Operator(UxAB_y_dagger), [i, i+1, i+2, i+3], label='Ux_d') # 也可能是dagger
        for i in range(1, self.total_mode, 2):
            if i not in list(range(2*self.num_y-1,self.total_mode,2*self.num_y)):
                self.qc.unitary(Operator(UxAA_y_dagger), [i, i+1, i+2], label='Ux_d')
                self.qc.unitary(Operator(Ux), [i, i+1], label='Ux') # 也可能是dagger # Ux是对的，刚好没有Jordan变换的负号。

    def nonHermitian_Circuit(self):
        # measurements 
        alpha = np.exp(-self.t*self.gamma)
        beta = np.sqrt(1-alpha**2)
        UM = np.identity(2**2, dtype=np.complex128)
        UM[1:3, 1:3] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128)
        # 增益耗散取决于辅助比特制备在什么量子态。
        
        dissi_site = sorted([i for i in range(0, self.total_mode, 2*self.num_y)] + [i for i in range(1, self.total_mode, 2*self.num_y)])
        # 预编译耗散操作电路模板
        for site in dissi_site:
            dissi_qc = QuantumCircuit(self.total_bit)
            #dissi_qc.x(total_bit-1)
            dissi_qc.unitary(Operator(UM), [site, self.total_bit-1])
            #dissi_qc.save_state()
            # dissi_circuits.append(transpile(dissi_qc, simulator))
            self.dissi_circuits.append(dissi_qc)

    def add_interaction_intra(self):
        # 加入相互作用
        #U = 0.4  # 粒子数守恒的相互作用。
        print(self.U)
        H_inter1 = np.zeros((4,4), dtype=np.complex128)
        H_inter1[3,3] = self.U
        U_inter = expm(-1j * H_inter1 * self.t)

        for i in range(0, self.total_mode, 2):
            self.qc.unitary(Operator(U_inter), [i, i+1], label='UI')
    
    def add_interaction_inter(self):
        print(self.U)
        H_inter1 = np.zeros((4,4), dtype=np.complex128)
        H_inter1[3,3] = self.U
        U_inter = expm(-1j * H_inter1 * self.t)
        # interaction 2
        for i in range(0, self.total_mode, 2):
            if i not in list(range(2*self.num_y-2,self.total_mode,2*self.num_y)):
                self.qc.unitary(Operator(U_inter), [i, i+2], label='UI')
                self.qc.unitary(Operator(U_inter), [i, i+3], label='UI')
                self.qc.unitary(Operator(U_inter), [i+1, i+2], label='UI')
                self.qc.unitary(Operator(U_inter), [i+1, i+3], label='UI')
        for i in range(0, self.total_mode, 2):
            # A->A in x direction
            distance = 2*self.num_y
            if i+distance<self.total_mode:
                self.qc.unitary(Operator(U_inter), [i, i+distance], label='UI')
                self.qc.unitary(Operator(U_inter), [i, i+distance+1], label='UI')
                self.qc.unitary(Operator(U_inter), [i+1, i+distance], label='UI')
                self.qc.unitary(Operator(U_inter), [i+1, i+distance+1], label='UI')

    def evolve(self, t_list, init_state_record=[0,1]): # 默认在(1,1)的元胞激发两个粒子。
        QWZ2 = ParticleConservedSystem(self.num_x, self.num_y, self.total_particles)
        QWZ2.add_onsite_term(self.u)
        QWZ2.add_xhopping_term()
        QWZ2.add_yhopping_term()
        QWZ2.add_interaction_inter(self.U)
        QWZ2.add_NH(self.gamma)
        print(self.qc)

        prob_list = np.zeros(len(t_list))
        prob_list[0] = 1.

        ham2 = QWZ2.hamiltonian
        self.basis = QWZ2.basis

        print('The system size:', ham2.shape, self.total_particles)

        simulator = AerSimulator(
            method='matrix_product_state',
            device='GPU' if 'GPU' in AerSimulator().available_devices() else 'CPU',
            max_parallel_threads=1,
        )# 'matrix_product_state', 'statevector',
        compiled_qc1 = transpile(self.qc, simulator)

        # device='GPU' if AerSimulator().available_devices(True)['name'] == 'GPU' else 'CPU', # optimization_level=2,
        init_state = np.zeros(2**self.total_bit, dtype=np.complex128)
        #init_state[(1<<4)+1] = 1.0  # 对应x([0,4])的激发态
        # init_state[(1<<2)+1] = 1.0  # 对应x([0,4])的激发态
        init_label = 0
        for idx in init_state_record:
            init_label += 1<<idx
        init_state[init_label] = 1.0
        initial_pick = init_state[self.basis].copy()
        print('Initial Norm:', np.linalg.norm(initial_pick))

        record_comp = np.zeros((2, len(self.basis), len(t_list)), dtype=np.complex128)
        record_comp[0,:,0] = initial_pick
        record_comp[1,:,0] = initial_pick  # digital

        # record_comp = np.zeros((2, len(basis), len(t_list)), dtype=np.complex128)
        # record_comp[0,:,0] = initial_pick
        # record_comp[1,:,0] = initial_pick  # digital
        # state = state[-2**(self.total_mode):]

        current_state = init_state.copy()

        #print(compiled_qc.draw())
        for i in range(len(t_list)-1):
            print(i)
            # --------------------------
            # 主演化过程
            # --------------------------
            # 直接使用模拟器进行状态演化
            #print(current_state[:2**self.total_mode])
            full_circuit = QuantumCircuit(self.total_mode)
            # full_circuit.initialize(current_state[:2**self.total_mode], list(range(0,self.total_mode,1)))
            # full_circuit.set_statevector(current_state[:2**self.total_mode])
            full_circuit = full_circuit.compose(self.qc)
            # full_circuit = full_circuit.compose(compiled_qc1)
            # full_circuit.save_statevector(label='my_sv')
            compiled_qc = transpile(full_circuit, simulator)
            #print(compiled_qc.draw())
            evolved = simulator.run(
                compiled_qc,
                shots=1,
                initial_statevector=current_state[:2**self.total_mode],
                memory=True
            ).result().data(0)['my_sv']   # get_statevector() # data['statevector'] 
            # memory=True
            # --------------------------
            # 耗散过程优化
            # --------------------------
            # 如果列表为空不会执行。
            temp_prob = 1.
            for j, dissi_qc in enumerate(self.dissi_circuits):
                # 合并耗散操作和测量
                # print('j', j)
                full_state = np.kron(np.array([0,1], dtype=np.complex128),evolved)  # 添加辅助比特
                full_circuit = QuantumCircuit(self.total_bit)
                # full_circuit.initialize(full_state, list(range(0,self.total_bit,1)))
                full_circuit.set_statevector(full_state) # 这个更快！！是对全部初始化。
                # full_circuit.compose(dissi_qc)
                full_circuit = full_circuit.compose(dissi_qc)
                full_circuit.save_state()
                compiled_qc1 = transpile(full_circuit, simulator)
                result = simulator.run(
                    compiled_qc1,
                    shots=1,
                    memory=True
                ).result().data(0)['my_sv'] #.result().get_statevector()
                # initial_statevector=full_state,
                
                # 直接操作状态向量（避免多次初始化）
                result = np.array(result)
                evolved = result[-2**self.total_mode:]
                norm = np.linalg.norm(evolved)
                #print('temp norm 2', np.linalg.norm(evolved))
                evolved /= norm
                temp_prob *= norm**2
            
            prob_list[i+1] = prob_list[i] * temp_prob
            
            result = np.array(evolved)
            evolved = result[-2**self.total_mode:]
            norm = np.linalg.norm(evolved)
            #print('temp norm 2', np.linalg.norm(evolved))
            evolved /= norm
            
            current_state = evolved
            print('temp norm 2', np.linalg.norm(current_state))
            # --------------------------
            # 记录结果
            # --------------------------
            # print(self.basis)
            record_comp[0,:,i+1] = evolved[self.basis]

            record_comp[1,:,i+1] = expm(-1j*ham2*t_list[i+1]) @ initial_pick
            record_comp[1,:,i+1] /= np.linalg.norm(record_comp[1,:,i+1])
        
        popu_list = np.zeros((2, self.total_mode, len(t_list)))
        for i in range(len(t_list)):
            popu_list[0,:,i] = self.Calcu_Occup(record_comp[0,:,i])
            popu_list[1,:,i] = self.Calcu_Occup(record_comp[1,:,i])
        record_comp = np.abs(record_comp)**2
        return record_comp, popu_list, prob_list
    
    def Calcu_Occup(self, eigen_state):
        # calculate the n for a given state
        #assert len(eigen_state) == self.dimension
        population = np.zeros(self.total_mode)
        for idx, state in enumerate(self.basis):
            #print(eigen_state[idx]**2, np.array([ReadBit(state, i) for i in range(self.total_mode)]))
            population += np.abs(eigen_state[idx])**2 * np.array([ReadBit(state, i) for i in range(self.total_mode)])
        return population



class CircuitSystem_Full:
    def __init__(self, num_xmodes: int, num_ymodes: int, total_particles: int, t_list: list, gamma:float=0.2, u:float=-1.5,U:float=0.1):
        self.num_x = num_xmodes
        self.num_y = num_ymodes
        self.gamma = gamma
        self.u = u
        self.U = U
        self.total_particles = total_particles
        self.sublat = 2 # 一个元胞中包含A，B两个原子。
        self.total_mode = 2*self.num_x*self.num_y
        self.total_bit = self.total_mode + 1
        self.t = t_list[1]

        self.qc = QuantumCircuit(self.total_mode + 1)  # Hermitian evolution.
        self.dissi_circuits = []
    
    def add_HermtianCircuit(self):
        # diagonal term.
        H0 = np.array([[0, 0],[0, self.u]], dtype=np.complex128)
        U0 = expm(-1j*H0*self.t) # A site
        U0_dagger = expm(1j*H0*self.t) # B site

        # y direction.
        Hx = np.zeros((4, 4), dtype=np.complex128)
        Hx[1, 2] = 1/2
        Hx[2, 1] = 1/2
        Ux = expm(-1j * Hx * self.t)
        Ux_dagger = expm(1j * Hx * self.t)

        HxAA_y = np.kron(sig_p, np.kron(sig_Z, sig_m))/2
        UxAA_y = expm(-1j * (HxAA_y+HxAA_y.T) * self.t)
        UxAA_y_dagger = expm(1j * (HxAA_y+HxAA_y.T) * self.t)

        HxAB_y = np.kron(sig_p, np.kron(sig_Z, np.kron(sig_Z, sig_m)))/2
        UxAB_y_dagger = expm(1j * (HxAB_y+HxAB_y.T) * self.t)

        # x direction hopping.
        Z_string = np.array([1.], dtype=np.complex128)
        for i in range(2*self.num_y-2):
            Z_string = np.kron(Z_string, sig_Z)

        HyBA_x = 1j * np.kron(sig_p, np.kron(Z_string, sig_m))/2 # 小到大系数为i
        UyBA_x = expm(-1j * (HyBA_x+np.conj(HyBA_x).T) * self.t)

        HyAB_x = 1j * np.kron(sig_p, np.kron(sig_Z, np.kron(Z_string, np.kron(sig_Z, sig_m))))/2 # 小到大系数为i
        UyAB_x = expm(-1j * (HyAB_x+np.conj(HyAB_x).T) * self.t)

        HxAA_x = np.kron(sig_p, np.kron(Z_string, np.kron(sig_Z, sig_m)))/2 # 小到大系数为i
        UxAA_x = expm(-1j * (HxAA_x+np.conj(HxAA_x).T) * self.t)
        UxAA_x_dagger = expm(1j * (HxAA_x+np.conj(HxAA_x).T) * self.t)

        # onsite
        for i in range(0, self.total_mode, 2):
            self.qc.unitary(Operator(U0), [i], label='U0')
        for i in range(1, self.total_mode, 2):
            self.qc.unitary(Operator(U0_dagger), [i], label='U0_d')
        # x direction.
        for i in range(0, self.total_mode, 2):
            # A->A in x direction
            if i+2*self.num_y<self.total_mode:
                self.qc.unitary(Operator(UxAA_x), list(range(i, i+2*self.num_y+1)), label='Ux')
                self.qc.unitary(Operator(UyAB_x), list(range(i, i+2*self.num_y+2)), label='Uy') # 也可能是dagger
        for i in range(1, self.total_mode, 2):
            # B->B in x direction
            if i+2*self.num_y<self.total_mode:
                self.qc.unitary(Operator(UxAA_x_dagger), list(range(i, i+2*self.num_y+1)), label='Ux_d')
                self.qc.unitary(Operator(UyBA_x), list(range(i, i+2*self.num_y)), label='Uy')
        # y direction
        for i in range(0, self.total_mode, 2):
            if i not in list(range(2*self.num_y-2,self.total_mode,2*self.num_y)):
                #print(i,self.num_x, self.num_y, self.total_mode)
                self.qc.unitary(Operator(UxAA_y), [i, i+1, i+2], label='Ux')
                self.qc.unitary(Operator(UxAB_y_dagger), [i, i+1, i+2, i+3], label='Ux_d') # 也可能是dagger
        for i in range(1, self.total_mode, 2):
            if i not in list(range(2*self.num_y-1,self.total_mode,2*self.num_y)):
                self.qc.unitary(Operator(UxAA_y_dagger), [i, i+1, i+2], label='Ux_d')
                self.qc.unitary(Operator(Ux), [i, i+1], label='Ux') # 也可能是dagger # Ux是对的，刚好没有Jordan变换的负号。

    def nonHermitian_Circuit(self):
        # measurements 
        alpha = np.exp(-self.t*self.gamma)
        beta = np.sqrt(1-alpha**2)
        UM = np.identity(2**2, dtype=np.complex128)
        UM[1:3, 1:3] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128)
        # 增益耗散取决于辅助比特制备在什么量子态。
        
        dissi_site = sorted([i for i in range(0, self.total_mode, 2*self.num_y)] + [i for i in range(1, self.total_mode, 2*self.num_y)])
        # 预编译耗散操作电路模板
        for site in dissi_site:
            dissi_qc = QuantumCircuit(self.total_bit)
            #dissi_qc.x(total_bit-1)
            dissi_qc.unitary(Operator(UM), [site, self.total_bit-1])
            #dissi_qc.save_state()
            # dissi_circuits.append(transpile(dissi_qc, simulator))
            self.dissi_circuits.append(dissi_qc)

    def add_interaction_intra(self):
        # 加入相互作用
        #U = 0.4  # 粒子数守恒的相互作用。
        print(self.U)
        H_inter1 = np.zeros((4,4), dtype=np.complex128)
        H_inter1[3,3] = self.U
        U_inter = expm(-1j * H_inter1 * self.t)

        for i in range(0, self.total_mode, 2):
            self.qc.unitary(Operator(U_inter), [i, i+1], label='UI')
    
    def add_interaction_inter(self):
        print(self.U)
        H_inter1 = np.zeros((4,4), dtype=np.complex128)
        H_inter1[3,3] = self.U
        U_inter = expm(-1j * H_inter1 * self.t)
        # interaction 2
        for i in range(0, self.total_mode, 2):
            if i not in list(range(2*self.num_y-2,self.total_mode,2*self.num_y)):
                self.qc.unitary(Operator(U_inter), [i, i+2], label='UI')
                self.qc.unitary(Operator(U_inter), [i, i+3], label='UI')
                self.qc.unitary(Operator(U_inter), [i+1, i+2], label='UI')
                self.qc.unitary(Operator(U_inter), [i+1, i+3], label='UI')
        for i in range(0, self.total_mode, 2):
            # A->A in x direction
            distance = 2*self.num_y
            if i+distance<self.total_mode:
                self.qc.unitary(Operator(U_inter), [i, i+distance], label='UI')
                self.qc.unitary(Operator(U_inter), [i, i+distance+1], label='UI')
                self.qc.unitary(Operator(U_inter), [i+1, i+distance], label='UI')
                self.qc.unitary(Operator(U_inter), [i+1, i+distance+1], label='UI')

    def evolve(self, t_list, init_state_record=[0,1]): # 默认在(1,1)的元胞激发两个粒子。
        project_M = Operator(np.array([[0,0.],[0.,1.]]))
        temp_register = len(t_list[:-1])*2*self.num_x
        QWZ2 = ParticleConservedSystem(self.num_x, self.num_y, self.total_particles)
        QWZ2.add_onsite_term(self.u)
        QWZ2.add_xhopping_term()
        QWZ2.add_yhopping_term()
        QWZ2.add_interaction_inter(self.U)
        QWZ2.add_NH(self.gamma)
        ham2 = QWZ2.hamiltonian
        self.basis = QWZ2.basis
        state_index = QWZ2.state_index
        init_state_record = [0,1]  # initial state. Excited which site
        print(self.qc)


        if self.qc == QuantumCircuit(self.total_mode+1):
            self.add_HermtianCircuit()
            self.add_interaction_inter()
            self.nonHermitian_Circuit()

        temp_register = len(t_list[:-1])*2*self.num_x
        full_circuit = QuantumCircuit(self.total_mode+1, temp_register+2*self.num_x*self.num_y)
        full_circuit.x(init_state_record + [2*self.num_x*self.num_y])
        for i in range(len(t_list[:-1])):
            full_circuit = full_circuit.compose(self.qc)
            for j, dissi_qc in enumerate(self.dissi_circuits):
                # # 合并耗散操作和测量
                # # print('j', j)
                # full_state = np.kron(np.array([0,1], dtype=np.complex128),evolved)  # 添加辅助比特
                # full_circuit = QuantumCircuit(self.total_bit)
                # # full_circuit.initialize(full_state, list(range(0,self.total_bit,1)))
                # full_circuit.set_statevector(full_state) # 这个更快！！是对全部初始化。
                # # full_circuit.compose(dissi_qc)
                full_circuit = full_circuit.compose(dissi_qc)
                # full_circuit.save_state()
                full_circuit.measure(self.total_bit-1, 2*self.num_x*i+j)
                # full_circuit.append(project_M, [self.total_bit])
                
        # measurement the final data
        for i in range(2*self.num_x*self.num_y):
            full_circuit.measure(i, temp_register+i)

        # from qiskit.visualization import plot_bloch_multivector, plot_histogram
        # print(full_circuit)
        simulator = AerSimulator(
            method='matrix_product_state',
            device='GPU' if 'GPU' in AerSimulator().available_devices() else 'CPU',
            max_parallel_threads=1
        ) # 'statevector',
        # compiled_qc1 = transpile(full_circuit, simulator)
        # result = simulator.run(
        #     compiled_qc1,
        #     shots=1,
        #     memory=True
        # ).result().data()['statevector'] #.result().get_statevector()

        # print(result)
        simulator_measurement = Aer.get_backend('qasm_simulator')
        qc_measure = transpile(full_circuit, simulator)
        shots = 10**3
        result_measurement = simulator.run(qc_measure, shots=shots).result()
        counts = result_measurement.get_counts()
        # print("测量结果：", counts)
        # plot_histogram(counts)

        # pick up the data.
        dict_result = dict()
        for key, value in counts.items():
            if key[-temp_register:] == '1'*temp_register:
                dict_result[key[:2*self.num_x*self.num_y]] = value  # from qubit 8  to qubit 0 . 因为先测0，所以结果在右边
        print("测量结果：", dict_result)
        print('Total', sum(dict_result.values()))

        total_event = sum(dict_result.values())
        print('The real trajectory is:', total_event)
        if total_event!=0:
            n_popu = np.zeros(self.total_mode)
            for key, value in dict_result.items():
                for i in range(1, self.total_mode+1):
                    n_popu[i-1] += int(key[-i])*value
            n_popu = n_popu / total_event


        # 这是为了和经典的进行比较
        init_state = np.zeros(len(self.basis), dtype=np.complex128)
        #init_state[(1<<4)+1] = 1.0  # 对应x([0,4])的激发态
        # init_state[(1<<2)+1] = 1.0  # 对应x([0,4])的激发态
        init_label = 0
        for idx in init_state_record[::-1]:
            init_label += 1<<idx
        print(init_label)
        init_state[state_index[init_label]] = 1.0
        initial_pick = init_state#init_state[self.basis].copy()

        final_pick = expm(-1j*ham2*t_list[-1]) @ initial_pick
        print(len(final_pick), np.linalg.norm(final_pick))
        final_pick /= np.linalg.norm(final_pick)
        a = self.Calcu_Occup(final_pick)

        print('The results of the circuit:', n_popu)
        print('The results of exact calculation:', a)



        # prob_list = np.zeros(len(t_list))
        # prob_list[0] = 1.

        # ham2 = QWZ2.hamiltonian
        # self.basis = QWZ2.basis

        # print('The system size:', ham2.shape, self.total_particles)

        # simulator = AerSimulator(
        #     method='statevector',
        #     device='GPU' if 'GPU' in AerSimulator().available_devices() else 'CPU',
        #     max_parallel_threads=1,
        # )
        # compiled_qc1 = transpile(self.qc, simulator)

        # # device='GPU' if AerSimulator().available_devices(True)['name'] == 'GPU' else 'CPU', # optimization_level=2,
        # init_state = np.zeros(2**self.total_bit, dtype=np.complex128)
        # #init_state[(1<<4)+1] = 1.0  # 对应x([0,4])的激发态
        # # init_state[(1<<2)+1] = 1.0  # 对应x([0,4])的激发态
        # init_label = 0
        # for idx in init_state_record:
        #     init_label += 1<<idx
        # init_state[init_label] = 1.0
        # initial_pick = init_state[self.basis].copy()
        # print('Initial Norm:', np.linalg.norm(initial_pick))

        # record_comp = np.zeros((2, len(self.basis), len(t_list)), dtype=np.complex128)
        # record_comp[0,:,0] = initial_pick
        # record_comp[1,:,0] = initial_pick  # digital

        # # record_comp = np.zeros((2, len(basis), len(t_list)), dtype=np.complex128)
        # # record_comp[0,:,0] = initial_pick
        # # record_comp[1,:,0] = initial_pick  # digital
        # # state = state[-2**(self.total_mode):]

        # current_state = init_state.copy()

        # #print(compiled_qc.draw())
        # for i in range(len(t_list)-1):
        #     print(i)
        #     # --------------------------
        #     # 主演化过程
        #     # --------------------------
        #     # 直接使用模拟器进行状态演化
        #     #print(current_state[:2**self.total_mode])
        #     full_circuit = QuantumCircuit(self.total_mode)
        #     # full_circuit.initialize(current_state[:2**self.total_mode], list(range(0,self.total_mode,1)))
        #     full_circuit.set_statevector(current_state[:2**self.total_mode])
        #     full_circuit = full_circuit.compose(self.qc)
        #     # full_circuit = full_circuit.compose(compiled_qc1)
        #     full_circuit.save_state()
        #     compiled_qc = transpile(full_circuit, simulator)
        #     #print(compiled_qc.draw())
        #     evolved = simulator.run(
        #         compiled_qc,
        #         shots=1,
        #         initial_statevector=current_state[:2**self.total_mode],
        #         memory=True
        #     ).result().data()['statevector']   # get_statevector() # data['statevector'] 
        #     # memory=True
        #     # --------------------------
        #     # 耗散过程优化
        #     # --------------------------
        #     # 如果列表为空不会执行。
        #     temp_prob = 1.
        #     for j, dissi_qc in enumerate(self.dissi_circuits):
        #         # 合并耗散操作和测量
        #         # print('j', j)
        #         full_state = np.kron(np.array([0,1], dtype=np.complex128),evolved)  # 添加辅助比特
        #         full_circuit = QuantumCircuit(self.total_bit)
        #         # full_circuit.initialize(full_state, list(range(0,self.total_bit,1)))
        #         full_circuit.set_statevector(full_state) # 这个更快！！是对全部初始化。
        #         # full_circuit.compose(dissi_qc)
        #         full_circuit = full_circuit.compose(dissi_qc)
        #         full_circuit.save_state()
        #         compiled_qc1 = transpile(full_circuit, simulator)
        #         result = simulator.run(
        #             compiled_qc1,
        #             shots=1,
        #             memory=True
        #         ).result().data()['statevector'] #.result().get_statevector()
        #         # initial_statevector=full_state,
                
        #         # 直接操作状态向量（避免多次初始化）
        #         result = np.array(result)
        #         evolved = result[-2**self.total_mode:]
        #         norm = np.linalg.norm(evolved)
        #         #print('temp norm 2', np.linalg.norm(evolved))
        #         evolved /= norm
        #         temp_prob *= norm**2
            
        #     prob_list[i+1] = prob_list[i] * temp_prob
            
        #     result = np.array(evolved)
        #     evolved = result[-2**self.total_mode:]
        #     norm = np.linalg.norm(evolved)
        #     #print('temp norm 2', np.linalg.norm(evolved))
        #     evolved /= norm
            
        #     current_state = evolved
        #     print('temp norm 2', np.linalg.norm(current_state))
        #     # --------------------------
        #     # 记录结果
        #     # --------------------------
        #     # print(self.basis)
        #     record_comp[0,:,i+1] = evolved[self.basis]

        #     record_comp[1,:,i+1] = expm(-1j*ham2*t_list[i+1]) @ initial_pick
        #     record_comp[1,:,i+1] /= np.linalg.norm(record_comp[1,:,i+1])
        
        # popu_list = np.zeros((2, self.total_mode, len(t_list)))
        # for i in range(len(t_list)):
        #     popu_list[0,:,i] = self.Calcu_Occup(record_comp[0,:,i])
        #     popu_list[1,:,i] = self.Calcu_Occup(record_comp[1,:,i])
        # record_comp = np.abs(record_comp)**2
        # return record_comp, popu_list, prob_list
    
    def Calcu_Occup(self, eigen_state):
        # calculate the n for a given state
        #assert len(eigen_state) == self.dimension
        population = np.zeros(self.total_mode)
        for idx, state in enumerate(self.basis):
            #print(eigen_state[idx]**2, np.array([ReadBit(state, i) for i in range(self.total_mode)]))
            population += np.abs(eigen_state[idx])**2 * np.array([ReadBit(state, i) for i in range(self.total_mode)])
        return population


class CircuitSystem_MPS:
    def __init__(self, num_xmodes: int, num_ymodes: int, total_particles: int, t_list: list, gamma:float=0.2, u:float=-1.5,U:float=0.1):
        self.num_x = num_xmodes
        self.num_y = num_ymodes
        self.gamma = gamma
        self.u = u
        self.U = U
        self.total_particles = total_particles
        self.sublat = 2 # 一个元胞中包含A，B两个原子。
        self.total_mode = 2*self.num_x*self.num_y
        self.total_bit = self.total_mode + 1
        self.t = t_list[1]

        self.qc = QuantumCircuit(self.total_mode + 1)  # Hermitian evolution.
        self.dissi_circuits = []
    
    def add_HermtianCircuit(self):
        # diagonal term.
        H0 = np.array([[0, 0],[0, self.u]], dtype=np.complex128)
        U0 = expm(-1j*H0*self.t) # A site
        U0_dagger = expm(1j*H0*self.t) # B site

        # y direction.
        Hx = np.zeros((4, 4), dtype=np.complex128)
        Hx[1, 2] = 1/2
        Hx[2, 1] = 1/2
        Ux = expm(-1j * Hx * self.t)
        Ux_dagger = expm(1j * Hx * self.t)

        HxAA_y = np.kron(sig_p, np.kron(sig_Z, sig_m))/2
        UxAA_y = expm(-1j * (HxAA_y+HxAA_y.T) * self.t)
        UxAA_y_dagger = expm(1j * (HxAA_y+HxAA_y.T) * self.t)

        HxAB_y = np.kron(sig_p, np.kron(sig_Z, np.kron(sig_Z, sig_m)))/2
        UxAB_y_dagger = expm(1j * (HxAB_y+HxAB_y.T) * self.t)

        # x direction hopping.
        Z_string = np.array([1.], dtype=np.complex128)
        for i in range(2*self.num_y-2):
            Z_string = np.kron(Z_string, sig_Z)

        HyBA_x = 1j * np.kron(sig_p, np.kron(Z_string, sig_m))/2 # 小到大系数为i
        UyBA_x = expm(-1j * (HyBA_x+np.conj(HyBA_x).T) * self.t)

        HyAB_x = 1j * np.kron(sig_p, np.kron(sig_Z, np.kron(Z_string, np.kron(sig_Z, sig_m))))/2 # 小到大系数为i
        UyAB_x = expm(-1j * (HyAB_x+np.conj(HyAB_x).T) * self.t)

        HxAA_x = np.kron(sig_p, np.kron(Z_string, np.kron(sig_Z, sig_m)))/2 # 小到大系数为i
        UxAA_x = expm(-1j * (HxAA_x+np.conj(HxAA_x).T) * self.t)
        UxAA_x_dagger = expm(1j * (HxAA_x+np.conj(HxAA_x).T) * self.t)

        # onsite
        for i in range(0, self.total_mode, 2):
            self.qc.unitary(Operator(U0), [i], label='U0')
        for i in range(1, self.total_mode, 2):
            self.qc.unitary(Operator(U0_dagger), [i], label='U0_d')
        # x direction.
        for i in range(0, self.total_mode, 2):
            # A->A in x direction
            if i+2*self.num_y<self.total_mode:
                self.qc.unitary(Operator(UxAA_x), list(range(i, i+2*self.num_y+1)), label='Ux')
                self.qc.unitary(Operator(UyAB_x), list(range(i, i+2*self.num_y+2)), label='Uy') # 也可能是dagger
        for i in range(1, self.total_mode, 2):
            # B->B in x direction
            if i+2*self.num_y<self.total_mode:
                self.qc.unitary(Operator(UxAA_x_dagger), list(range(i, i+2*self.num_y+1)), label='Ux_d')
                self.qc.unitary(Operator(UyBA_x), list(range(i, i+2*self.num_y)), label='Uy')
        # y direction
        for i in range(0, self.total_mode, 2):
            if i not in list(range(2*self.num_y-2,self.total_mode,2*self.num_y)):
                #print(i,self.num_x, self.num_y, self.total_mode)
                self.qc.unitary(Operator(UxAA_y), [i, i+1, i+2], label='Ux')
                self.qc.unitary(Operator(UxAB_y_dagger), [i, i+1, i+2, i+3], label='Ux_d') # 也可能是dagger
        for i in range(1, self.total_mode, 2):
            if i not in list(range(2*self.num_y-1,self.total_mode,2*self.num_y)):
                self.qc.unitary(Operator(UxAA_y_dagger), [i, i+1, i+2], label='Ux_d')
                self.qc.unitary(Operator(Ux), [i, i+1], label='Ux') # 也可能是dagger # Ux是对的，刚好没有Jordan变换的负号。

    def nonHermitian_Circuit(self):
        # measurements 
        alpha = np.exp(-self.t*self.gamma)
        beta = np.sqrt(1-alpha**2)
        UM = np.identity(2**2, dtype=np.complex128)
        UM[1:3, 1:3] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128)
        # 增益耗散取决于辅助比特制备在什么量子态。
        
        dissi_site = sorted([i for i in range(0, self.total_mode, 2*self.num_y)] + [i for i in range(1, self.total_mode, 2*self.num_y)])
        # 预编译耗散操作电路模板
        for site in dissi_site:
            dissi_qc = QuantumCircuit(self.total_bit)
            #dissi_qc.x(total_bit-1)
            dissi_qc.unitary(Operator(UM), [site, self.total_bit-1])
            #dissi_qc.save_state()
            # dissi_circuits.append(transpile(dissi_qc, simulator))
            self.dissi_circuits.append(dissi_qc)

    def add_interaction_intra(self):
        # 加入相互作用
        #U = 0.4  # 粒子数守恒的相互作用。
        print(self.U)
        H_inter1 = np.zeros((4,4), dtype=np.complex128)
        H_inter1[3,3] = self.U
        U_inter = expm(-1j * H_inter1 * self.t)

        for i in range(0, self.total_mode, 2):
            self.qc.unitary(Operator(U_inter), [i, i+1], label='UI')
    
    def add_interaction_inter(self):
        print(self.U)
        H_inter1 = np.zeros((4,4), dtype=np.complex128)
        H_inter1[3,3] = self.U
        U_inter = expm(-1j * H_inter1 * self.t)
        # interaction 2
        for i in range(0, self.total_mode, 2):
            if i not in list(range(2*self.num_y-2,self.total_mode,2*self.num_y)):
                self.qc.unitary(Operator(U_inter), [i, i+2], label='UI')
                self.qc.unitary(Operator(U_inter), [i, i+3], label='UI')
                self.qc.unitary(Operator(U_inter), [i+1, i+2], label='UI')
                self.qc.unitary(Operator(U_inter), [i+1, i+3], label='UI')
        for i in range(0, self.total_mode, 2):
            # A->A in x direction
            distance = 2*self.num_y
            if i+distance<self.total_mode:
                self.qc.unitary(Operator(U_inter), [i, i+distance], label='UI')
                self.qc.unitary(Operator(U_inter), [i, i+distance+1], label='UI')
                self.qc.unitary(Operator(U_inter), [i+1, i+distance], label='UI')
                self.qc.unitary(Operator(U_inter), [i+1, i+distance+1], label='UI')

    def evolve(self, t_list, init_state_record=[0,1]): # 默认在(1,1)的元胞激发两个粒子。
        temp_register = len(t_list[:-1])*2*self.num_x
        QWZ2 = ParticleConservedSystem(self.num_x, self.num_y, self.total_particles)
        QWZ2.add_onsite_term(self.u)
        QWZ2.add_xhopping_term()
        QWZ2.add_yhopping_term()
        # QWZ2.add_interaction_inter(self.U)
        QWZ2.add_NH(self.gamma)
        ham2 = QWZ2.hamiltonian
        self.basis = QWZ2.basis
        state_index = QWZ2.state_index
        init_state_record = [0,1]  # initial state. Excited which site
        print(self.qc)
        project_M = Operator([[1, 0], [0, 0]])

        if self.qc == QuantumCircuit(self.total_mode+1):
            self.add_HermtianCircuit()
            self.add_interaction_inter()
            self.nonHermitian_Circuit()
        simulator = AerSimulator(
            method='matrix_product_state',
            device='GPU' if 'GPU' in AerSimulator().available_devices() else 'CPU',
            max_parallel_threads=1,
        )

        temp_register = len(t_list[:-1])*2*self.num_x
        full_circuit = QuantumCircuit(self.total_mode+1, temp_register+2*self.num_x*self.num_y)
        full_circuit.x(init_state_record + [2*self.num_x*self.num_y])
        full_circuit.save_matrix_product_state(label='my_mps')
        tcirc = transpile(full_circuit, simulator)
        result = simulator.run(tcirc).result()
        init_MPS = result.data(0)['my_mps'][0]
        print(init_MPS, len(init_MPS))

        for i in range(len(t_list[:-1])):
            full_circuit = QuantumCircuit(self.total_mode+1, temp_register+2*self.num_x*self.num_y)
            # full_circuit.initialize_matrix_product_state(init_MPS, list(range(0,self.total_mode,1)))
            
            full_circuit = full_circuit.compose(self.qc)
            print(i)
            for j, dissi_qc in enumerate(self.dissi_circuits):
                # # 合并耗散操作和测量
                # # print('j', j)
                # full_state = np.kron(np.array([0,1], dtype=np.complex128),evolved)  # 添加辅助比特
                # full_circuit = QuantumCircuit(self.total_bit)
                # # full_circuit.initialize(full_state, list(range(0,self.total_bit,1)))
                # full_circuit.set_statevector(full_state) # 这个更快！！是对全部初始化。
                # # full_circuit.compose(dissi_qc)
                full_circuit = full_circuit.compose(dissi_qc)
                full_circuit.save_matrix_product_state(label='my_mps')
                # full_circuit.save_state()
                #full_circuit.measure(self.total_bit-1, 2*self.num_x*i+j)
                tcirc = transpile(full_circuit, simulator)
                result = simulator.run(tcirc).result()
                data = result.data(0)['my_mps']
        # measurement the final data
        for i in range(2*self.num_x*self.num_y):
            full_circuit.measure(i, temp_register+i)

        # from qiskit.visualization import plot_bloch_multivector, plot_histogram
        # print(full_circuit)
        # compiled_qc1 = transpile(full_circuit, simulator)
        # result = simulator.run(
        #     compiled_qc1,
        #     shots=1,
        #     memory=True
        # ).result().data()['statevector'] #.result().get_statevector()

        # print(result)
        simulator_measurement = Aer.get_backend('qasm_simulator')
        qc_measure = transpile(full_circuit, simulator_measurement)
        shots = 10**6
        result_measurement = simulator_measurement.run(qc_measure, shots=shots).result()
        counts = result_measurement.get_counts()
        # print("测量结果：", counts)
        # plot_histogram(counts)

        # pick up the data.
        dict_result = dict()
        for key, value in counts.items():
            if key[-temp_register:] == '1'*temp_register:
                dict_result[key[:2*self.num_x*self.num_y]] = value  # from qubit 8  to qubit 0 . 因为先测0，所以结果在右边
        print("测量结果：", dict_result)
        print('Total', sum(dict_result.values()))

        total_event = sum(dict_result.values())
        print('The real trajectory is:', total_event)
        if total_event!=0:
            n_popu = np.zeros(self.total_mode)
            for key, value in dict_result.items():
                for i in range(1, self.total_mode+1):
                    n_popu[i-1] += int(key[-i])*value
            n_popu = n_popu / total_event


        # 这是为了和经典的进行比较
        init_state = np.zeros(len(self.basis), dtype=np.complex128)
        #init_state[(1<<4)+1] = 1.0  # 对应x([0,4])的激发态
        # init_state[(1<<2)+1] = 1.0  # 对应x([0,4])的激发态
        init_label = 0
        for idx in init_state_record[::-1]:
            init_label += 1<<idx
        print(init_label)
        init_state[state_index[init_label]] = 1.0
        initial_pick = init_state#init_state[self.basis].copy()

        final_pick = expm(-1j*ham2*t_list[-1]) @ initial_pick
        print(len(final_pick), np.linalg.norm(final_pick))
        final_pick /= np.linalg.norm(final_pick)
        a = self.Calcu_Occup(final_pick)

        print('The results of the circuit:', n_popu)
        print('The results of exact calculation:', a)



        # prob_list = np.zeros(len(t_list))
        # prob_list[0] = 1.

        # ham2 = QWZ2.hamiltonian
        # self.basis = QWZ2.basis

        # print('The system size:', ham2.shape, self.total_particles)

        # simulator = AerSimulator(
        #     method='statevector',
        #     device='GPU' if 'GPU' in AerSimulator().available_devices() else 'CPU',
        #     max_parallel_threads=1,
        # )
        # compiled_qc1 = transpile(self.qc, simulator)

        # # device='GPU' if AerSimulator().available_devices(True)['name'] == 'GPU' else 'CPU', # optimization_level=2,
        # init_state = np.zeros(2**self.total_bit, dtype=np.complex128)
        # #init_state[(1<<4)+1] = 1.0  # 对应x([0,4])的激发态
        # # init_state[(1<<2)+1] = 1.0  # 对应x([0,4])的激发态
        # init_label = 0
        # for idx in init_state_record:
        #     init_label += 1<<idx
        # init_state[init_label] = 1.0
        # initial_pick = init_state[self.basis].copy()
        # print('Initial Norm:', np.linalg.norm(initial_pick))

        # record_comp = np.zeros((2, len(self.basis), len(t_list)), dtype=np.complex128)
        # record_comp[0,:,0] = initial_pick
        # record_comp[1,:,0] = initial_pick  # digital

        # # record_comp = np.zeros((2, len(basis), len(t_list)), dtype=np.complex128)
        # # record_comp[0,:,0] = initial_pick
        # # record_comp[1,:,0] = initial_pick  # digital
        # # state = state[-2**(self.total_mode):]

        # current_state = init_state.copy()

        # #print(compiled_qc.draw())
        # for i in range(len(t_list)-1):
        #     print(i)
        #     # --------------------------
        #     # 主演化过程
        #     # --------------------------
        #     # 直接使用模拟器进行状态演化
        #     #print(current_state[:2**self.total_mode])
        #     full_circuit = QuantumCircuit(self.total_mode)
        #     # full_circuit.initialize(current_state[:2**self.total_mode], list(range(0,self.total_mode,1)))
        #     full_circuit.set_statevector(current_state[:2**self.total_mode])
        #     full_circuit = full_circuit.compose(self.qc)
        #     # full_circuit = full_circuit.compose(compiled_qc1)
        #     full_circuit.save_state()
        #     compiled_qc = transpile(full_circuit, simulator)
        #     #print(compiled_qc.draw())
        #     evolved = simulator.run(
        #         compiled_qc,
        #         shots=1,
        #         initial_statevector=current_state[:2**self.total_mode],
        #         memory=True
        #     ).result().data()['statevector']   # get_statevector() # data['statevector'] 
        #     # memory=True
        #     # --------------------------
        #     # 耗散过程优化
        #     # --------------------------
        #     # 如果列表为空不会执行。
        #     temp_prob = 1.
        #     for j, dissi_qc in enumerate(self.dissi_circuits):
        #         # 合并耗散操作和测量
        #         # print('j', j)
        #         full_state = np.kron(np.array([0,1], dtype=np.complex128),evolved)  # 添加辅助比特
        #         full_circuit = QuantumCircuit(self.total_bit)
        #         # full_circuit.initialize(full_state, list(range(0,self.total_bit,1)))
        #         full_circuit.set_statevector(full_state) # 这个更快！！是对全部初始化。
        #         # full_circuit.compose(dissi_qc)
        #         full_circuit = full_circuit.compose(dissi_qc)
        #         full_circuit.save_state()
        #         compiled_qc1 = transpile(full_circuit, simulator)
        #         result = simulator.run(
        #             compiled_qc1,
        #             shots=1,
        #             memory=True
        #         ).result().data()['statevector'] #.result().get_statevector()
        #         # initial_statevector=full_state,
                
        #         # 直接操作状态向量（避免多次初始化）
        #         result = np.array(result)
        #         evolved = result[-2**self.total_mode:]
        #         norm = np.linalg.norm(evolved)
        #         #print('temp norm 2', np.linalg.norm(evolved))
        #         evolved /= norm
        #         temp_prob *= norm**2
            
        #     prob_list[i+1] = prob_list[i] * temp_prob
            
        #     result = np.array(evolved)
        #     evolved = result[-2**self.total_mode:]
        #     norm = np.linalg.norm(evolved)
        #     #print('temp norm 2', np.linalg.norm(evolved))
        #     evolved /= norm
            
        #     current_state = evolved
        #     print('temp norm 2', np.linalg.norm(current_state))
        #     # --------------------------
        #     # 记录结果
        #     # --------------------------
        #     # print(self.basis)
        #     record_comp[0,:,i+1] = evolved[self.basis]

        #     record_comp[1,:,i+1] = expm(-1j*ham2*t_list[i+1]) @ initial_pick
        #     record_comp[1,:,i+1] /= np.linalg.norm(record_comp[1,:,i+1])
        
        # popu_list = np.zeros((2, self.total_mode, len(t_list)))
        # for i in range(len(t_list)):
        #     popu_list[0,:,i] = self.Calcu_Occup(record_comp[0,:,i])
        #     popu_list[1,:,i] = self.Calcu_Occup(record_comp[1,:,i])
        # record_comp = np.abs(record_comp)**2
        # return record_comp, popu_list, prob_list
    
    def Calcu_Occup(self, eigen_state):
        # calculate the n for a given state
        #assert len(eigen_state) == self.dimension
        population = np.zeros(self.total_mode)
        for idx, state in enumerate(self.basis):
            #print(eigen_state[idx]**2, np.array([ReadBit(state, i) for i in range(self.total_mode)]))
            population += np.abs(eigen_state[idx])**2 * np.array([ReadBit(state, i) for i in range(self.total_mode)])
        return population
    

if __name__=="__main__":
    start_time = time.time()
    #Circuit3(np.linspace(0,5,41))  # 0,1,31区别较小，否则还是有不同。

    # t_list = np.linspace(0,5,51)
    # circuit_qwz = CircuitSystem(4, 2, 3, t_list, U=0.4)  # y大反而运行更快
    # circuit_qwz.add_HermtianCircuit()
    # circuit_qwz.add_interaction_inter()
    # circuit_qwz.nonHermitian_Circuit()
    # results, result_popu, prob_list = circuit_qwz.evolve(t_list, [0,1,2])
    # fig, ax = plt.subplots(1,2,sharex=True, sharey=True)
    # # for i in range(len(circuit_qwz.basis)//2):
    # #     ax[0].plot(t_list, results[0, i], 'o-')
    # #     ax[1].plot(t_list, results[1, i], 'o-')
    # for i in range(len(result_popu[0])):
    #     ax[0].plot(t_list, result_popu[0, i], 'o-')
    #     ax[1].plot(t_list, result_popu[1, i], 'o-')
    # ax[0].set_title('Circuit')
    # ax[1].set_title('Exact')
    # plt.show()

    # plt.plot(t_list, prob_list, '^--')
    # plt.yscale('log')
    # plt.xlabel('t')
    # plt.ylabel('post-selection probability')
    # plt.show()

    t_list = np.linspace(0,3,21)
    circuit_qwz = CircuitSystem2(2, 4, 2, t_list, U=0.4)  # y大反而运行更快
    circuit_qwz.add_HermtianCircuit()
    circuit_qwz.add_interaction_inter()
    circuit_qwz.nonHermitian_Circuit()
    circuit_qwz.evolve(t_list)


    print((time.time()-start_time)/60)