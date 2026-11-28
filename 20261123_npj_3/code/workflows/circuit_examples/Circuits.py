from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from scipy.linalg import expm
import numpy as np
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt
from qiskit.visualization import circuit_drawer
from qiskit.quantum_info import Operator, Statevector
num_x = 2
num_y = 4

# def simulate_H_evolution(t):
#     # 构建哈密顿量矩阵 (8x8)
#     H = np.zeros((8, 8))
#     H[3, 5] = 1  # |011><101|
#     H[5, 3] = 1  # Hermitian conjugate
    
#     # 计算时间演化算符
#     U = expm(-1j * H * t)
    
#     # 创建量子电路并应用演化算符
#     qc = QuantumCircuit(3)
#     qc.unitary(Operator(U), [0, 1, 2], label=f'exp(-iHt), t={t}')
    
#     # 初始化到 |011> 态（假设 q0 是最低位）
#     init_qc = QuantumCircuit(3)
#     init_qc.x([0, 1])  # q0=1, q1=1, q2=0 -> |011⟩
    
#     # 组合初始化和演化
#     full_circuit = init_qc.compose(qc)
#     full_circuit.measure_all()
    
#     print(full_circuit)

#     # 运行模拟
#     simulator = AerSimulator()
#     compiled = transpile(full_circuit, simulator)
#     result = simulator.run(compiled, shots=1024).result()
#     counts = result.get_counts()
#     return counts

def get_wavefunction(t_list):
    # 构建哈密顿量矩阵 (8x8)
    u = -1.5 
    t = t_list[1]
    H0 = np.zeros((4, 4), dtype=np.complex128)
    H0[3, 3] = u
    U0 = expm(-1j * H0 * t)
    U0_dagger = expm(1j * H0 * t)

    # hopping in x dirrection.
    H1_x = np.zeros((8, 8), dtype=np.complex128)
    H1_x[5, 6] = 1/2 # |101><110|（注意量子位顺序）
    H1_x[6, 5] = 1/2 # # 厄米共轭项
    U1_x = expm(-1j * H1_x * t)
    U1_x_dagger = expm(1j * H1_x * t)

    H1_y = np.zeros((8, 8), dtype=np.complex128)
    H1_y[5, 6] = 1j/2 # |101><110|（注意量子位顺序）.x坐标小的在前面[x,x+1]
    H1_y[6, 5] = -1j/2 # # 厄米共轭项
    U1_y = expm(-1j * H1_y * t)
    U1_y_dagger = expm(1j * H1_y * t)

    H2 = np.zeros((16, 16), dtype=np.complex128)
    H2[5, 10], H2[9, 6] = -1/2, 1/2 # -|0101><1010|+|1001><0110|（注意量子位顺序）.x坐标小的在前面[xA,xB,y,y+1]
    H2[10, 5], H2[6, 9] = -1/2, 1/2 # # 厄米共轭项
    # hopping in x dirrection.
    U2 = expm(-1j * H2 * t)
    U2_dagger = expm(1j * H2 * t)

    # H = np.zeros((8, 8), dtype=np.complex128)
    # H[3, 5] = 1  # |011><101|（注意量子位顺序）
    # H[5, 3] = 1  # 厄米共轭项
    
    # 计算时间演化算符 U = e^(-iHt)
    # U = expm(-1j * H * t)
    
    # 1. 创建量子电路并应用演化，作用U0。核对了没有问题
    qc = QuantumCircuit(8)
    for i in range(2):
        for j in range(4, 8):
            #print(i,j)
            qc.unitary(Operator(U0), [i*2, j], label='U0')
    for i in range(2):
        for j in range(4, 8):
            #print(i,j)
            qc.unitary(Operator(U0_dagger), [i*2+1, j], label='U0-')
    
    # 2. x hoppping
    for i in range(4,8):
        qc.unitary(Operator(U1_x), [2, 0, i], label='U1x')
    for i in range(4,8):
        qc.unitary(Operator(U1_x_dagger), [3, 1, i], label='U1x-')
    for i in range(4,8):
        qc.unitary(Operator(U1_y_dagger), [3, 0, i], label='U1y-')
    for i in range(4,8):
        qc.unitary(Operator(U1_y_dagger), [2, 1, i], label='U1y-')

    # qc.unitary(Operator(U1_x), [2,0,4], label='U1x')
    # print(U1_x) # 应该反过来，从前往后写作用的比特。

    # 3. y hopping digonal
    for i in range(4,7):
        qc.unitary(Operator(U1_x), [i+1, i, 0], label='U1x')
    for i in range(4,7):
        qc.unitary(Operator(U1_x_dagger), [i+1, i, 1], label='U1x-')
    for i in range(4,7):
        qc.unitary(Operator(U1_x), [i+1, i, 2], label='U1x')
    for i in range(4,7):
        qc.unitary(Operator(U1_x_dagger), [i+1, i, 3], label='U1x-')

    # 4. y hopping
    for i in range(4,7):
        qc.unitary(Operator(U2), [i+1, i, 1, 0], label='U2') # qc.unitary(Operator(U2), [0, 1, i, i+1], label='U2')
        qc.unitary(Operator(U2), [i+1, i, 3, 2], label='U2')

    # 初始化到 |011⟩（q0=1, q1=1, q2=0）
    init_qc = QuantumCircuit(8)
    init_qc.x([0, 4])
    
    ### record
    total_evolve = np.zeros((16, len(t_list)), dtype='complex')
    state = Statevector(init_qc)
    total_evolve[:,0] = np.abs(Pick_Eigen(state))**2
    # 合并初始化电路和演化电路
    full_circuit = init_qc.compose(qc)
    state = Statevector(full_circuit)
    total_evolve[:,1] = np.abs(Pick_Eigen(state))**2
    for i in range(len(t_list)-2):
        full_circuit = full_circuit.compose(qc)
        state = Statevector(full_circuit)
        total_evolve[:,i+2] = np.abs(Pick_Eigen(state))**2

    #print(full_circuit)
    # 直接通过 Statevector 计算波函数
    #state = Statevector(full_circuit)
    return total_evolve

def get_wavefunction_NonHermi(t_list):
    # 构建哈密顿量矩阵 (8x8)
    u = -1.5 
    t = t_list[1]
    H0 = np.zeros((4, 4), dtype=np.complex128)
    H0[3, 3] = u
    U0 = expm(-1j * H0 * t)
    U0_dagger = expm(1j * H0 * t)

    # hopping in x dirrection.
    H1_x = np.zeros((8, 8), dtype=np.complex128)
    H1_x[5, 6] = 1/2 # |101><110|（注意量子位顺序）
    H1_x[6, 5] = 1/2 # # 厄米共轭项
    U1_x = expm(-1j * H1_x * t)
    U1_x_dagger = expm(1j * H1_x * t)

    H1_y = np.zeros((8, 8), dtype=np.complex128)
    H1_y[5, 6] = 1j/2 # |101><110|（注意量子位顺序）.x坐标小的在前面[x,x+1]
    H1_y[6, 5] = -1j/2 # # 厄米共轭项
    U1_y = expm(-1j * H1_y * t)
    U1_y_dagger = expm(1j * H1_y * t)

    H2 = np.zeros((16, 16), dtype=np.complex128)
    H2[5, 10], H2[9, 6] = -1/2, 1/2 # -|0101><1010|+|1001><0110|（注意量子位顺序）.x坐标小的在前面[xA,xB,y,y+1]。反过来就是 -|1010><0101|+|1001><0110|
    H2[10, 5], H2[6, 9] = -1/2, 1/2 # # 厄米共轭项
    # hopping in x dirrection.
    U2 = expm(-1j * H2 * t)
    U2_dagger = expm(1j * H2 * t)

    # Defined the measurement.
    def UM_Matrix(mea_t):
        # 由于厄米和非厄米不对易，所以只能在3s以下才可以。
        gamma = 0.2
        alpha = np.exp(-mea_t*gamma)
        beta = np.sqrt(1-alpha**2)
        UM = np.identity(2**4, dtype=np.complex128)
        UM[0:2,0:2] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0000
        UM[2:4,2:4] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0010
        UM[4:6,4:6] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0100
        UM[8:10,8:10] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #1000
        # UM[12:14,12:14] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #1100
        # 耗散
        # UM[6:8,6:8] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0110
        # UM[10:12,10:12] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #1010
        return UM
    def UM_Matrix_Six(mea_t):
        # 由于厄米和非厄米不对易，所以只能在3s以下才可以。
        gamma = 0.2
        alpha = np.exp(-mea_t*gamma)
        beta = np.sqrt(1-alpha**2)
        UM = np.identity(2**6, dtype=np.complex128)
        UM[0:2,0:2] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0000
        UM[2:4,2:4] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0010
        UM[4:6,4:6] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0100
        UM[8:10,8:10] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #1000
        UM[16:18,16:18] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #1000
        UM[32:34,32:34] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #1000
        # UM[12:14,12:14] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #1100
        # 耗散
        # UM[6:8,6:8] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0110
        # UM[10:12,10:12] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #1010
        return UM
    #print(UM, np.dot(UM,np.conj(UM.T)))

    # H = np.zeros((8, 8), dtype=np.complex128)
    # H[3, 5] = 1  # |011><101|（注意量子位顺序）
    # H[5, 3] = 1  # 厄米共轭项
    
    # 计算时间演化算符 U = e^(-iHt)
    # U = expm(-1j * H * t)
    
    # 1. 创建量子电路并应用演化，作用U0。核对了没有问题
    qc = QuantumCircuit(10)
    for i in range(2):
        for j in range(4, 8):
            #print(i,j)
            qc.unitary(Operator(U0), [i*2, j], label='U0')
    for i in range(2):
        for j in range(4, 8):
            #print(i,j)
            qc.unitary(Operator(U0_dagger), [i*2+1, j], label='U0-')
    
    # 2. x hoppping
    for i in range(4,8):
        qc.unitary(Operator(U1_x), [2, 0, i], label='U1x')
    for i in range(4,8):
        qc.unitary(Operator(U1_x_dagger), [3, 1, i], label='U1x-')
    for i in range(4,8):
        qc.unitary(Operator(U1_y_dagger), [0, 3, i], label='U1y-')
    for i in range(4,8):
        qc.unitary(Operator(U1_y_dagger), [1, 2, i], label='U1y-')
    # for i in range(4,8):
    #     qc.unitary(Operator(U1_y_dagger), [3, 0, i], label='U1y-')
    # for i in range(4,8):
    #     qc.unitary(Operator(U1_y_dagger), [2, 1, i], label='U1y-')

    # qc.unitary(Operator(U1_x), [2,0,4], label='U1x')
    # print(U1_x) # 应该反过来，从前往后写作用的比特。

    # 3. y hopping digonal
    for i in range(4,7):
        qc.unitary(Operator(U1_x), [i+1, i, 0], label='U1x')
    for i in range(4,7):
        qc.unitary(Operator(U1_x_dagger), [i+1, i, 1], label='U1x-')
    for i in range(4,7):
        qc.unitary(Operator(U1_x), [i+1, i, 2], label='U1x')
    for i in range(4,7):
        qc.unitary(Operator(U1_x_dagger), [i+1, i, 3], label='U1x-')

    # 4. y hopping
    for i in range(4,7):
        #qc.unitary(Operator(U2), [i+1, i, 1, 0], label='U2') # qc.unitary(Operator(U2), [0, 1, i, i+1], label='U2')
        #qc.unitary(Operator(U2), [i+1, i, 3, 2], label='U2')
        qc.unitary(Operator(U2), [0,1,i,i+1], label='U2')
        qc.unitary(Operator(U2), [2,3,i,i+1], label='U2')

    # 初始化到 |011⟩（q0=1, q1=1, q2=0）
    init_qc = QuantumCircuit(10)
    # method 1
    #init_qc.x([0, 4])
    # method 2
    init_qc.h([0,2,4,6])
    init_qc.cx(0,1)
    init_qc.cx(2,3)
    init_qc.cx(4,5)
    init_qc.cx(6,7)
    init_qc.x([1,3,5,7])
    init_qc.cx(0,2)
    init_qc.cx(2,1)
    init_qc.cx(1,3)
    #init_qc.cx(3,0)
    init_qc.ccx(0,1,3)
    init_qc.cx(3,0)
    init_qc.cx(3,1)
    init_qc.cx(3,2)

    init_qc.cx(4,6)
    init_qc.cx(6,5)
    init_qc.cx(5,7)
    #init_qc.cx(7,4)
    init_qc.ccx(4,5,7)
    init_qc.cx(7,4)
    init_qc.cx(7,5)
    init_qc.cx(7,6)


    # method 2。 从10001000+10000100出发，并没有反过来。
    # init_qc.x(0)
    # init_qc.h(4)
    # init_qc.cx(4,5)
    # init_qc.x(4)
    # wavefunc = Statevector(init_qc)
    # print(wavefunc, wavefunc[17],wavefunc[33])

    # measurement
    # qc_m = QuantumCircuit(10)
    # qc_m.unitary(Operator(UM), [8,4,1,0], label='U2')
    # qc_m.unitary(Operator(UM), [9,4,3,2], label='U2')
    
    ### record
    total_evolve = np.zeros((16, len(t_list)), dtype='complex')
    state = Statevector(init_qc)
    total_evolve[:,0] = np.abs(Pick_Eigen(state))**2
    # 合并初始化电路和演化电路
    full_circuit = init_qc
    
    for i in range(len(t_list)-1):
        full_circuit = full_circuit.compose(qc)
        # state = Statevector(full_circuit)
        # proj_state = state[:2**8]
        # total_evolve[:,i+2] = np.abs(Pick_Eigen(state))**2
        qc_m = QuantumCircuit(10)
        qc_m.unitary(Operator(UM_Matrix(t_list[i+1])), [8,4,1,0], label='UM')
        qc_m.unitary(Operator(UM_Matrix(t_list[i+1])), [9,4,3,2], label='UM')
        # qc_m.unitary(Operator(UM_Matrix_Six(t_list[i+1])), [8,4,3,2,1,0], label='UM')

        measurement_circuit = full_circuit.compose(qc_m)
        state = Statevector(measurement_circuit)
        #state = Statevector(full_circuit)
        state = np.array(state)
        proj_state = state[:2**8]
        print('probability', np.sum(np.abs(proj_state)**2))
        #print(proj_state,len(proj_state), Pick_Eigen(proj_state),np.sum(np.abs(proj_state)**2))
        total_evolve[:,i+1] = Pick_Eigen(proj_state)
        #proj_state = proj_state/np.sum(np.abs(proj_state)**2)
    # state = Statevector(full_circuit)
    # state = np.array(state)
    # proj_state = state[:2**8]
    # print('probability', np.sum(np.abs(proj_state)**2))
    # proj_state = proj_state/np.sum(np.abs(proj_state)**2)


    #print(full_circuit)
    # 直接通过 Statevector 计算波函数
    #state = Statevector(full_circuit)
    return total_evolve


def get_wavefunction_NonHermi2(t_list):
    # 构建哈密顿量矩阵 (8x8)
    u = -1.5 
    t = t_list[1]
    H0 = np.zeros((4, 4), dtype=np.complex128)
    H0[3, 3] = u
    U0 = expm(-1j * H0 * t)
    U0_dagger = expm(1j * H0 * t)

    # hopping in x dirrection.
    H1_x = np.zeros((8, 8), dtype=np.complex128)
    H1_x[5, 6] = 1/2 # |101><110|（注意量子位顺序）
    H1_x[6, 5] = 1/2 # # 厄米共轭项
    U1_x = expm(-1j * H1_x * t)
    U1_x_dagger = expm(1j * H1_x * t)

    H1_y = np.zeros((8, 8), dtype=np.complex128)
    H1_y[5, 6] = 1j/2 # |101><110|（注意量子位顺序）.x坐标小的在前面[x,x+1]
    H1_y[6, 5] = -1j/2 # # 厄米共轭项
    U1_y = expm(-1j * H1_y * t)
    U1_y_dagger = expm(1j * H1_y * t)

    H2 = np.zeros((16, 16), dtype=np.complex128)
    H2[5, 10], H2[9, 6] = -1/2, 1/2 # -|0101><1010|+|1001><0110|（注意量子位顺序）.x坐标小的在前面[xA,xB,y,y+1]。反过来就是 -|1010><0101|+|1001><0110|
    H2[10, 5], H2[6, 9] = -1/2, 1/2 # # 厄米共轭项
    # hopping in x dirrection.
    U2 = expm(-1j * H2 * t)
    U2_dagger = expm(1j * H2 * t)

    # Defined the measurement.
    def UM_Matrix(mea_t):
        # 由于厄米和非厄米不对易，所以只能在3s以下才可以。
        gamma = 0.2
        alpha = np.exp(-mea_t*gamma)
        beta = np.sqrt(1-alpha**2)
        UM = np.identity(2**4, dtype=np.complex128)
        UM[0:2,0:2] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0000
        UM[2:4,2:4] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0010
        UM[4:6,4:6] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0100
        UM[8:10,8:10] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #1000
        # UM[12:14,12:14] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #1100
        # 耗散
        # UM[6:8,6:8] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0110
        # UM[10:12,10:12] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #1010
        return UM
    def UM_Matrix_Six(mea_t):
        # 由于厄米和非厄米不对易，所以只能在3s以下才可以。
        gamma = 0.2
        alpha = np.exp(-mea_t*gamma)
        beta = np.sqrt(1-alpha**2)
        UM = np.identity(2**6, dtype=np.complex128)
        UM[0:2,0:2] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0000 00
        UM[2:4,2:4] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0000 10
        UM[4:6,4:6] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0001 00
        UM[8:10,8:10] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0010 00
        UM[16:18,16:18] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0100 00
        UM[32:34,32:34] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #1000 00
        # UM[12:14,12:14] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #1100
        # 耗散
        # UM[6:8,6:8] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #000110
        # UM[10:12,10:12] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #001010
        # UM[18:20,18:20] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #010010
        # UM[34:36,34:36] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #100010
        return UM
    #print(UM, np.dot(UM,np.conj(UM.T)))

    # H = np.zeros((8, 8), dtype=np.complex128)
    # H[3, 5] = 1  # |011><101|（注意量子位顺序）
    # H[5, 3] = 1  # 厄米共轭项
    
    # 计算时间演化算符 U = e^(-iHt)
    # U = expm(-1j * H * t)
    
    # 1. 创建量子电路并应用演化，作用U0。核对了没有问题
    qc = QuantumCircuit(10)
    for i in range(2):
        for j in range(4, 8):
            #print(i,j)
            qc.unitary(Operator(U0), [i*2, j], label='U0')
    for i in range(2):
        for j in range(4, 8):
            #print(i,j)
            qc.unitary(Operator(U0_dagger), [i*2+1, j], label='U0-')
    
    # 2. x hoppping
    for i in range(4,8):
        qc.unitary(Operator(U1_x), [2, 0, i], label='U1x')
    for i in range(4,8):
        qc.unitary(Operator(U1_x_dagger), [3, 1, i], label='U1x-')
    for i in range(4,8):
        qc.unitary(Operator(U1_y_dagger), [0, 3, i], label='U1y-')
    for i in range(4,8):
        qc.unitary(Operator(U1_y_dagger), [1, 2, i], label='U1y-')
    # for i in range(4,8):
    #     qc.unitary(Operator(U1_y_dagger), [3, 0, i], label='U1y-')
    # for i in range(4,8):
    #     qc.unitary(Operator(U1_y_dagger), [2, 1, i], label='U1y-')

    # qc.unitary(Operator(U1_x), [2,0,4], label='U1x')
    # print(U1_x) # 应该反过来，从前往后写作用的比特。

    # 3. y hopping digonal
    for i in range(4,7):
        qc.unitary(Operator(U1_x), [i+1, i, 0], label='U1x')
    for i in range(4,7):
        qc.unitary(Operator(U1_x_dagger), [i+1, i, 1], label='U1x-')
    for i in range(4,7):
        qc.unitary(Operator(U1_x), [i+1, i, 2], label='U1x')
    for i in range(4,7):
        qc.unitary(Operator(U1_x_dagger), [i+1, i, 3], label='U1x-')

    # 4. y hopping
    for i in range(4,7):
        #qc.unitary(Operator(U2), [i+1, i, 1, 0], label='U2') # qc.unitary(Operator(U2), [0, 1, i, i+1], label='U2')
        #qc.unitary(Operator(U2), [i+1, i, 3, 2], label='U2')
        qc.unitary(Operator(U2), [0,1,i,i+1], label='U2')
        qc.unitary(Operator(U2), [2,3,i,i+1], label='U2')

    # 初始化到 |011⟩（q0=1, q1=1, q2=0）
    init_qc = QuantumCircuit(10)
    # method 1
    init_qc.x([0, 4])
    # method 2
    # init_qc.h([0,2,4,6])
    # init_qc.cx(0,1)
    # init_qc.cx(2,3)
    # init_qc.cx(4,5)
    # init_qc.cx(6,7)
    # init_qc.x([1,3,5,7])
    # init_qc.cx(0,2)
    # init_qc.cx(2,1)
    # init_qc.cx(1,3)
    # #init_qc.cx(3,0)
    # init_qc.ccx(0,1,3)
    # init_qc.cx(3,0)
    # init_qc.cx(3,1)
    # init_qc.cx(3,2)

    # init_qc.cx(4,6)
    # init_qc.cx(6,5)
    # init_qc.cx(5,7)
    # #init_qc.cx(7,4)
    # init_qc.ccx(4,5,7)
    # init_qc.cx(7,4)
    # init_qc.cx(7,5)
    # init_qc.cx(7,6)


    
    # method 2。 从10001000+10000100出发，并没有反过来。
    # init_qc.x(0)
    # init_qc.h(4)
    # init_qc.cx(4,5)
    # init_qc.x(4)
    # wavefunc = Statevector(init_qc)
    # print(wavefunc, wavefunc[17],wavefunc[33])

    # measurement
    qc_m = QuantumCircuit(10)
    # qc_m.unitary(Operator(UM), [8,4,1,0], label='U2')
    # qc_m.unitary(Operator(UM), [9,4,3,2], label='U2')
    qc_m.unitary(Operator(UM_Matrix_Six(t_list[1])), [8,4,3,2,1,0], label='UM')
    
    ### record
    total_evolve = np.zeros((16, len(t_list)), dtype='complex')
    proba_list = np.zeros(len(t_list), dtype='complex')
    proba_list[0] = 1.
    state = Statevector(init_qc)
    total_evolve[:,0] = np.abs(Pick_Eigen(state))**2
    # 合并初始化电路和演化电路
    #full_circuit = init_qc
    state = np.array(state)
    state = state[:2**8]
    
    for i in range(len(t_list)-1):
        #full_circuit = full_circuit.compose(qc)
        # state = Statevector(full_circuit)
        # proj_state = state[:2**8]
        # total_evolve[:,i+2] = np.abs(Pick_Eigen(state))**2
        #qc_m.unitary(Operator(UM_Matrix_Six(t)), [8,4,3,2,1,0], label='UM')

        # #measurement_circuit = full_circuit.compose(qc_m)
        # qc_m = QuantumCircuit(10)
        # # qc_m.unitary(Operator(UM_Matrix(t_list[1])), [8,4,1,0], label='UM')
        # # qc_m.unitary(Operator(UM_Matrix(t_list[1])), [9,4,3,2], label='UM')
        # qc_m.unitary(Operator(UM_Matrix_Six(t_list[1])), [8,4,3,2,1,0], label='UM')
        # # measurement_circuit = full_circuit.compose(qc_m)
        # full_circuit = full_circuit.compose(qc_m)
        # #print(full_circuit)
        # full_circuit.reset([8,9])
        # #state = Statevector(measurement_circuit)
        # #measurement_circuit = full_circuit
        # state = Statevector.from_instruction(full_circuit)
        # state = np.array(state)
        # proj_state = state[:2**8]
        # print('probability', np.sum(np.abs(proj_state)**2))
        # #print(proj_state,len(proj_state), Pick_Eigen(proj_state),np.sum(np.abs(proj_state)**2))
        # total_evolve[:,i+1] = Pick_Eigen(proj_state)
        # print('probability', np.sum(np.abs(total_evolve[:,i+1])**2))

        state = Statevector(state)
        full_circuit = QuantumCircuit(10)
        full_circuit.initialize(state, [0, 1,2,3,4,5,6,7])  # 设定初态
        full_circuit = full_circuit.compose(qc)
        full_circuit = full_circuit.compose(qc_m)
        initial_state = Statevector(full_circuit)
        state = np.array(initial_state)
        state = state[:2**8]
        total_evolve[:,i+1] = Pick_Eigen(state)
        proba_list[i+1] = proba_list[i] * np.sum(np.abs(state)**2)
        state = state / np.sqrt(np.sum(np.abs(state)**2))
        print('probability', np.sum(np.abs(total_evolve[:,i+1])**2))


        #proj_state = proj_state/np.sum(np.abs(proj_state)**2)
    # state = Statevector(full_circuit)
    # state = np.array(state)
    # proj_state = state[:2**8]
    # print('probability', np.sum(np.abs(proj_state)**2))
    # proj_state = proj_state/np.sum(np.abs(proj_state)**2)


    #print(full_circuit)
    # 直接通过 Statevector 计算波函数
    #state = Statevector(full_circuit)
    return total_evolve, proba_list


def get_wavefunction_NonHermi2_Onemeasuremment(t_list):
    # 构建哈密顿量矩阵 (8x8)
    u = -1.5 
    t = t_list[1]
    H0 = np.zeros((4, 4), dtype=np.complex128)
    H0[3, 3] = u
    U0 = expm(-1j * H0 * t)
    U0_dagger = expm(1j * H0 * t)

    # hopping in x dirrection.
    H1_x = np.zeros((8, 8), dtype=np.complex128)
    H1_x[5, 6] = 1/2 # |101><110|（注意量子位顺序）
    H1_x[6, 5] = 1/2 # # 厄米共轭项
    U1_x = expm(-1j * H1_x * t)
    U1_x_dagger = expm(1j * H1_x * t)

    H1_y = np.zeros((8, 8), dtype=np.complex128)
    H1_y[5, 6] = 1j/2 # |101><110|（注意量子位顺序）.x坐标小的在前面[x,x+1]
    H1_y[6, 5] = -1j/2 # # 厄米共轭项
    U1_y = expm(-1j * H1_y * t)
    U1_y_dagger = expm(1j * H1_y * t)

    H2 = np.zeros((16, 16), dtype=np.complex128)
    H2[5, 10], H2[9, 6] = -1/2, 1/2 # -|0101><1010|+|1001><0110|（注意量子位顺序）.x坐标小的在前面[xA,xB,y,y+1]。反过来就是 -|1010><0101|+|1001><0110|
    H2[10, 5], H2[6, 9] = -1/2, 1/2 # # 厄米共轭项
    # hopping in x dirrection.
    U2 = expm(-1j * H2 * t)
    U2_dagger = expm(1j * H2 * t)

    # Defined the measurement.
    def UM_Matrix(mea_t):
        # 由于厄米和非厄米不对易，所以只能在3s以下才可以。
        gamma = 0.2
        alpha = np.exp(-mea_t*gamma)
        beta = np.sqrt(1-alpha**2)
        UM = np.identity(2**4, dtype=np.complex128)
        UM[0:2,0:2] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0000
        UM[2:4,2:4] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0010
        UM[4:6,4:6] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0100
        UM[8:10,8:10] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #1000
        # UM[12:14,12:14] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #1100
        # 耗散
        # UM[6:8,6:8] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0110
        # UM[10:12,10:12] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #1010
        return UM
    def UM_Matrix_Six(mea_t):
        # 由于厄米和非厄米不对易，所以只能在3s以下才可以。
        gamma = 0.2
        alpha = np.exp(-mea_t*gamma)
        beta = np.sqrt(1-alpha**2)
        UM = np.identity(2**6, dtype=np.complex128)
        UM[0:2,0:2] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0000 00
        UM[2:4,2:4] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0000 10
        UM[4:6,4:6] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0001 00
        UM[8:10,8:10] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0010 00
        UM[16:18,16:18] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #0100 00
        UM[32:34,32:34] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #1000 00
        # UM[12:14,12:14] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #1100
        # 耗散
        # UM[6:8,6:8] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #000110
        # UM[10:12,10:12] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #001010
        # UM[18:20,18:20] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #010010
        # UM[34:36,34:36] = np.array([[alpha, beta],[-beta,alpha]], dtype=np.complex128) #100010
        return UM
    #print(UM, np.dot(UM,np.conj(UM.T)))

    # H = np.zeros((8, 8), dtype=np.complex128)
    # H[3, 5] = 1  # |011><101|（注意量子位顺序）
    # H[5, 3] = 1  # 厄米共轭项
    
    # 计算时间演化算符 U = e^(-iHt)
    # U = expm(-1j * H * t)
    
    # 1. 创建量子电路并应用演化，作用U0。核对了没有问题
    qc = QuantumCircuit(10)
    for i in range(2):
        for j in range(4, 8):
            #print(i,j)
            qc.unitary(Operator(U0), [i*2, j], label='U0')
    for i in range(2):
        for j in range(4, 8):
            #print(i,j)
            qc.unitary(Operator(U0_dagger), [i*2+1, j], label='U0-')
    
    # 2. x hoppping
    for i in range(4,8):
        qc.unitary(Operator(U1_x), [2, 0, i], label='U1x')
    for i in range(4,8):
        qc.unitary(Operator(U1_x_dagger), [3, 1, i], label='U1x-')
    for i in range(4,8):
        qc.unitary(Operator(U1_y_dagger), [0, 3, i], label='U1y-')
    for i in range(4,8):
        qc.unitary(Operator(U1_y_dagger), [1, 2, i], label='U1y-')
    # for i in range(4,8):
    #     qc.unitary(Operator(U1_y_dagger), [3, 0, i], label='U1y-')
    # for i in range(4,8):
    #     qc.unitary(Operator(U1_y_dagger), [2, 1, i], label='U1y-')

    # qc.unitary(Operator(U1_x), [2,0,4], label='U1x')
    # print(U1_x) # 应该反过来，从前往后写作用的比特。

    # 3. y hopping digonal
    for i in range(4,7):
        qc.unitary(Operator(U1_x), [i+1, i, 0], label='U1x')
    for i in range(4,7):
        qc.unitary(Operator(U1_x_dagger), [i+1, i, 1], label='U1x-')
    for i in range(4,7):
        qc.unitary(Operator(U1_x), [i+1, i, 2], label='U1x')
    for i in range(4,7):
        qc.unitary(Operator(U1_x_dagger), [i+1, i, 3], label='U1x-')

    # 4. y hopping
    for i in range(4,7):
        #qc.unitary(Operator(U2), [i+1, i, 1, 0], label='U2') # qc.unitary(Operator(U2), [0, 1, i, i+1], label='U2')
        #qc.unitary(Operator(U2), [i+1, i, 3, 2], label='U2')
        qc.unitary(Operator(U2), [0,1,i,i+1], label='U2')
        qc.unitary(Operator(U2), [2,3,i,i+1], label='U2')

    # 初始化到 |011⟩（q0=1, q1=1, q2=0）
    init_qc = QuantumCircuit(10)
    # method 1
    init_qc.x([0, 4])
    # method 2
    # init_qc.h([0,2,4,6])
    # init_qc.cx(0,1)
    # init_qc.cx(2,3)
    # init_qc.cx(4,5)
    # init_qc.cx(6,7)
    # init_qc.x([1,3,5,7])
    # init_qc.cx(0,2)
    # init_qc.cx(2,1)
    # init_qc.cx(1,3)
    # #init_qc.cx(3,0)
    # init_qc.ccx(0,1,3)
    # init_qc.cx(3,0)
    # init_qc.cx(3,1)
    # init_qc.cx(3,2)

    # init_qc.cx(4,6)
    # init_qc.cx(6,5)
    # init_qc.cx(5,7)
    # #init_qc.cx(7,4)
    # init_qc.ccx(4,5,7)
    # init_qc.cx(7,4)
    # init_qc.cx(7,5)
    # init_qc.cx(7,6)


    
    # method 2。 从10001000+10000100出发，并没有反过来。
    # init_qc.x(0)
    # init_qc.h(4)
    # init_qc.cx(4,5)
    # init_qc.x(4)
    # wavefunc = Statevector(init_qc)
    # print(wavefunc, wavefunc[17],wavefunc[33])

    # measurement
    qc_m = QuantumCircuit(10)
    # qc_m.unitary(Operator(UM), [8,4,1,0], label='U2')
    # qc_m.unitary(Operator(UM), [9,4,3,2], label='U2')
    qc_m.unitary(Operator(UM_Matrix_Six(t_list[1])), [8,4,3,2,1,0], label='UM')
    
    ### record
    total_evolve = np.zeros((16, len(t_list)), dtype='complex')
    proba_list = np.zeros(len(t_list), dtype='complex')
    proba_list[0] = 1.
    state = Statevector(init_qc)
    total_evolve[:,0] = np.abs(Pick_Eigen(state))**2
    # 合并初始化电路和演化电路
    full_circuit = init_qc
    
    for i in range(len(t_list)-1):
        full_circuit = full_circuit.compose(qc)
        # state = Statevector(full_circuit)
        # proj_state = state[:2**8]
        # total_evolve[:,i+2] = np.abs(Pick_Eigen(state))**2
        #qc_m.unitary(Operator(UM_Matrix_Six(t)), [8,4,3,2,1,0], label='UM')

        # #measurement_circuit = full_circuit.compose(qc_m)
        qc_m = QuantumCircuit(10)
        # qc_m.unitary(Operator(UM_Matrix(t_list[1])), [8,4,1,0], label='UM')
        # qc_m.unitary(Operator(UM_Matrix(t_list[1])), [9,4,3,2], label='UM')
        qc_m.unitary(Operator(UM_Matrix_Six(t_list[i+1])), [8,4,3,2,1,0], label='UM')
        measurement_circuit = full_circuit.compose(qc_m)

        # 生成 Matplotlib 图像
        # figure = measurement_circuit.draw(
        #     output='mpl',
        #     style={
        #         'name': 'bw',         # 黑白风格
        #         'subfontsize': 9,     # 字体大小
        #         'fold': 20            # 折叠阈值（超过20层门则折叠）
        #     },
        #     idle_wires=False          # 隐藏空闲的量子比特线
        # )

        # # 调整图像布局
        # plt.tight_layout()
        # plt.show()

        # full_circuit = full_circuit.compose(qc_m)
        # #print(full_circuit)
        # full_circuit.reset([8,9])
        state = Statevector(measurement_circuit)
        # #measurement_circuit = full_circuit
        # state = Statevector.from_instruction(full_circuit)
        state = np.array(state)
        proj_state = state[:2**8]
        print('probability', np.sum(np.abs(proj_state)**2))
        # #print(proj_state,len(proj_state), Pick_Eigen(proj_state),np.sum(np.abs(proj_state)**2))
        total_evolve[:,i+1] = Pick_Eigen(proj_state)
        # print('probability', np.sum(np.abs(total_evolve[:,i+1])**2))

        # # 这是将测量和演化都完全分开的结果。
        # state = Statevector(state)
        # full_circuit = QuantumCircuit(10)
        # full_circuit.initialize(state, [0, 1,2,3,4,5,6,7])  # 设定初态
        # full_circuit = full_circuit.compose(qc)
        # full_circuit = full_circuit.compose(qc_m)
        # initial_state = Statevector(full_circuit)
        # state = np.array(initial_state)
        # state = state[:2**8]
        # total_evolve[:,i+1] = Pick_Eigen(state)
        # proba_list[i+1] = proba_list[i] * np.sum(np.abs(state)**2)
        # state = state / np.sqrt(np.sum(np.abs(state)**2))
        # print('probability', np.sum(np.abs(total_evolve[:,i+1])**2))


        #proj_state = proj_state/np.sum(np.abs(proj_state)**2)
    # state = Statevector(full_circuit)
    # state = np.array(state)
    # proj_state = state[:2**8]
    # print('probability', np.sum(np.abs(proj_state)**2))
    # proj_state = proj_state/np.sum(np.abs(proj_state)**2)


    #print(full_circuit)
    # 直接通过 Statevector 计算波函数
    #state = Statevector(full_circuit)
    return total_evolve, proba_list



def Pick_Eigen(large_eigen):
    num_x = 2
    num_y = 4
    eigen = np.zeros(2*num_x*num_y, dtype=np.complex128)
    for i in range(2*num_x):
        for j in range(num_y):
            eigen[j*4+i] = large_eigen[(1<<(i))+(1<<(4+j))]
    return eigen


# # 示例：取 t=π 并打印结果
# print(simulate_H_evolution(np.pi/4))




# # # 获取 t=π 时的波函数
# total_t_list = np.linspace(0,14,100)
# wavefunction = get_wavefunction(total_t_list)
# popu_evolve = wavefunction[::2,:]+wavefunction[1::2,:]
# print('The sum of the wavefunction:', np.sum(popu_evolve, axis=0))
# import matplotlib.pyplot as plt
# for i in range(len(popu_evolve)):
#     plt.plot(total_t_list, popu_evolve[i])
# plt.plot(total_t_list, popu_evolve[0],'^-',color='k',label='(%d,%d)'%(1,1))
# plt.plot(total_t_list, popu_evolve[2],'o-',color='lime',label='(%d,%d)'%(2,1))
# plt.show()
# # 输出波函数信息
# print("Final Statevector:", popu_evolve[:,-1])




# Non-Hermitian
total_t_list = np.linspace(0,14,201)
wavefunction, prob_list = get_wavefunction_NonHermi2(total_t_list)
print(prob_list)
proba_list = np.zeros(len(total_t_list))
for i in range(len(total_t_list)):
    proba_list[i] = np.sum(np.abs(wavefunction[:,i])**2)
plt.plot(total_t_list, prob_list,'^-')
plt.ylabel('probability')
plt.yscale('log')
plt.show()

# state
norm_wave = np.zeros_like(wavefunction)
for i in range(len(total_t_list)):
    norm_wave[:,i] = wavefunction[:,i]/np.sqrt(proba_list[i])
norm_wave = np.abs(norm_wave)**2
norm_wave = norm_wave[::2]+norm_wave[1::2]
for i in range(len(norm_wave)):
    plt.plot(total_t_list,norm_wave[i],label='(%d,%d)'%(i//2+1,i%2+1))
plt.plot(total_t_list, norm_wave[0],'^-',color='k',label='(%d,%d)'%(1,1))
plt.plot(total_t_list, norm_wave[2],'o-',color='lime',label='(%d,%d)'%(2,1))
# reference line. 从10001000+10000100出发，并没有反过来。
# reference = np.ones((2, len(total_t_list)))/np.sqrt(2)
# for i in range(len(total_t_list)):
#     reference[0,i] = 0.5*np.exp(-0.2*total_t_list[i]*2)/(1/2+0.5*np.exp(-0.2*total_t_list[i]*2))
#     reference[1,i] = 0.5/(1/2+0.5*np.exp(-0.2*total_t_list[i]*2))
# plt.plot(total_t_list, reference[0],'--')
# plt.plot(total_t_list, reference[1],'--')
#plt.vlines(14, ymin=0, ymax=1,linestyles='--')
plt.ylim(ymin=0,ymax=1)
plt.vlines(3.7, ymin=0, ymax=1,linestyles='--')
plt.legend()
plt.show()




# wavefunction = Pick_Eigen(wavefunction)

# popu_evolve = np.abs(wavefunction[::2])**2+np.abs(wavefunction[1::2])**2
#print(popu_evolve, np.sum(popu_evolve))

#print(len(wavefunction.data), wavefunction.data)  # 复数数组表示各基态振幅
#pick_eigen = Pick_Eigen(wavefunction.data)
#print(pick_eigen, np.abs(pick_eigen)[::2]**2+np.abs(pick_eigen)[1::2]**2)
# t_list = np.linspace(0,14,8)
# popo = np.zeros(len(t_list))
# for i in range(len(popo)):
#     wavefunction = get_wavefunction(t_list)
#     popo[i] = np.abs(wavefunction.data[17])**2
# import matplotlib.pyplot as plt
# plt.plot(t_list, popo,'o-')
# plt.show()


