"4个比特的单粒子的拓扑可以直接训练出来。"
"""
ope_results.h5   是单粒子的。..2.h5是全部参数都优化的

ope_results_particle.h5   是双粒子粒子的。..2_particle2.h5是全部参数都优化的
 _NoH都是针对双粒子的，是指不添加H gate。3是5*104迭代。

"""
import numpy as np
import time
from scipy.linalg import expm
import scipy as sy
import matplotlib.pyplot as plt
from matplotlib import colors
from numba import jit, cuda
import matplotlib as mpl

sigma_0 = np.array([[1., 0.], [0., 1.]], dtype='complex')
sigma_x = np.array([[0., 1.], [1., 0.]], dtype='complex')
sigma_y = np.array([[0., -1j], [1j, 0.]], dtype='complex')
sigma_z = np.array([[1., 0.], [0.,-1.]], dtype='complex')
np.random.seed(1)

def Calculate_IPR(state):
    # smaller than 1, skin state have O(1/Ly) but corner state have O(1) 'IPR'
    temp_state = np.abs(state) ** 2
    popu_state = temp_state[::2] + temp_state[1::2]
    denominator = np.sum(popu_state) ** 2
    nominator = np.sum(np.abs(popu_state) ** 2)
    return nominator / denominator


def RealHam_QWZ_Dissapation(u, A, gamma, num_xsite, num_ysite, boundary='open', B=1):
    """
    Give out the Hamiltonian in real space of QiWuZhang model.
    :param u: parameter u
    :param A:
    :param num_xsite: how many sites in x direction
    :param num_ysite: how many sites in y direction
    :return:
    """
    ham = np.zeros((2 * num_xsite * num_ysite, 2 * num_xsite * num_ysite), dtype='complex')
    for delta_x in range(num_xsite):
        for delta_y in range(num_ysite):
            row = 2 * (delta_x * num_ysite + delta_y)
            ham[row:row + 2, row:row + 2] = [[u, 0], [0, -u]]
    for delta_x in range(num_xsite - 1):
        for delta_y in range(num_ysite):
            # A site to A site in x direction.
            delta_col = 2 * (delta_x * num_ysite + delta_y)
            delta_row = 2 * ((delta_x + 1) * num_ysite + delta_y)
            ham[delta_row:delta_row + 2, delta_col:delta_col + 2] = [[1 / 2, A * 1j / 2], [A * 1j / 2, -1 / 2]]
            ham[delta_col:delta_col + 2, delta_row:delta_row + 2] = [[1 / 2, -A * 1j / 2], [-A * 1j / 2, -1 / 2]]
    for delta_x in range(num_xsite):
        for delta_y in range(num_ysite - 1):
            # A site to A site
            delta_col = 2 * (delta_x * num_ysite + delta_y)
            delta_row = 2 * ((delta_x) * num_ysite + delta_y + 1)
            ham[delta_row:delta_row + 2, delta_col:delta_col + 2] = [[1 / 2, B / 2], [-B / 2, -1 / 2]]
            ham[delta_col:delta_col + 2, delta_row:delta_row + 2] = [[1 / 2, -B / 2], [B / 2, -1 / 2]]
    if boundary=='close_x':
        # x方向封闭，但是y方向open
        site_constant = 2*num_ysite*(num_xsite-1)
        for i in range(num_ysite):
            ham[2 * i:2 * i + 2, site_constant + 2 * i:site_constant + 2 * i + 2] = [[1 / 2, A * 1j / 2],[A * 1j / 2, -1 / 2]]
            ham[site_constant + 2 * i:site_constant + 2 * i + 2, 2 * i:2 * i + 2] = [[1 / 2, -A * 1j / 2],
                                                                                     [-A * 1j / 2, -1 / 2]]
    if boundary=='close':
        # close x
        site_constant = 2 * num_ysite * (num_xsite - 1)
        for i in range(num_ysite):
            ham[2 * i:2 * i + 2, site_constant + 2 * i:site_constant + 2 * i + 2] = [[1 / 2, A * 1j / 2],[A * 1j / 2, -1 / 2]]
            ham[site_constant + 2 * i:site_constant + 2 * i + 2, 2 * i:2 * i + 2] = [[1 / 2, -A * 1j / 2],[-A * 1j / 2, -1 / 2]]
        # close y
        site_constant = 2 * (num_ysite - 1)
        for i in range(num_xsite):
            ham[2 * i * num_ysite:2 * i * num_ysite + 2,
            2 * i * num_ysite + site_constant:2 * i * num_ysite + 2 + site_constant] = [[1 / 2, 1 / 2], [-1 / 2, -1 / 2]]
            ham[2 * i * num_ysite + site_constant:2 * i * num_ysite + 2 + site_constant,
            2 * i * num_ysite:2 * i * num_ysite + 2] = [[1 / 2, -1 / 2], [1 / 2, -1 / 2]]
    # 在x=0的边界出加decay.  
    # 2 5
    # 1 4
    # 0 3
    # edge dissipation
    # for delta_x in range(num_xsite):
    #     row = 2 * (delta_x * num_ysite + 0)
    #     ham[row:row + 2, row:row + 2] += [[-1j*gamma, 0.], [0., -1j*gamma]]
    # dissipation at the edge y=0,上面写错了
    for delta_y in range(num_xsite):
        row = 2 * (delta_y * num_ysite + 0)
        ham[row:row + 2, row:row + 2] += [[-1j*gamma, 0.], [0., -1j*gamma]]
    return ham

def GeneratePOVM(gamma, T, num_xsite, num_ysite, position):
    # position used to be the x edge(y=0). Now we choose which position is dissipative.
    # 其实直接展开的方案与直接通过非厄米计算得到的也不是完全相等。
    # the hopping from edge to ancilla is different from 'gamma'.
    povm_list = np.zeros((len(position), 2 * num_xsite * num_ysite, 2 * num_xsite * num_ysite), dtype='complex')
    para = np.sqrt(np.exp(2*gamma*T)-1)  # approxition, np.sqrt(2*gamma * T), np.sqrt(1-(1-T*gamma)**2).
    print(gamma)
    if position==None:
        position = [0] * num_xsite
        for i in range(num_xsite):
            position[i] = i*num_ysite
    for site in range(len(position)):
        temp_povm = np.identity(2 * num_xsite * num_ysite, dtype='complex')
        row = 2*position[site]
        #temp_povm[row:row + 2, row:row + 2] += [[-T*gamma, 0.], [0., -T*gamma]]
        #temp_povm[row, row] += -T*gamma
        #temp_povm[row+1, row+1] += -T*gamma
        # loss non-Hermitian Hamiltonian
        temp_povm[row, row] += -1+np.exp(-gamma*T)  # The one we take is better.
        temp_povm[row+1, row + 1] += -1+np.exp(-gamma*T)  #np.sqrt(1-(1-T*gamma)**2)
        povm_list[site] = temp_povm
        '''
        # gain non-Hermitian Hamiltonian
        #temp_povm[row, row] += -1 + np.exp(gamma * T)  # The one we take is better.
        #temp_povm[row + 1, row + 1] += -1 + np.exp(gamma * T)  # np.sqrt(1-(1-T*gamma)**2)
        #povm_list[site] = temp_povm'''

    return povm_list


def GenerateHeff(p, N, num_xsite, num_ysite, u, gamma=0.2, T=0.001, position=None, boundary='open', ret_type='ham'):
    # 如果在这里的采用@jit, 那么最后得到的ham不会有区别，也不知道为啥。
    evo_ham = RealHam_QWZ_Dissapation(u, 1, 0, num_xsite, num_ysite, boundary)
    evo = expm(-1j*evo_ham*T)
    Mj_plus = GeneratePOVM(gamma, T, num_xsite, num_ysite, position)
    fps = np.identity((2*num_xsite*num_ysite), dtype='complex')
    for i in range(N):
        random_num = np.random.random(num_xsite)
        total_measure = np.identity(2*num_xsite*num_ysite, dtype='complex')
        #  这一个不太方便采用半满时候同样的分析，因为site太大了，也就是说需要储存很多。还是这种方式更方便。
        for j in range(num_xsite):
            if random_num[j] <= p:
                total_measure = np.dot(Mj_plus[j], total_measure)
        #for j in range(len(total_measure[0])):
        #    total_measure[:, j] = total_measure[:, j] / np.sqrt(np.sum(np.abs(total_measure[:, j]) ** 2))
        Fps = np.dot(total_measure, evo)
        # 这里对每一个列向量进行归一化就对应遍历了不同的初态。
        # 但是这里存在一个问题，不能对态进行归一化，归一化就不能得到正确的结果，就抹去了测量的效果。所以不在对应测量？
        fps = np.dot(Fps, fps)
    heff = 1j/(N*T) * sy.linalg.logm(fps)
    if ret_type == 'evo':
        return fps
    else:
        return heff

def VQE22():
    u = -1.5 # u 太大之后使topological trivial的。
    gamma = 0.2
    num_xsite = 2
    num_ysite = 4
    ham22 = RealHam_QWZ_Dissapation(u, 1, gamma, num_xsite, num_ysite, boundary='open')

    eigval, eigvec = np.linalg.eig(ham22)
    eigvec = eigvec[:, np.argsort(np.abs(eigval.real))]
    eigval = eigval[np.argsort(np.abs(eigval.real))]

    print(eigval) # 
    return eigvec[:,0]

def VQE23():
    # 非厄米的结果
    u = -1.5 # u 太大之后使topological trivial的。
    gamma = 0.2
    num_xsite = 2
    num_ysite = 4
    total_particle = 2
    from MeasureNonhermitianSkin.Experimental.ManyBodyQWZ import ParticleConservedSystem
    QWZ1 = ParticleConservedSystem(num_xsite, num_ysite, total_particle)
    QWZ1.add_onsite_term(u)
    QWZ1.add_xhopping_term()
    QWZ1.add_yhopping_term()
    # QWZ1.add_interaction_inter(U)
    QWZ1.add_NH(gamma)
    ham2 = QWZ1.hamiltonian
    eigval2, eigvec2 = np.linalg.eig(ham2)

    ham1 = RealHam_QWZ_Dissapation(u, 1, gamma, num_xsite, num_ysite, boundary='open')
    eigval, eigvec = np.linalg.eig(ham1)
    eigvec = eigvec[:, np.argsort(np.abs(eigval.real))]
    eigval = eigval[np.argsort(np.abs(eigval.real))]

    eigvec2 = eigvec2[:, np.argsort(np.abs(eigval2-eigval[0]-eigval[1]))]
    eigval2 = eigval2[np.argsort(np.abs(eigval2-eigval[0]-eigval[1]))]
    return eigvec2[:,0], ham2, eigval2

import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.algorithms.optimizers import COBYLA
from qiskit.quantum_info import state_fidelity
from qiskit.algorithms import VQE
from qiskit.opflow import StateFn, CircuitStateFn

# 目标态 |ψ⟩ (示例为GHZ态)
# target_state = (np.array([1] + [0]*14 + [1])/np.sqrt(2)).reshape(-1,1)
target_state = np.zeros(128, dtype='complex')
target_state[:120], ham_particle2, eigval_particle2 = VQE23()
M_E = np.zeros_like(ham_particle2, dtype='complex')
M_E = ham_particle2 - eigval_particle2 * np.identity(len(M_E))
M_E = np.dot(np.conj(M_E.T), M_E)
target_state = target_state.reshape(-1,1)
print(target_state.shape)
def create_layer(num_qubits, layer_params):
    """
    创建单层量子线路
    layer_params: [theta1, theta2, theta3] 对应三个两比特门的参数

    theta1, phi1, lambda1, theta2, phi2, lambda2,..
    """
    qc = QuantumCircuit(num_qubits)
    params_iter = iter(layer_params)
    
    # 在[1,2], [3,4]和[2,3]上施加两比特门（示例使用CRY门）
    pairs = [(0,1), (2,3),(4,5)]  # 注意Qiskit的qubit编号从0开始
    for qi in range(num_qubits-1):
        theta, phi, lam = layer_params[3*qi], layer_params[3*qi+1], layer_params[3*qi+2]
        qc.u3(theta, phi, lam, qi)
    param_begin = 3*(num_qubits-1)
    for (q1, q2) in pairs:
        qc.cz(q1, q2)
    for qi in range(num_qubits-1):
        theta, phi, lam = layer_params[3*qi+param_begin], layer_params[3*qi+1+param_begin], layer_params[3*qi+2+param_begin]
        qc.u3(theta, phi, lam, qi)
    param_begin = (3*(num_qubits-1)*2)

    # second 
    pairs = [(1,2),(3,4),(5,6)]  # 注意Qiskit的qubit编号从0开始
    for qi in range(num_qubits-1):
        theta, phi, lam = layer_params[3*qi+param_begin], layer_params[3*qi+1+param_begin], layer_params[3*qi+2+param_begin]
        qc.u3(theta, phi, lam, qi+1)
    param_begin = (3*(num_qubits-1)*3)
    for (q1, q2) in pairs:
        qc.cz(q1, q2)
    for qi in range(num_qubits-1):
        theta, phi, lam = layer_params[3*qi+param_begin], layer_params[3*qi+1+param_begin], layer_params[3*qi+2+param_begin]
        qc.u3(theta, phi, lam, qi+1)
    return qc


def cost_function(params, num_layers, num_qubits, fixed_prev_params=None, record_loss=None, type_optimize=None):
    """
    计算当前参数下的状态保真度代价函数
    fixed_prev_params: 之前层的固定参数（用于逐层优化）
    """
    if type_optimize=='Optimize_All':
        new_params = []
        params_per_layer = 3*(4*num_qubits-4) # 36  # 根据create_layer的计算
        for i in range(int(len(params)/params_per_layer)-1):
            new_params.append(params[i*params_per_layer:(i+1)*params_per_layer])
        fixed_prev_params = new_params
        params = params[-params_per_layer:]
    qc = QuantumCircuit(num_qubits)
    
    for qi in range(num_qubits):
        qc.h(qi)
    if fixed_prev_params is not None:
        # 添加之前优化好的层
        for l in range(len(fixed_prev_params)):
            layer_params = fixed_prev_params[l]
            qc.compose(create_layer(num_qubits, layer_params), inplace=True)

    # 添加当前层（正在优化的层）
    current_layer_params = params
    qc.compose(create_layer(num_qubits, current_layer_params), inplace=True)
    
    # 模拟电路
    simulator = Aer.get_backend('statevector_simulator')
    result = execute(qc, simulator).result()
    final_state = result.get_statevector(qc)

    # 计算与目标态的距离（1-保真度）
    fid = state_fidelity(final_state, target_state)
    if record_loss is not None:
        record_loss[num_layers].append(1-fid)

    final_state = final_state.data
    final_state = final_state[:120]
    Expect_Ener = np.dot(np.conj(final_state.T), np.dot(M_E, final_state)) + 100 * np.dot(np.conj(final_state[-8:].T), final_state[-8:])
    # if record_loss is not None:
    #     record_loss[num_layers].append(Expect_Ener)
    
    # return 1 - fid
    return Expect_Ener

def optimize_circuit(max_layers=5, threshold=0.01):
    """
    分层优化直到收敛（逐层优化）
    :param max_layers: 最大层数
    :param threshold: 收敛阈值
    """
    # 存储损失的列表
    loss_history = []
    from collections import defaultdict
    all_loss = defaultdict(list)


    # def callback(eval_count, parameters, value, std):
    #     loss_history.append(value)
    #     print(f"Iteration {eval_count}: Loss = {value:.4f}")

    num_qubits = 7
    optimizer = COBYLA(maxiter=5000)
    best_params_per_layer = []
    params_per_layer = 3*(4*num_qubits-4) # 36  # 根据create_layer的计算
    
    for num_layers in range(1, max_layers+1):
        print(f"\nOptimizing layer {num_layers}...")
        
        # 初始化当前层参数
        # initial_params = np.random.uniform(0, 2*np.pi, size=params_per_layer)
        initial_params = np.abs(np.random.normal(loc=0, scale=10-9*(num_layers)/max_layers, size=params_per_layer))
        initial_params = np.mod(initial_params, 2*np.pi)  # 保证范围    
        
        # 优化当前层（固定之前层的参数）
        result = optimizer.minimize(
            fun=lambda p: cost_function(p, num_layers, num_qubits, fixed_prev_params=best_params_per_layer, record_loss=all_loss),
            x0=initial_params
        )
        
        # 保存当前层优化结果
        best_params_per_layer.append(result.x)
        # best_params_per_layer = result.x
        current_error = cost_function(result.x, num_layers, num_qubits, best_params_per_layer[:-1])
        print(f"Layer {num_layers} achieved error: {current_error:.4f}")
        
        loss_history.append(current_error)
        # 检查是否收敛
        if current_error < threshold:
            print(f"Converged at layer {num_layers} with error {current_error:.4f}")
            break
    return best_params_per_layer, num_layers, all_loss

def optimize_circuit2(max_layers=5, threshold=0.01):
    """
    层数逐渐增加，但是还是直接优化。
    :param max_layers: 最大层数
    :param threshold: 收敛阈值
    """
    # 存储损失的列表
    loss_history = []
    from collections import defaultdict
    all_loss = defaultdict(list)


    # def callback(eval_count, parameters, value, std):
    #     loss_history.append(value)
    #     print(f"Iteration {eval_count}: Loss = {value:.4f}")

    num_qubits = 7
    optimizer = COBYLA(maxiter=50000)
    best_params_per_layer = np.array([])
    params_per_layer = 3*(4*num_qubits-4) # 36  # 根据create_layer的计算
    
    for num_layers in range(1, max_layers+1):
        print(f"\nOptimizing layer {num_layers}...")
        
        # 初始化当前层参数
        # initial_params = np.random.uniform(0, 2*np.pi, size=params_per_layer)
        initial_params = np.abs(np.random.normal(loc=0, scale=10-9*(num_layers)/max_layers, size=params_per_layer))
        initial_params = np.mod(initial_params, 2*np.pi)  # 保证范围    
        
        # # 优化当前层（固定之前层的参数）
        # best_params_per_layer = np.append(best_params_per_layer, initial_params)
        # result = optimizer.minimize(
        #     fun=lambda p: cost_function(p, num_layers, num_qubits,  record_loss=all_loss, type_optimize='Optimize_All'),
        #     x0=best_params_per_layer
        # ) # fixed_prev_params=best_params_per_layer,
        # # 保存当前层优化结果
        # best_params_per_layer = result.x
        # current_error = result.fun # all_loss[num_layers][-1]

        # 采用粒子群算法。
        from pyswarm import pso
        lb = [0 ] * len(best_params_per_layer)
        ub = [2*np.pi ] * len(best_params_per_layer)
        best_params_per_layer, current_error = pso(cost_function, lb, ub, swarmsize=50, maxiter=5000, args=(num_layers, num_qubits, None, all_loss,'Optimize_All'))

        print(f"Layer {num_layers} achieved error: {current_error:.4f}")
        # print(all[num_layers][-1])
        if current_error < threshold:
            print(f"Converged at layer {num_layers} with error {current_error:.4f}")
            break
    new_params = []
    for i in range(int(len(best_params_per_layer)/params_per_layer)):
        new_params.append(best_params_per_layer[i*params_per_layer:(i+1)*params_per_layer])
    best_params_per_layer = new_params
    return best_params_per_layer, num_layers, all_loss


# 运行优化（逐层优化）
optimal_params_per_layer, final_num_layers, loss = optimize_circuit2(max_layers=7)

# print(optimal_params_per_layer[-1], len(optimal_params_per_layer))
# 构建最终电路
final_circuit = QuantumCircuit(7)
for qi in range(7):
    final_circuit.h(qi)
for layer_params in optimal_params_per_layer:
    final_circuit.compose(create_layer(7, layer_params), inplace=True)

# print("\nFinal circuit:")
# print(final_circuit.draw())

# 验证最终保真度
simulator = Aer.get_backend('statevector_simulator')
result = execute(final_circuit, simulator).result()
final_state = result.get_statevector(final_circuit)
fidelity = state_fidelity(final_state, target_state)
print(f"\nFinal state fidelity: {fidelity:.4f}")

import h5py
with h5py.File('optimization_results_particle_PSO_Loss2.h5', 'w') as f:
    for key, value in loss.items():
        f.create_dataset(str(key), data=value)

for i in range(1,15):
    plt.plot(loss[i])
plt.show()


if __name__=="__main__":
    print('final')