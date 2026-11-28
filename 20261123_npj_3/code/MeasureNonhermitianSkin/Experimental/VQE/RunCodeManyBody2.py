import numpy as np
import time
from scipy.linalg import expm
import scipy as sy
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib as mpl
import h5py
from collections import defaultdict

import numpy as np
import qiskit
print(qiskit.__version__, sy.__version__)

from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.circuit import Parameter
# from qiskit.algorithms.optimizers import COBYLA
# from qiskit_algorithms.optimizers import COBYLA
from scipy.optimize import minimize
from qiskit.quantum_info import state_fidelity
from itertools import combinations

sigma_0 = np.array([[1., 0.], [0., 1.]], dtype='complex')
sigma_x = np.array([[0., 1.], [1., 0.]], dtype='complex')
sigma_y = np.array([[0., -1j], [1j, 0.]], dtype='complex')
sigma_z = np.array([[1., 0.], [0.,-1.]], dtype='complex')
np.random.seed(1)

sigma_0 = np.array([[1., 0.], [0., 1.]], dtype='complex')
sigma_x = np.array([[0., 1.], [1., 0.]], dtype='complex')
sigma_y = np.array([[0., -1j], [1j, 0.]], dtype='complex')
sigma_z = np.array([[1., 0.], [0.,-1.]], dtype='complex')
np.random.seed(1)

def FlipBit(i,n):
    """
    Flip nth bit
    """
    return i^(1<<n)

def ReadBit(i, n):
    """
    read nth bit. 0 corresponds to the right most bit.
    """
    return (i&(1<<n))>>n

class ParticleConservedSystem:
    def __init__(self, num_xmodes: int, num_ymodes: int, total_particles: int):
        """
        初始化量子系统,
        一个晶格中有两个子晶格，排列的顺序是首先一个子晶格在一起，然后把一条y的全部排列完。然后排列下一个x的。

        从右向左的，也就是(x,y,1)=(1,1,1)的basis标号是1，0b000001!
        注意这里没有自旋向上和自旋向下。那种情况是把自旋向上和向下粘起来。
        :param num_modes: 系统的模数（如晶格格点数）
        :param total_particles: 总粒子数（守恒量）

        """
        self.num_x = num_xmodes
        self.num_y = num_ymodes
        self.total_particles = total_particles
        self.sublat = 2 # 一个元胞中包含A，B两个原子。
        self.total_mode = self.sublat*self.num_x*self.num_y
        
        # 生成所有可能的量子态基矢
        self.basis = self._generate_basis()
        self.dimension = len(self.basis) # C_N^{n}
        #print(type(self.basis), self.basis)
        
        # 建立态向量到矩阵索引的映射
        self.state_index = {state: idx for idx, state in enumerate(self.basis)}  # 0: 00001111, 1:00010111 ....
        # print(self.state_index)
        # 初始化哈密顿矩阵
        self.hamiltonian = np.zeros((self.dimension, self.dimension), dtype=np.complex128)

    # def _generate_basis(self):
    #     """生成所有满足总粒子数守恒的量子态"""
    #     count = np.array([PopCntBit(i) for i in range(2**self.total_mode)])
    #     # using the subspace to determine the spin configuration.
    #     basis = np.where(count == self.total_particles)[0]
    #     basis = list(basis)
    #     return sorted(basis) # int64
    
    def _generate_basis(self):
        # 当我们处理5*5的晶格时，但是却只要双激发，总的维度只有1125，但是指数却有2**50。
        combs = combinations(range(self.total_mode), self.total_particles)
        basis = [ sum([1<<bit for bit in bits]) for bits in combs]
        return sorted(basis)
    
    def _compute_fermion_sign(self, state, i, j):
        """计算从i到j跳跃时的费米子符号（i < j）"""
        if i > j:
            i, j = j, i
            sign = (-1) ** sum([ReadBit(state, k) for k in range(i+1, j)])
        else:
            sign = (-1) ** sum([ReadBit(state, k) for k in range(i+1, j)])
        return sign

    def add_onsite_term(self, u: float):
        """添加局域能量项: u * a†_m a_m"""
        #u = -1.5
        for idx, state in enumerate(self.basis): #第几个basis，矩阵的第几个。
            #print(idx, state, bin(state), ReadBit(state,0), ReadBit(state,1))
            self.hamiltonian[idx, idx] += u*np.sum([ReadBit(state, i) for i in range(0, self.total_mode, 2)]) - u*np.sum([ReadBit(state, i) for i in range(1, self.total_mode, 2)])
            # print(np.sum([ReadBit(state, i) for i in range(0, 2*self.num_x*self.num_y, 2)]))

    def add_NH(self, gamma: float, which_dissi:int=0):
        """在y=1的边界添加非厄米项: i * gamma * a†_m a_m
        which dissi:表示在y等于几加耗散。默认在y=1的边界
        """
        for idx, state in enumerate(self.basis): #第几个basis，矩阵的第几个。
            #print(idx, state, bin(state), ReadBit(state,0), ReadBit(state,1))
            self.hamiltonian[idx, idx] += 1j * gamma*np.sum([ReadBit(state, i) for i in range(2*which_dissi, self.total_mode, 2*self.num_y)]) + 1j * gamma * np.sum([ReadBit(state, i) for i in range(2*which_dissi+1, self.total_mode, 2*self.num_y)])
    
    def add_NH_SingleSite(self, gamma: float, which_dissi:list=[]):
        """在y=1的边界添加非厄米项: i * gamma * a†_m a_m
        which dissi:表示在y等于几加耗散。默认在y=1的边界
        """
        for idx, state in enumerate(self.basis): #第几个basis，矩阵的第几个。
            #print(idx, state, bin(state), ReadBit(state,0), ReadBit(state,1))
            self.hamiltonian[idx, idx] += 1j * gamma*np.sum([ReadBit(state, 2*i) for i in which_dissi]) + 1j * gamma * np.sum([ReadBit(state, 2*i+1) for i in which_dissi])

    def add_xhopping_term(self):
        """添加跃迁项: strength * a†_source a_target + h.c."""
        for idx, state in enumerate(self.basis):
            # from A particle
            for i in range(0, self.total_mode, 2): # A sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 2*self.num_y # mx A->mx+1，A
                    if ReadBit(state, i+distance1)==0 and i+distance1<self.total_mode:
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        self.hamiltonian[jdx1, idx] += 1.0/2 * sign
                        self.hamiltonian[idx, jdx1] += 1.0/2 * sign
                    distance2 = 2*self.num_y + 1 # mx A->mx+1，B
                    if ReadBit(state, i+distance2)==0 and i+distance2<self.total_mode:
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        self.hamiltonian[jdx2, idx] += 1j/2 * sign
                        self.hamiltonian[idx, jdx2] += -1j/2 * sign

            # from B particle
            for i in range(1, self.total_mode,2): # A sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 2*self.num_y # mx B->mx+1，B
                    if ReadBit(state, i+distance1)==0 and i+distance1<self.total_mode:
                        #print(distance1, i, bin(state))
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        self.hamiltonian[jdx1, idx] += -1.0/2 * sign
                        self.hamiltonian[idx, jdx1] += -1.0/2 * sign
                    distance2 = 2*self.num_y-1 # mx B->mx+1，A
                    if ReadBit(state, i+distance2)==0 and i+distance2<self.total_mode:
                        state2 = FlipBit(FlipBit(state, i), i+distance2) # mx A->mx+1，B
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        self.hamiltonian[jdx2, idx] += 1j/2 * sign
                        self.hamiltonian[idx, jdx2] += -1j/2 * sign
    def add_yhopping_term(self):
        """添加跃迁项: strength * a†_source a_target + h.c."""
        for idx, state in enumerate(self.basis):
            #from A particle
            for i in range(0, self.total_mode,2): # A sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 2 # my A->my+1，A
                    if ReadBit(state, i+distance1)==0 and i not in list(range(2*self.num_y-2, self.total_mode, 2*self.num_y)):
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        self.hamiltonian[jdx1, idx] += 1.0/2 * sign
                        self.hamiltonian[idx, jdx1] += 1.0/2 * sign
                    distance2 = 3 # my A->my+1，B
                    if ReadBit(state, i+distance2)==0 and i not in list(range(2*self.num_y-2, self.total_mode, 2*self.num_y)):
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        self.hamiltonian[jdx2, idx] += -1/2 * sign
                        self.hamiltonian[idx, jdx2] += -1/2 * sign

            # from B particle
            for i in range(1, self.total_mode,2): # A sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 2 # my B->my+1，B
                    if ReadBit(state, i+distance1)==0 and i not in list(range(2*self.num_y-1, self.total_mode, 2*self.num_y)):
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        self.hamiltonian[jdx1, idx] += -1.0/2 * sign
                        self.hamiltonian[idx, jdx1] += -1.0/2 * sign
                    distance2 = 1 # my B->my+1，A
                    if ReadBit(state, i+distance2)==0 and i not in list(range(2*self.num_y-1, self.total_mode, 2*self.num_y)):
                        state2 = FlipBit(FlipBit(state, i), i+distance2) # my B->my+1，A
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        self.hamiltonian[jdx2, idx] += 1/2 * sign #-1/2
                        self.hamiltonian[idx, jdx2] += 1/2 * sign #-1/2
                
    def add_interaction_intra(self, U):
        """添加相互作用U n_{x,y, A} n_{x,y, B}，当一个元胞内有两个粒子能量变大，趋向于不呆在或者呆在一个元胞内。或者是相互之间的，这样chiral skin会让它趋向于局域，但是相互作用可能让它更局域或者弥散。"""
        self.inter = np.zeros_like(self.hamiltonian)
        for idx, state in enumerate(self.basis):
            # # U n_{x,y, A} n_{x,y, B}
            for i in range(0, self.total_mode, 2): # A sublattice.
                if ReadBit(state, i)==1 and ReadBit(state, i+1)==1:
                    self.inter[idx, idx] += U
        self.hamiltonian += self.inter
        return self.inter

    def add_interaction_inter(self, U):
        """添加相互作用U( n_{x,y, A}+n_{x,y, B}) * (n_{x+1,y, A}+n_{x+1,y, B}))，当近邻元胞有不同的电子数时，会出现相互作用项。这样chiral skin会让它趋向于局域，但是相互作用可能让它更局域或者弥散。但是现象好像是相互作用会使它更局域"""
        self.inter = np.zeros_like(self.hamiltonian)
        for idx, state in enumerate(self.basis):
            # X: (U n_{x,y, A}+n_{x,y, B}) * (U n_{x+1,y, A}+n_{x+1,y, B}). Y同样。
            distance1 = 2*self.num_y
            for i in range(0, self.total_mode, 2): # A sublattice.
                # Y direction
                if i not in list(range(2*self.num_y-2, self.total_mode, 2*self.num_y)):
                    self.inter[idx, idx] += U*(ReadBit(state, i)+ReadBit(state,i+1))*(ReadBit(state, i+2)+ReadBit(state,i+3))
                # X direction
                if i+distance1<self.total_mode:
                    self.inter[idx, idx] += U*(ReadBit(state, i)+ReadBit(state,i+1))*(ReadBit(state, i+distance1)+ReadBit(state,i+distance1+1))
        self.hamiltonian += self.inter
        return self.inter


    def Calcu_Occup(self, eigen_state=None):
        # calculate the n for a given state
        if eigen_state is not None:
            assert len(eigen_state) == self.dimension
            population = np.zeros(self.total_mode)
            for idx, state in enumerate(self.basis):
                #print(eigen_state[idx]**2, np.array([ReadBit(state, i) for i in range(self.total_mode)]))
                population += np.abs(eigen_state[idx])**2 * np.array([ReadBit(state, i) for i in range(self.total_mode)])
            return population
        else:
            popu_indice = defaultdict(list)
            for idx, state in enumerate(self.basis):
                for i in range(self.total_mode):
                    if ReadBit(state, i)==1:
                        popu_indice[i].append(idx)
            # population = np.zeros(self.total_mode)
            # eigen_state = np.abs(eigen_state)**2
            # for i in range(len(population)):
            #     population[i] = np.sum(eigen_state[popu_indice[i]])
            return popu_indice


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
        ham[row:row + 2, row:row + 2] += [[1j*gamma, 0.], [0., 1j*gamma]]
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

def VQE23():
    # 非厄米的结果
    u = -1.5 # u 太大之后使topological trivial的。
    gamma = 0.2
    num_xsite = 2
    num_ysite = 4
    total_particle = 2
    U = 1.
    QWZ1 = ParticleConservedSystem(num_xsite, num_ysite, total_particle)
    QWZ1.add_onsite_term(u)
    QWZ1.add_xhopping_term()
    QWZ1.add_yhopping_term()
    QWZ1.add_interaction_inter(U)
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

def VQE23():
    # 非厄米的结果
    u = -1.5 # u 太大之后使topological trivial的。
    gamma = 0.2
    num_xsite = 2
    num_ysite = 4
    total_particle = 2
    U = 1.
    QWZ1 = ParticleConservedSystem(num_xsite, num_ysite, total_particle)
    QWZ1.add_onsite_term(u)
    QWZ1.add_xhopping_term()
    QWZ1.add_yhopping_term()
    QWZ1.add_interaction_inter(U)
    QWZ1.add_NH(gamma)
    ham2 = QWZ1.hamiltonian
    eigval2, eigvec2 = np.linalg.eig(ham2)

    eigvec2 = eigvec2[:, np.argsort(np.imag(eigval2))]
    eigval2 = eigval2[np.argsort(np.imag(eigval2))]
    return eigvec2[:,-1], ham2, eigval2
# 目标态 |ψ⟩ (示例为GHZ态)
# target_state = (np.array([1] + [0]*14 + [1])/np.sqrt(2)).reshape(-1,1)
target_state = np.zeros(128, dtype='complex')
target_state[:120], ham_particle2, eigval_particle2 = VQE23()
# M_E = np.zeros_like(ham_particle2, dtype='complex')
# M_E = ham_particle2 - eigval_particle2 * np.identity(len(M_E))
# M_E = np.dot(np.conj(M_E.T), M_E)
target_state = target_state.reshape(-1,1)
# print(target_state.shape)
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
    for (q1, q2) in pairs:
        qc.cz(q1, q2)
    for qi in range(num_qubits-1):
        theta, phi, lam = layer_params[3*qi], layer_params[3*qi+1], layer_params[3*qi+2]
        qc.u(theta, phi, lam, qi)
    param_begin = 3*(num_qubits-1)
    pairs = [(1,2),(3,4),(5,6)]  # 注意Qiskit的qubit编号从0开始
    for (q1, q2) in pairs:
        qc.cz(q1, q2)
    for qi in range(num_qubits-1):
        theta, phi, lam = layer_params[3*qi+param_begin], layer_params[3*qi+1+param_begin], layer_params[3*qi+2+param_begin]
        qc.u(theta, phi, lam, qi+1)
    # param_begin = (3*(num_qubits-1)*2)

    # second 
    # pairs = [(1,2),(3,4),(5,6)]  # 注意Qiskit的qubit编号从0开始
    # for qi in range(num_qubits-1):
    #     theta, phi, lam = layer_params[3*qi+param_begin], layer_params[3*qi+1+param_begin], layer_params[3*qi+2+param_begin]
    #     qc.u(theta, phi, lam, qi+1)
    # param_begin = (3*(num_qubits-1)*3)
    # for (q1, q2) in pairs:
    #     qc.cz(q1, q2)
    # for qi in range(num_qubits-1):
    #     theta, phi, lam = layer_params[3*qi+param_begin], layer_params[3*qi+1+param_begin], layer_params[3*qi+2+param_begin]
    #     qc.u(theta, phi, lam, qi+1)
    return qc


def cost_function(params, num_layers, num_qubits, fixed_prev_params=None, record_loss=None, type_optimize=None):
    """
    计算当前参数下的状态保真度代价函数
    fixed_prev_params: 之前层的固定参数（用于逐层优化）
    """
    one_layer_para = 3 * num_qubits
    if type_optimize=='Optimize_All':
        new_params = []
        params_per_layer = 3*(2*num_qubits-2) # 36  # 根据create_layer的计算
        for i in range(int(len(params)/params_per_layer)-1):
            new_params.append(params[one_layer_para+i*params_per_layer:one_layer_para+(i+1)*params_per_layer])
        fixed_prev_params = new_params
        params1 = params[-params_per_layer:]
    qc = QuantumCircuit(num_qubits)
    
    # for qi in range(num_qubits):
    #     qc.h(qi)

    for qi in range(num_qubits):
        theta, phi, lam = params[3*qi], params[3*qi+1], params[3*qi+2]
        qc.u(theta, phi, lam, qi)

    if fixed_prev_params is not None:
        # 添加之前优化好的层
        for l in range(len(fixed_prev_params)):
            layer_params = fixed_prev_params[l]
            qc.compose(create_layer(num_qubits, layer_params), inplace=True)

    # 添加当前层（正在优化的层）
    current_layer_params = params1 if fixed_prev_params is not None else params
    qc.compose(create_layer(num_qubits, current_layer_params), inplace=True)


    # 模拟电路
    simulator = Aer.get_backend('statevector_simulator')
    # result = execute(qc, simulator).result()
    result = simulator.run(qc).result()
    final_state = result.get_statevector(qc)

    # 计算与目标态的距离（1-保真度）
    fid = state_fidelity(final_state, target_state)
    if record_loss is not None:
        record_loss[num_layers].append(1-fid)

    # final_state = final_state.data
    # final_state = final_state[:120]
    # Expect_Ener = np.dot(np.conj(final_state.T), np.dot(M_E, final_state)) + 100 * np.dot(np.conj(final_state[-8:].T), final_state[-8:])
    # if record_loss is not None:
    #     record_loss[num_layers].append(Expect_Ener)
    
    return 1 - fid
    # return Expect_Ener


def Opt_VQE(max_layers=5, threshold=0.01):
    """
    层数逐渐增加，但是还是直接优化。
    :param max_layers: 最大层数
    :param threshold: 收敛阈值
    """
    # 存储损失的列表
    loss_history = []
    all_loss = defaultdict(list)


    # def callback(eval_count, parameters, value, std):
    #     loss_history.append(value)
    #     print(f"Iteration {eval_count}: Loss = {value:.4f}")

    num_qubits = 7
    # optimizer = COBYLA(maxiter=50000)
    best_params_per_layer = np.array([])
    initial_params = np.abs(np.random.normal(loc=0, scale=2, size=3*num_qubits))
    initial_params = np.mod(initial_params, 2*np.pi)  # 保证范围
    best_params_per_layer = np.append(best_params_per_layer, initial_params)
    
    params_per_layer = 3*(2*num_qubits-2) # 42  # 根据create_layer的计算
    
    for num_layers in range(1, max_layers+1):
        print(f"\nOptimizing layer {num_layers}...")
        
        # 初始化当前层参数
        # initial_params = np.random.uniform(0, 2*np.pi, size=params_per_layer)
        initial_params = np.abs(np.random.normal(loc=0, scale=2-(0.4)*(num_layers)/max_layers, size=params_per_layer))
        # initial_params = np.zeros(params_per_layer)
        initial_params = np.mod(initial_params, 2*np.pi)  # 保证范围    
        
        # 优化当前层（固定之前层的参数）
        best_params_per_layer = np.append(best_params_per_layer, initial_params)
        # result = optimizer.minimize(
        #     fun=lambda p: cost_function(p, num_layers, num_qubits,  record_loss=all_loss, type_optimize='Optimize_All'),
        #     x0=best_params_per_layer
        # ) # fixed_prev_params=best_params_per_layer,
        result = minimize(
            fun=lambda p: cost_function(p, num_layers, num_qubits,  record_loss=all_loss, type_optimize='Optimize_All'),
            x0=best_params_per_layer,
            method='COBYLA',            # 也可用 'SLSQP', 'L-BFGS-B' COBYLA 等
            options={'maxiter': 15000}    # 最大迭代次数
        ) # fixed_prev_params=best_params_per_layer,
        # 保存当前层优化结果
        best_params_per_layer = result.x
        current_error = result.fun # all_loss[num_layers][-1]

        # 采用粒子群算法。
        # from pyswarm import pso
        # lb = [0 ] * len(best_params_per_layer)
        # ub = [2*np.pi ] * len(best_params_per_layer)
        # best_params_per_layer, current_error = pso(cost_function, lb, ub, swarmsize=50, maxiter=5000, args=(num_layers, num_qubits, None, all_loss,'Optimize_All'))

        print(f"Layer {num_layers} achieved error: {current_error:.4f}")
        # print(all[num_layers][-1])
        if current_error < threshold:
            print(f"Converged at layer {num_layers} with error {current_error:.4f}")
            break
    print(cost_function(best_params_per_layer, 4, num_qubits,type_optimize='Optimize_All'))
    new_params = []
    one_layer_param = 3*num_qubits
    new_params.append(best_params_per_layer[:one_layer_param])
    for i in range(int(len(best_params_per_layer)/params_per_layer)):
        new_params.append(best_params_per_layer[one_layer_param+i*params_per_layer:one_layer_param+(i+1)*params_per_layer])
    best_params_per_layer1 = new_params
    # print('para', best_params_per_layer[:21])
    return best_params_per_layer1, num_layers, all_loss

def Check_Opt():
    # 运行优化（逐层优化）
    num_qubits = 7
    optimal_params_per_layer, final_num_layers, loss = Opt_VQE(max_layers=5)

    # print(optimal_params_per_layer[0])

    # print(optimal_params_per_layer[-1], len(optimal_params_per_layer))
    # 构建最终电路
    final_circuit = QuantumCircuit(7)
    # for qi in range(7):
    #     final_circuit.h(qi)
    
    for qi in range(num_qubits):
        theta, phi, lam = optimal_params_per_layer[0][3*qi], optimal_params_per_layer[0][3*qi+1], optimal_params_per_layer[0][3*qi+2]
        final_circuit.u(theta, phi, lam, qi)
    # print(optimal_params_per_layer[0])

    for i, layer_params in enumerate(optimal_params_per_layer):
        print(i)
        if i>0:
            final_circuit.compose(create_layer(7, layer_params), inplace=True)

    # print("\nFinal circuit:")
    # print(final_circuit.draw())

    # 验证最终保真度
    simulator = Aer.get_backend('statevector_simulator')
    # result = execute(final_circuit, simulator).result()
    result = simulator.run(final_circuit).result()
    final_state = result.get_statevector(final_circuit)
    fidelity = state_fidelity(final_state, target_state)
    print(f"\nFinal state fidelity: {fidelity:.4f}")

    with h5py.File('optimization_results_particle_Interaction.h5', 'w') as f:
        for key, value in loss.items():
            f.create_dataset(str(key), data=value)
        f.create_dataset('final_state', data=final_state.data)
        f.create_dataset('target_state', data=target_state)


    for i in range(1,15):
        plt.plot(loss[i])
    plt.show()

    # show the state and target state
    fig, ax = plt.subplots(1,2)
    u = -1.5 # u 太大之后使topological trivial的。
    gamma = 0.2
    num_xsite = 2
    num_ysite = 4
    total_particle = 2
    U = 1.
    color_string = 'gist_heat_r' # 'plasma'  # 'seismic'这一个就完全使红蓝的。
    QWZ1 = ParticleConservedSystem(num_xsite, num_ysite, total_particle)
    popu_indice = QWZ1.Calcu_Occup()

    print(type(final_state.data), type(target_state))

    popu_eig1 = np.zeros(QWZ1.total_mode)
    for k in range(2*num_xsite*num_ysite): # 应该要加2。
        #print(temp_state[popu_indice[k]])
        popu_eig1[k] = np.sum(np.abs(final_state.data[popu_indice[k]])**2)
    popu_eig1 = popu_eig1[::2] + popu_eig1[1::2]
    popu_eig2 = np.zeros(QWZ1.total_mode)
    for k in range(2*num_xsite*num_ysite): # 应该要加2。
        #print(temp_state[popu_indice[k]])
        popu_eig2[k] = np.sum(np.abs(target_state[popu_indice[k],0])**2)
    popu_eig2 = popu_eig2[::2] + popu_eig2[1::2]
    vmin, vmax = np.min([np.min(popu_eig1), np.min(popu_eig2)]), np.max([np.max(popu_eig1), np.max(popu_eig2)])

    im1 = ax[0].imshow(popu_eig1.reshape(num_xsite, num_ysite).T, origin='lower', cmap=color_string, vmin=vmin,vmax=vmax)
    # plt.colorbar(im1, ax=ax[0])  # 为第一个子图添加 colorbar
    ax[0].set_title('final state')

    im2 = ax[1].imshow(popu_eig2.reshape(num_xsite, num_ysite).T, origin='lower', cmap=color_string, vmin=vmin,vmax=vmax)
    ax[1].set_title('target state')
    fig.suptitle(f'Fidelity: {fidelity:.4f}')
    plt.colorbar(im2, ax=ax[1])  # 为第一个子图添加 colorbar
    # plt.colorbar()
    plt.tight_layout()  # 调整子图间距，防止重叠
    plt.show()

if __name__=="__main__":
    start_time = time.time()
    Check_Opt()
    print((time.time()-start_time)/60)