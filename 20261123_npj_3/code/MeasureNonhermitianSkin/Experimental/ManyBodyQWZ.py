"""
Q: 两粒子的能量不能表示为单粒子的之和
正确考虑费米子对易关系。可以看作一维的，只要费米子算符有一个顺序就可以。
态|000...0...i...0>要跃迁为|000...j...0...0>
那么对于的算符是a†_ja_i \Pi_{j0>j} a†_{j0} \dot \Pi_{j>k>i} a†_{k} \dot a†_{i} \dot \Pi_{i>k} a†_{k} 
首先穿过第一个\Pi是一起穿越，不出现符号，然后a†_j留下，然后a_i独自穿越直到a_{i}a†_{i}变为1。所以出现的符号就是独自穿越留下的。

Q: 双粒子态的<n>是不是和两个单粒子的<n>相加的结果相同？
# 双粒子的到底是不是和单粒子的相同。我理解应该不是，一个简单的例子，非厄米趋肤效应，所有本征态都是局域在边界，填充两三个态就会导致粒子数大于1，显然不合理，实际的情况是由于费米子的反对称，所以在写为反对称的时候，由于都处于边界所以会有|2..>,因此虽然波函数简单得到，但是粒子数并不是二者之和。
比如我们参考的非互易非厄米的文章。
"""
import numpy as np
import time
from scipy.linalg import eigh
from ComputationalPhysics.Package.BitOperation import ReadBit, FlipBit, PopCntBit, JointBit, PickBit
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.linalg import expm

from tqdm import tqdm 

def IPR(state, sign=None):
    """
    return the IPR of the given state"""
    if sign=='occ':
        pass
    else:
        state = np.abs(state)**2
    # state = state[::2]+state[1::2] # 没办法把元胞的加起来。因为不是单体。
    return np.sum(np.abs(state)**2) / np.sum(state)

# def Occu_IPR(state):
    
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
            from collections import defaultdict
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

class ParticleConservedSystem_Coo:
    """
    The same as the last function, but return coo sparse matrix"""
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
        # self.hamiltonian = np.zeros((self.dimension, self.dimension), dtype=np.complex64)
        self.row = []
        self.col = []
        self.data = []

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
            # self.hamiltonian[idx, idx] += u*np.sum([ReadBit(state, i) for i in range(0, self.total_mode, 2)]) - u*np.sum([ReadBit(state, i) for i in range(1, self.total_mode, 2)])
            self.row.append(idx)
            self.col.append(idx)
            self.data.append(u*np.sum([ReadBit(state, i) for i in range(0, self.total_mode, 2)]) - u*np.sum([ReadBit(state, i) for i in range(1, self.total_mode, 2)]))
            # print(np.sum([ReadBit(state, i) for i in range(0, 2*self.num_x*self.num_y, 2)]))

    def add_NH(self, gamma: float, which_dissi:int=0):
        """在y=1的边界添加非厄米项: i * gamma * a†_m a_m
        which dissi:表示在y等于几加耗散。默认在y=1的边界
        """
        for idx, state in enumerate(self.basis): #第几个basis，矩阵的第几个。
            #print(idx, state, bin(state), ReadBit(state,0), ReadBit(state,1))
            # self.hamiltonian[idx, idx] += 1j * gamma*np.sum([ReadBit(state, i) for i in range(2*which_dissi, self.total_mode, 2*self.num_y)]) + 1j * gamma * np.sum([ReadBit(state, i) for i in range(2*which_dissi+1, self.total_mode, 2*self.num_y)])
            self.row.append(idx)
            self.col.append(idx)
            self.data.append(1j * gamma*np.sum([ReadBit(state, i) for i in range(2*which_dissi, self.total_mode, 2*self.num_y)]) + 1j * gamma * np.sum([ReadBit(state, i) for i in range(2*which_dissi+1, self.total_mode, 2*self.num_y)]))

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
                        # self.hamiltonian[jdx1, idx] += 1.0/2 * sign
                        # self.hamiltonian[idx, jdx1] += 1.0/2 * sign
                        self.row.append(jdx1)
                        self.col.append(idx)
                        self.data.append(1.0/2 * sign)
                        self.row.append(idx)
                        self.col.append(jdx1)
                        self.data.append(1.0/2 * sign)
                    distance2 = 2*self.num_y + 1 # mx A->mx+1，B
                    if ReadBit(state, i+distance2)==0 and i+distance2<self.total_mode:
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        # self.hamiltonian[jdx2, idx] += 1j/2 * sign
                        # self.hamiltonian[idx, jdx2] += -1j/2 * sign
                        self.row.append(jdx2)
                        self.col.append(idx)
                        self.data.append(1j/2 * sign)
                        self.row.append(idx)
                        self.col.append(jdx2)
                        self.data.append(-1j/2 * sign)

            # from B particle
            for i in range(1, self.total_mode,2): # A sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 2*self.num_y # mx B->mx+1，B
                    if ReadBit(state, i+distance1)==0 and i+distance1<self.total_mode:
                        #print(distance1, i, bin(state))
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        # self.hamiltonian[jdx1, idx] += -1.0/2 * sign
                        # self.hamiltonian[idx, jdx1] += -1.0/2 * sign
                        self.row.append(jdx1)
                        self.col.append(idx)
                        self.data.append(-1.0/2 * sign)
                        self.row.append(idx)
                        self.col.append(jdx1)
                        self.data.append(-1.0/2 * sign)
                    distance2 = 2*self.num_y-1 # mx B->mx+1，A
                    if ReadBit(state, i+distance2)==0 and i+distance2<self.total_mode:
                        state2 = FlipBit(FlipBit(state, i), i+distance2) # mx A->mx+1，B
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        # self.hamiltonian[jdx2, idx] += 1j/2 * sign
                        # self.hamiltonian[idx, jdx2] += -1j/2 * sign
                        self.row.append(jdx2)
                        self.col.append(idx)
                        self.data.append(1j/2 * sign)
                        self.row.append(idx)
                        self.col.append(jdx2)
                        self.data.append(-1j/2 * sign)
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
                        # self.hamiltonian[jdx1, idx] += 1.0/2 * sign
                        # self.hamiltonian[idx, jdx1] += 1.0/2 * sign
                        self.row.append(jdx1)
                        self.col.append(idx)
                        self.data.append(1.0/2 * sign)
                        self.row.append(idx)
                        self.col.append(jdx1)
                        self.data.append(1.0/2 * sign)
                    distance2 = 3 # my A->my+1，B
                    if ReadBit(state, i+distance2)==0 and i not in list(range(2*self.num_y-2, self.total_mode, 2*self.num_y)):
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        # self.hamiltonian[jdx2, idx] += -1/2 * sign
                        # self.hamiltonian[idx, jdx2] += -1/2 * sign
                        self.row.append(jdx2)
                        self.col.append(idx)
                        self.data.append(-1.0/2 * sign)
                        self.row.append(idx)
                        self.col.append(jdx2)
                        self.data.append(-1.0/2 * sign)

            # from B particle
            for i in range(1, self.total_mode,2): # A sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 2 # my B->my+1，B
                    if ReadBit(state, i+distance1)==0 and i not in list(range(2*self.num_y-1, self.total_mode, 2*self.num_y)):
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        # self.hamiltonian[jdx1, idx] += -1.0/2 * sign
                        # self.hamiltonian[idx, jdx1] += -1.0/2 * sign
                        self.row.append(jdx1)
                        self.col.append(idx)
                        self.data.append(-1.0/2 * sign)
                        self.row.append(idx)
                        self.col.append(jdx1)
                        self.data.append(-1.0/2 * sign)
                    distance2 = 1 # my B->my+1，A
                    if ReadBit(state, i+distance2)==0 and i not in list(range(2*self.num_y-1, self.total_mode, 2*self.num_y)):
                        state2 = FlipBit(FlipBit(state, i), i+distance2) # my B->my+1，A
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        # self.hamiltonian[jdx2, idx] += 1/2 * sign #-1/2
                        # self.hamiltonian[idx, jdx2] += 1/2 * sign #-1/2
                        self.row.append(jdx2)
                        self.col.append(idx)
                        self.data.append(1.0/2 * sign)
                        self.row.append(idx)
                        self.col.append(jdx2)
                        self.data.append(1.0/2 * sign)
                
    def add_interaction_intra(self, U):
        """添加相互作用U n_{x,y, A} n_{x,y, B}，当一个元胞内有两个粒子能量变大，趋向于不呆在或者呆在一个元胞内。或者是相互之间的，这样chiral skin会让它趋向于局域，但是相互作用可能让它更局域或者弥散。"""
        self.inter = np.zeros_like(self.hamiltonian)
        for idx, state in enumerate(self.basis):
            # # U n_{x,y, A} n_{x,y, B}
            for i in range(0, self.total_mode, 2): # A sublattice.
                if ReadBit(state, i)==1 and ReadBit(state, i+1)==1:
                    # self.inter[idx, idx] += U
                    self.row.append(idx)
                    self.col.append(idx)
                    self.data.append(U)
                    self.row.append(idx)
                    self.col.append(idx)
                    self.data.append(U)
        self.hamiltonian += self.inter
        return self.inter

    def add_interaction_inter(self, U):
        """添加相互作用U( n_{x,y, A}+n_{x,y, B}) * (n_{x+1,y, A}+n_{x+1,y, B}))，当近邻元胞有不同的电子数时，会出现相互作用项。这样chiral skin会让它趋向于局域，但是相互作用可能让它更局域或者弥散。但是现象好像是相互作用会使它更局域"""
        # self.inter = np.zeros_like(self.hamiltonian)
        for idx, state in enumerate(self.basis):
            # X: (U n_{x,y, A}+n_{x,y, B}) * (U n_{x+1,y, A}+n_{x+1,y, B}). Y同样。
            distance1 = 2*self.num_y
            for i in range(0, self.total_mode, 2): # A sublattice.
                # Y direction
                if i not in list(range(2*self.num_y-2, self.total_mode, 2*self.num_y)):
                    # self.inter[idx, idx] += U*(ReadBit(state, i)+ReadBit(state,i+1))*(ReadBit(state, i+2)+ReadBit(state,i+3))
                    self.row.append(idx)
                    self.col.append(idx)
                    self.data.append(U*(ReadBit(state, i)+ReadBit(state,i+1))*(ReadBit(state, i+2)+ReadBit(state,i+3)))
                # X direction
                if i+distance1<self.total_mode:
                    # self.inter[idx, idx] += U*(ReadBit(state, i)+ReadBit(state,i+1))*(ReadBit(state, i+distance1)+ReadBit(state,i+distance1+1))
                    self.row.append(idx)
                    self.col.append(idx)
                    self.data.append(U*(ReadBit(state, i)+ReadBit(state,i+1))*(ReadBit(state, i+distance1)+ReadBit(state,i+distance1+1)))
        # self.hamiltonian += self.inter
        # return self.inter


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
            from collections import defaultdict
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



def Comp():
    num_x = 4
    num_y = 4
    many_body = 3
    print('---One particle:---')
    QWZ1 = ParticleConservedSystem(num_x, num_y, 1)
    QWZ1.add_onsite_term(-1.5)
    QWZ1.add_NH(0.2)
    QWZ1.add_xhopping_term()
    QWZ1.add_yhopping_term()
    print('---Get Hamiltonian---', QWZ1.dimension)
    ham1 = QWZ1.hamiltonian
    #print(ham1)
    print('---Two particle:---')
    QWZ2 = ParticleConservedSystem(num_x, num_y, many_body)
    QWZ2.add_onsite_term(-1.5)
    QWZ2.add_NH(0.2)
    QWZ2.add_xhopping_term()
    QWZ2.add_yhopping_term()
    QWZ2.add_interaction_intra(1.)
    print('---Get Hamiltonian---', QWZ2.dimension)
    ham2 = QWZ2.hamiltonian
    #print(ham2)

    eigval1, eigvec1 = np.linalg.eig(ham1)
    eigvec1 = eigvec1[:, np.argsort(np.abs(eigval1.real))]
    eigval1 = eigval1[np.argsort(np.abs(eigval1.real))]
    popu1 = QWZ1.Calcu_Occup(eigvec1[:,0])
    print('population:', popu1[::2]+popu1[1::2])
    eigval2, eigvec2 = np.linalg.eig(ham2)
    #print(eigval1, eigval2)
    new_eigval = []
    for i in range(len(eigval1)-1):
        for j in range(i+1, len(eigval1)):
            new_eigval.append(eigval1[i] + eigval1[j] )
    new_eigval =np.array(sorted(new_eigval))
    #print(eigval1, new_eigval, eigval2, len(new_eigval), len(eigval2))
    import matplotlib.pyplot as plt
    # plt.plot(new_eigval,'o-')
    # plt.plot(eigval2,'^-')
    # plt.show()
    # check the results for non-Hermitian.
    #print(eigval1)
    IPR_ham2 = np.zeros(len(eigval2))
    for i in range(len(IPR_ham2)):
        IPR_ham2[i] = IPR(eigvec2[:,i])
    IPR2_ham2 = np.zeros(len(eigval2))
    for i in range(len(IPR_ham2)):
        temp_occu = QWZ2.Calcu_Occup(eigvec2[:,i])
        IPR2_ham2[i] = IPR(temp_occu)
    
    plt.plot(new_eigval.real, new_eigval.imag, 'o')
    plt.plot(eigval2.real, eigval2.imag, '^')
    # plt.scatter(eigval2.real, eigval2.imag,c=IPR2_ham2, s=50, cmap='viridis',vmin=0)
    # plt.colorbar(label='IPR')
    #print((eigval1[0]+eigval1[1]), eigval1[0], eigval1[1])
    plt.scatter((np.sum(eigval1[:many_body])).real,(np.sum(eigval1[:many_body])).imag, s=100, marker='o', facecolors='none', edgecolors='r')
    # plt.xlim([-0.5,0.5])
    # plt.ylim([0.095,0.10])
    plt.show()

    absolute_diff = np.abs(eigval2 - (np.sum(eigval1[:many_body])))
    popu2 = QWZ2.Calcu_Occup(eigvec2[:, np.argmin(absolute_diff)])
    print(eigval2[np.argmin(absolute_diff)],'population:', popu2[::2]+popu2[1::2], 2*popu1[::2]+popu1[1::2]*2)

    popu1 = (popu1[::2]+popu1[1::2]).reshape(num_x, num_y).T
    popu2 = (popu2[::2]+popu2[1::2]).reshape(num_x, num_y).T
    fig, ax = plt.subplots(1,2)
    color_string = 'Reds'
    ax[0].imshow(popu1,cmap=color_string)
    ax[1].imshow(popu2,cmap=color_string)
    for x in range(num_x):
        for y in range(num_y):
            ax[0].text(x,y, f'{popu1[y, x]:.3f}', ha='center', va='center', color='black', fontsize=10)
            ax[1].text(x,y, f'{popu2[y, x]:.3f}', ha='center', va='center', color='black', fontsize=10)
    plt.show()

def Comp2():
    num_x = 5
    num_y = 5
    #print(ham1)
    print('---Two particle:---')
    QWZ2 = ParticleConservedSystem(num_x, num_y, 2)
    QWZ2.add_onsite_term(-1.5)
    # QWZ2.add_NH(0.2)
    QWZ2.add_xhopping_term()
    QWZ2.add_yhopping_term()
    QWZ2.add_interaction_inter(-0.2)
    print('---Get Hamiltonian---', QWZ2.dimension)
    ham2 = QWZ2.hamiltonian
    #print(ham2)

    eigval2, eigvec2 = np.linalg.eig(ham2)
    eigvec2 = eigvec2[:, np.argsort(np.abs(eigval2.real))]
    eigval2 = eigval2[np.argsort(np.abs(eigval2.real))]
    #print(eigval1, eigval2)
    #print(eigval1, new_eigval, eigval2, len(new_eigval), len(eigval2))
    import matplotlib.pyplot as plt
    # plt.plot(new_eigval,'o-')
    # plt.plot(eigval2,'^-')
    # plt.show()
    # check the results for non-Hermitian.
    #print(eigval1)
    IPR_ham2 = np.zeros(len(eigval2))
    for i in range(len(IPR_ham2)):
        IPR_ham2[i] = IPR(eigvec2[:,i])
    IPR2_ham2 = np.zeros(len(eigval2))
    for i in range(len(IPR_ham2)):
        temp_occu = QWZ2.Calcu_Occup(eigvec2[:,i])
        IPR2_ham2[i] = IPR(temp_occu)
    
    plt.plot(eigval2.real, eigval2.imag, '^')
    # plt.scatter(eigval2.real, eigval2.imag,c=IPR2_ham2, s=50, cmap='viridis',vmin=0)
    # plt.colorbar(label='IPR')
    #print((eigval1[0]+eigval1[1]), eigval1[0], eigval1[1])
    # plt.xlim([-0.5,0.5])
    # plt.ylim([0.095,0.10])
    plt.show()

    # 假设 eigvec2 是您的本征向量矩阵，每一列是一个本征态
    num_states_to_plot = 5  # 要绘制的前几个本征态

    # 创建1行5列的子图
    fig, axes = plt.subplots(1, num_states_to_plot, figsize=(20, 4))

    # 遍历前几个本征态
    for i in range(num_states_to_plot):
        # 计算当前本征态的占据数
        popu = QWZ2.Calcu_Occup(eigvec2[:, i])
        popu = (popu[::2] + popu[1::2]).reshape(num_x, num_y).T
        
        # 绘制图像
        im = axes[i].imshow(popu, cmap='Reds')
        axes[i].set_title(f'State {i}\nEigenvalue: {eigval2[i]:.4f}')
        
        # 添加数值标签
        for x in range(num_x):
            for y in range(num_y):
                axes[i].text(x, y, f'{popu[y, x]:.3f}', 
                            ha='center', va='center', 
                            color='black', fontsize=8)
        
        # 添加颜色条
        fig.colorbar(im, ax=axes[i])

    plt.tight_layout()
    plt.show()

    # popu = 0
    # for i in range(len(eigval2)):
    #     plt.plot(np.abs(eigvec2[:, i]))
    # plt.show()
    ipr_list = np.zeros(len(eigval2))
    for i in range(len(eigval2)):
        ipr_list[i] = IPR(eigvec2[:, i])
    ipr_list = sorted(ipr_list)
    plt.plot(ipr_list,'.')
    plt.show()

def FockSkin_InterNonHermitian():
    num_x = 5
    num_y = 5
    gamma_list = np.linspace(0, 0.2, 6)
    inter_list = np.linspace(0, 0.2, 6)
    inter_list = - inter_list
    ipr_non_list = np.zeros((len(gamma_list), num_x*num_y*(2*num_x*num_y-1)))
    ipr_inter_list = np.zeros((len(gamma_list), num_x*num_y*(2*num_x*num_y-1)))
    occu_non_list = np.zeros((len(gamma_list), num_x*num_y*(2*num_x*num_y-1))) 
    occu_inter_list = np.zeros((len(gamma_list), num_x*num_y*(2*num_x*num_y-1)))
    # ipr_non_list = np.zeros((len(gamma_list), num_x*num_y*2))
    # ipr_inter_list = np.zeros((len(gamma_list), num_x*num_y*2))
    # occu_non_list = np.zeros((len(gamma_list), num_x*num_y*2)) 
    # occu_inter_list = np.zeros((len(gamma_list), num_x*num_y*2))
    #print(ham1)
    for i in range(len(gamma_list)):
        QWZ2 = ParticleConservedSystem(num_x, num_y, 2)
        QWZ2.add_onsite_term(-1.5)
        QWZ2.add_NH(gamma_list[i])
        QWZ2.add_xhopping_term()
        QWZ2.add_yhopping_term()
        # QWZ2.add_interaction_inter(-0.2)
        print('---Get Hamiltonian---', QWZ2.dimension)
        ham2 = QWZ2.hamiltonian
        popu_indice = QWZ2.Calcu_Occup()

        eigval2, eigvec2 = np.linalg.eig(ham2)
        temp_ipr_list = np.zeros(len(eigval2))
        temp_occu_list = np.zeros(len(eigval2))

        #eigen_state = np.abs(eigen_state)**2
        # for i in range(len(population)):
        #     population[i] = np.sum(eigen_state[popu_indice[i]])

        for j in range(len(eigval2)):
            temp_ipr_list[j] = IPR(eigvec2[:, j])
            temp_eigvec = np.abs(eigvec2[:, j])**2
            temp_population = np.zeros(QWZ2.total_mode)
            for k in range(num_x*num_y):
                temp_population[k] = np.sum(temp_eigvec[popu_indice[k]]) # IPR()
            temp_occu_list[j] = IPR(temp_population, 'occ')
        temp_ipr_list = sorted(temp_ipr_list)
        ipr_non_list[i] = temp_ipr_list
        temp_occu_list = sorted(temp_occu_list)
        occu_non_list[i] = temp_occu_list

    for i in range(1, len(inter_list)):
        QWZ2 = ParticleConservedSystem(num_x, num_y, 2)
        QWZ2.add_onsite_term(-1.5)
        # QWZ2.add_NH(0.2)
        QWZ2.add_xhopping_term()
        QWZ2.add_yhopping_term()
        QWZ2.add_interaction_inter(inter_list[i])
        print('---Get Hamiltonian---', QWZ2.dimension)
        ham2 = QWZ2.hamiltonian

        eigval2, eigvec2 = np.linalg.eig(ham2)
        temp_ipr_list = np.zeros(len(eigval2))
        temp_occu_list = np.zeros(len(eigval2))
        for j in range(len(eigval2)):
            temp_ipr_list[j] = IPR(eigvec2[:, j])
            #temp_occu_list[j] = IPR(QWZ2.Calcu_Occup(eigvec2[:, j]))
            temp_eigvec = np.abs(eigvec2[:, j])**2
            temp_population = np.zeros(QWZ2.total_mode)
            for k in range(num_x*num_y):
                temp_population[k] = np.sum(temp_eigvec[popu_indice[k]]) # IPR()
            temp_occu_list[j] = IPR(temp_population, 'occ')
        temp_ipr_list = sorted(temp_ipr_list)
        ipr_inter_list[i] = temp_ipr_list
        temp_occu_list = sorted(temp_occu_list)
        occu_inter_list[i] = temp_occu_list
    print(ham2)
    #np.savez('1.npz', ipr_inter_list=ipr_inter_list, ipr_non_list=ipr_non_list, occu_non_list=occu_non_list, occu_inter_list=occu_inter_list)

    # with np.load('3.npz') as files:
    #     ipr_inter_list = files['ipr_inter_list']
    #     ipr_non_list = files['ipr_non_list']
    #     occu_inter_list = files['occu_inter_list']
    #     occu_non_list = files['occu_non_list']
    # gamma_list = np.linspace(0, 0.2, 6)
    # inter_list = np.linspace(0, 0.2, 6)

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # 生成示例数据
    # x = np.linspace(0, 10, 100)
    # gamma_list = np.linspace(0.1, 1, 10)
    # inter_list = np.linspace(0.2, 2, 10)

    x = np.linspace(1, ipr_inter_list.shape[1]+1, ipr_inter_list.shape[1])
    #print(ipr_inter_list.shape,ipr_inter_list.shape[1])

    # 高级颜色配置
    highlight_color = '#00FF00' # '#E64B35'  # 强调色 (Brick Red)
    color_config = {
        'non_inter': plt.cm.Reds(np.linspace(0.2, 1.0, len(gamma_list))),
        'inter': plt.cm.Blues(np.linspace(0.2, 0.9, len(inter_list)))
    }

    # 创建画布
    font_title = 16
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=100)
    plt.subplots_adjust(wspace=0.25)

    # 通用绘图函数
    def plot_dataset(ax, non_data, inter_data, ylabel, zoom_range):
        """统一绘图逻辑"""
        # 绘制基准线
        ax.plot(x, non_data[0], 
                color=highlight_color,
                linewidth=3,
                marker='*',
                markersize=8,
                zorder=10,
                label='Baseline (γ=0)')
               # markeredgecolor='black',
        
        # 绘制其他非相互作用数据
        for i in range(1, len(gamma_list)):
            ax.plot(x, non_data[i],
                    color=color_config['non_inter'][i],
                    linestyle='-',
                    linewidth=1,
                    marker='o',
                    markersize=4,
                    alpha=0.7)
        
        # 绘制相互作用数据
        for i in range(1, len(inter_list)):
            ax.plot(x, inter_data[i],
                    color=color_config['inter'][i],
                    linestyle='--',
                    linewidth=1.2,
                    marker='s',
                    markersize=5,
                    alpha=0.8) # markeredgecolor='k',
        
        # 添加放大镜
        ax_inset = inset_axes(ax, 
                            width="40%", 
                            height="40%",
                            loc='upper left',
                            borderpad=1.5)
        
        #绘制放大区域
        zoom_mask = (x >= zoom_range[0]) & (x <= zoom_range[1])
        ax_inset.plot(x[zoom_mask], non_data[0][zoom_mask], 'o--',
                    color=highlight_color, lw=2)
        for i in range(1, len(gamma_list)):  # 仅显示前5组对比数据
            ax_inset.plot(x[zoom_mask], non_data[i][zoom_mask], 'o--',
                        color=color_config['non_inter'][i], 
                        lw=1, alpha=0.7)
            ax_inset.plot(x[zoom_mask], inter_data[i][zoom_mask], 'o--',
                        color=color_config['inter'][i],
                        lw=1, alpha=0.7)
        
        # 设置放大镜样式
        ax_inset.set_xlim(zoom_range)
        y_frame_min = np.min([np.min(non_data[-1][zoom_mask]), np.min(inter_data[-1][zoom_mask])])*0.9
        y_frame_max = np.max([np.max(non_data[-1][zoom_mask]), np.max(inter_data[-1][zoom_mask])])*1.1

        # rect_pos = ax.get_position()
        # line_start = (zoom_range[0], y_frame_max)  # 放大区域框的左上角
        # line_end = (rect_pos.x0 + 0.1, rect_pos.y0 + rect_pos.height - 0.1)  # 子图中的对应位置
        # ax.annotate('', 
        #      xy=line_start, 
        #      xytext=line_end,
        #      arrowprops=dict(arrowstyle='->', color='red'))

        ax_inset.set_ylim(y_frame_min, y_frame_max)
        ax_inset.set_xticks([])  # 隐藏 x 轴刻度
        ax_inset.set_yticks([])  # 隐藏 y 轴刻度
        ax_inset.grid(alpha=0.3)
        ax.indicate_inset_zoom(ax_inset, edgecolor="gray")
        
        # 坐标轴设置
        ax.set_xlabel('Number', fontsize=font_title)
        ax.set_ylabel(ylabel, fontsize=font_title)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)
        #ax.set_xlim(0, 10)
        
        # 智能图例
        # handles, labels = ax.get_legend_handles_labels()
        # new_handles = [handles[0], handles[1], handles[len(gamma_list)]]
        # new_labels = [labels[0], 'Non-interacting', 'Interacting']
        # ax.legend(new_handles, new_labels, 
        #         frameon=True, 
        #         framealpha=0.9,
        #         loc='upper right')

    # 绘制IPR子图
    plot_dataset(axs[0], ipr_non_list, ipr_inter_list, 
                'IPR', 
                zoom_range=(x[-10], x[-1]+1))

    # 绘制Occupancy子图
    plot_dataset(axs[1], occu_non_list, occu_inter_list,
                'IPR',
                zoom_range=(x[-10], x[-1]+1))

    # 全局标题
    # plt.suptitle('Eigenstate Analysis', 
    #             fontsize=14, 
    #             y=0.98, 
    #             fontweight='semibold')
    axs[0].set_title('Eigenstate', fontsize=font_title)
    axs[1].set_title('occupation',fontsize=font_title)
    plt.subplots_adjust(left=0.1, right=0.97, bottom=0.1, top=0.95, wspace=0.2, hspace=0.1)

    # 保存输出
    plt.show()


def FockSkin():
    num_x = 5
    num_y = 5
    print('---L particle:---')
    QWZ2 = ParticleConservedSystem(num_x, num_y, 2)
    QWZ2.add_onsite_term(-1.5)
    # QWZ2.add_NH(0.2)
    QWZ2.add_xhopping_term()
    QWZ2.add_yhopping_term()
    # QWZ2.add_interaction_inter(0.2)
    print('---Get Hamiltonian with dimension---', QWZ2.dimension)
    ham2 = QWZ2.hamiltonian
    #print(ham2)


    # eigval2, eigvec2 = np.linalg.eig(ham2)
    # #print(eigval1, eigval2)
    # #print(eigval1, new_eigval, eigval2, len(new_eigval), len(eigval2))
    # import matplotlib.pyplot as plt
    # # plt.plot(new_eigval,'o-')
    # # plt.plot(eigval2,'^-')
    # # plt.show()
    # # check the results for non-Hermitian.
    # #print(eigval1)
    # IPR_ham2 = np.zeros(len(eigval2))
    # for i in range(len(IPR_ham2)):
    #     IPR_ham2[i] = IPR(eigvec2[:,i])
    # IPR2_ham2 = np.zeros(len(eigval2))
    # for i in range(len(IPR_ham2)):
    #     temp_occu = QWZ2.Calcu_Occup(eigvec2[:,i])
    #     IPR2_ham2[i] = IPR(temp_occu)
    
    # #plt.plot(new_eigval.real, new_eigval.imag, 'o')
    # plt.plot(eigval2.real, eigval2.imag, '^')


    # plt.scatter(eigval2.real, eigval2.imag,c=IPR2_ham2, s=50, cmap='viridis',vmin=0)
    # plt.colorbar(label='IPR')
    #print((eigval1[0]+eigval1[1]), eigval1[0], eigval1[1])
    #plt.scatter((eigval1[0]+eigval1[1]).real,(eigval1[0]+eigval1[1]).imag, s=100, marker='o', facecolors='none', edgecolors='r')
    # plt.xlim([-0.5,0.5])
    # plt.ylim([0.095,0.10])
    # plt.show()

    #absolute_diff = np.abs(eigval2 - (eigval1[0]+eigval1[1]))
    
    # popu2 = QWZ2.Calcu_Occup(eigvec2[:, np.argmin(absolute_diff)])
    # print(eigval2[np.argmin(absolute_diff)],'population:', popu2[::2]+popu2[1::2], 2*popu1[::2]+popu1[1::2]*2)

    # popu1 = (popu1[::2]+popu1[1::2]).reshape(num_x, num_y).T
    # popu2 = (popu2[::2]+popu2[1::2]).reshape(num_x, num_y).T
    # fig, ax = plt.subplots(1,2)
    # color_string = 'Reds'
    # ax[0].imshow(popu1,cmap=color_string)
    # ax[1].imshow(popu2,cmap=color_string)
    # for x in range(num_x):
    #     for y in range(num_y):
    #         ax[0].text(x,y, f'{popu1[y, x]:.3f}', ha='center', va='center', color='black', fontsize=10)
    #         ax[1].text(x,y, f'{popu2[y, x]:.3f}', ha='center', va='center', color='black', fontsize=10)
    # plt.show()

    # from the state start.

    # str_i = '1'*2+'0'*(2*num_y-2)
    # str_i = str_i*num_x
    # str_i = '0b'+str_i
    # print(str_i, type(str_i))
    # initial_state = np.zeros((QWZ2.dimension), dtype='complex')
    # initial_state[QWZ2.state_index[int(str_i, 2)]] = 1.
    initial_state = np.zeros((QWZ2.dimension), dtype='complex')
    initial_state[0] = 1.
    t_list = np.linspace(0,30,30)
    fidelity_list = np.zeros_like(t_list)
    temp_state = initial_state.copy()
    for i in range(len(t_list)):
        print('---Processing:', i)
        temp_state = expm(-1j*QWZ2.hamiltonian*t_list[i]) @ temp_state
        temp_state = temp_state / np.linalg.norm(temp_state)
        fidelity_list[i] = np.abs(np.dot(initial_state, temp_state))**2
    print(temp_state)
    plt.plot(t_list, fidelity_list, 'o-')
    plt.ylabel('Fidelity')
    plt.xlabel('time--t')
    plt.show()


def CalPopu():
    num_x = 7
    num_y = 7
    print('---L particle:---')
    QWZ2 = ParticleConservedSystem(num_x, num_y, 2)
    QWZ2.add_onsite_term(-1.5)
    QWZ2.add_NH(0.2)
    QWZ2.add_xhopping_term()
    QWZ2.add_yhopping_term()
    # QWZ2.add_interaction_inter(-0.2)
    print('---Get Hamiltonian with dimension---', QWZ2.dimension)
    ham2 = QWZ2.hamiltonian
    popu_indice = QWZ2.Calcu_Occup()
    eigval2, eigvec2 = np.linalg.eig(ham2)
    temp_ipr_list = np.zeros(len(eigval2))
    temp_occu_list = np.zeros(len(eigval2))
    thresh_popu = 1/num_x
    record_popu_bigger = np.zeros(QWZ2.total_mode)  # 统计在某一个mode大于阈值的的态有多少个？
    # [i for i, x in enumerate(numbers) if x > 0.5]
    for j in range(len(eigval2)):
        #temp_ipr_list[j] = IPR(eigvec2[:, j])
        temp_eigvec = np.abs(eigvec2[:, j])**2
        temp_population = np.zeros(QWZ2.total_mode) # 该本征态某一个mode上popu的大小。
        for k in range(2*num_x*num_y): # 应该要加2。
            temp_population[k] = np.sum(temp_eigvec[popu_indice[k]]) # IPR()
        #temp_occu_list[j] = IPR(temp_population, 'occ')
        for idx, value_x in enumerate(temp_population):
            if value_x>=thresh_popu:
                record_popu_bigger[idx] += 1
    # 记录某一个mode上大于阈值的本征态有多少个。

    plt.plot(record_popu_bigger)
    plt.show()


def CalPopu_t(gamma, interaction, t_total):
    # 随机入射看演化。
    num_x = 7
    num_y = 7
    print('---L particle:---')
    QWZ2 = ParticleConservedSystem(num_x, num_y, 2)
    QWZ2.add_onsite_term(-1.5)
    QWZ2.add_NH(gamma)
    QWZ2.add_xhopping_term()
    QWZ2.add_yhopping_term()
    QWZ2.add_interaction_inter(interaction)
    print('---Get Hamiltonian with dimension---', QWZ2.dimension)
    ham2 = QWZ2.hamiltonian
    popu_indice = QWZ2.Calcu_Occup()
    t_list = np.linspace(0, t_total, 11) # 当为2的时候会先减少。
    thresh_popu = 1/(2*num_x)
    evo_ham = expm(-1j*ham2*t_list[1])
    NumSamp = 1000
    record_popu_bigger = np.zeros((len(t_list), QWZ2.total_mode))

    for temp_samp in tqdm(range(NumSamp)):
        # [print('samp', temp_samp) if temp_samp % 10 == 0 else None]
        initial_state = np.zeros(QWZ2.dimension, dtype=complex)
        initial_state[np.random.randint(QWZ2.dimension)] = 1.
        for tx in range(len(t_list)):
            temp_state = np.dot(evo_ham, initial_state)
            temp_state = temp_state / np.linalg.norm(temp_state)
            for k in range(2*num_x*num_y): # 应该要加2。
                #print(temp_state[popu_indice[k]])
                temp_population = np.sum(np.abs(temp_state[popu_indice[k]])**2) # IPR()
                if temp_population>thresh_popu:
                    record_popu_bigger[tx,k] += 1
            initial_state = temp_state
    # np.savez('CountState_interaction%.2f.npz'%interaction, record_popu_bigger=record_popu_bigger, t_list=t_list)
    # for i in range(len(t_list)):
    #     plt.plot(record_popu_bigger[i],label=t_list[i])
    # plt.legend()
    # plt.show()  # 在耗散格点的增幅增大，
    # 其实这里还可以统计分布，只在角落上的？通过图来看可以统计y=1的分布。

    plt.imshow(record_popu_bigger[-1].reshape(num_x, num_y*2).T, cmap='Reds')
    plt.show()

    # with np.load('CountState_interaction1.00.npz') as files:
    #     record_popu_bigger = files['record_popu_bigger']
    #     t_list = files['t_list']
    
    # import seaborn as sns
    # Set Seaborn style for better aesthetics
    # sns.set_theme(
    #     style="whitegrid", 
    #     rc={"axes.facecolor": "#fcfaf5"}
    # )

    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 6))

    # --------------------------------------------------
    # Custom color palette (adjust colors to enhance visibility)
    #colors = sns.color_palette("husl", n_colors=len(t_list))

    # ------------------------------
    # Plotting each line with distinct styling
    for idx, time_point in enumerate(t_list):

        # Convert time point to integer for coloring
        time_val = int(time_point)
        
        # Compute color based on normalized time value
        norm_time = time_val / max(int(k) for k in t_list)
        color = plt.cm.Blues(norm_time)  

        # Line properties (width, alpha)
        linewidth = (0.25 + 0.75 * norm_time)  # Wider lines for later times
        
        ax.plot(
            record_popu_bigger[idx], 'o--',
            label=f't={time_point}',
            color=color,
            linewidth=2.5,
            alpha=0.85,
        ) #             z_order=10 - time_val,  # Ensure earlier times are plotted first zorder=20 - time_val

    #-----------------------------------------------------
    # Configure axes & labels
    ax.set_xlabel(r"$2L_y x + y$", fontsize=13, labelpad=10)
    ax.set_ylabel(r"$\mathcal{P}_x$", fontsize=14) # \,\,\,(a.u.)
    ax.set_title("Time Evolution of Distribution", pad=30, fontsize=16)

    # Use scientific notation where needed
    if any(y >= 1e4 for y in ax.get_yticks()):
        plt.ticklabel_format(style='sci', scilimits=(0,0), useMathText=True)

    # Add legend with improved placement & formatting
    legend = ax.legend(
        title=r"Time Steps ($\Delta t$) ",
        loc='upper left',
        frameon=False,
        bbox_to_anchor=(0.97, 0.82),
        borderpad=1.5
    )#.get_frame().set_linewidth(0) # frame_color="#fcfaf5",

    # Enhance grid visibility
    ax.grid(True, which='both', linestyle=':', linewidth=0.6)

    # Improve colorbar integration (example for other plots)
    # plt.colorbar(...)  # If a colorbar is used...

    # Save the figure with tight layout and high resolution
    plt.tight_layout()
    plt.show()


def CalPopu_inter_non(gamma, t_final):
    # 随机入射，在末态时间看如何末态在各个格点的激发。不同相互作用强度
    # 就关注几个角上的，这个数目会随着非厄米强度和相互作用强度如何变化。但是需要固定时间了！
    # 随机入射看演化。
    num_x = 4
    num_y = 4
    QWZ2 = ParticleConservedSystem(num_x, num_y, 3)
    QWZ2.add_onsite_term(-1.5)
    QWZ2.add_NH(gamma)
    QWZ2.add_xhopping_term()
    QWZ2.add_yhopping_term()
    popu_indice = QWZ2.Calcu_Occup()
    single_hamiltonian = QWZ2.hamiltonian.copy()
    t_list = np.linspace(0, 20, 11) # 当为2的时候会先减少。
    #t_final = 60
    thresh_popu = 1/(2*num_x)
    NumSamp = 10000
    inter_list = np.linspace(-1, 1., 21)
    record_popu_bigger = np.zeros((len(inter_list), QWZ2.total_mode))

    for inter_dx, inter_value in enumerate(inter_list):
        print('---Get Hamiltonian with dimension---', QWZ2.dimension)
        temp_ham = QWZ2.add_interaction_inter(inter_value)
        ham2 = single_hamiltonian + temp_ham
        # # QWZ2 = ParticleConservedSystem(num_x, num_y, 2)
        # # QWZ2.add_onsite_term(-1.5)
        # # QWZ2.add_NH(0.2)
        # # QWZ2.add_xhopping_term()
        # # QWZ2.add_yhopping_term()
        # # QWZ2.add_interaction_inter(inter_value)
        # # ham2 = QWZ2.hamiltonian
        popu_indice = QWZ2.Calcu_Occup()

        popu_indice = QWZ2.Calcu_Occup()
        evo_ham = expm(-1j*ham2*t_final) # 这一步是时间久的，前面不是。
        for temp_samp in tqdm(range(NumSamp)):
            #[print('samp', temp_samp) if temp_samp % 10 == 0 else None]
            initial_state = np.zeros(QWZ2.dimension, dtype=complex)
            initial_state[np.random.randint(QWZ2.dimension)] = 1.
            temp_state = np.dot(evo_ham, initial_state)
            temp_state = temp_state / np.linalg.norm(temp_state)
            for k in range(2*num_x*num_y): # 应该要加2。
                #print(temp_state[popu_indice[k]])
                temp_population = np.sum(np.abs(temp_state[popu_indice[k]])**2) # IPR()
                if temp_population>thresh_popu:
                    record_popu_bigger[inter_dx,k] += 1
    
    np.savez('ThreeExcitation_CountState_interList_gamma%.2f_t%d_inter.npz'%(gamma, t_final), record_popu_bigger=record_popu_bigger, inter_list=inter_list)
    #for i in range(len(t_list)):
    # with np.load('CountState_interList_gamma0.20_t60_intra.npz') as files:
    #     record_popu_bigger = files['record_popu_bigger']
    #     inter_list = files['inter_list']
    # plt.plot(inter_list, record_popu_bigger[:,0], 'o-', label=1)
    # plt.plot(inter_list, record_popu_bigger[:,1], 'o-', label=1)
    # plt.plot(inter_list, record_popu_bigger[:,2*num_y-2], 'o-', label=2)
    # plt.plot(inter_list, record_popu_bigger[:,2*num_y-1], 'o-', label=2)
    # plt.plot(inter_list, record_popu_bigger[:,-(2*num_y)], 'o-', label=3)
    # plt.plot(inter_list, record_popu_bigger[:,-(2*num_y-1)], 'o-', label=3)
    # plt.plot(inter_list, record_popu_bigger[:,-2], 'o-', label=4)
    # plt.plot(inter_list, record_popu_bigger[:,-1], 'o-', label=4)

    print(len(record_popu_bigger[0]))
    for i in range(len(record_popu_bigger[0])):
        plt.plot(inter_list, record_popu_bigger[:,i], 'o-')
        # if record_popu_bigger[10,i]>500:
        #     print(i%(2*num_y),i)
    #plt.legend()
    fontsize = 14
    plt.xlabel(r'$U$', fontsize=fontsize)
    plt.ylabel(r'$\mathcal{P}_x$', fontsize=fontsize)
    plt.grid()
    # plt.show()  # 在耗散格点的增幅增大，
    # # 其实这里还可以统计分布，只在角落上的？通过图来看可以统计y=1的分布。

    import matplotlib.cm as cm

    # 假设 record_popu_bigger 是一个二维数组，inter_list 是 x 轴的数据
    num_curves = len(record_popu_bigger[0])

    # 创建颜色映射
    colors = cm.viridis(np.linspace(0, 1, num_curves))

    plt.figure(figsize=(12, 8))

    for i in range(num_curves):
        plt.plot(inter_list, record_popu_bigger[:, i], 
                marker='o', linestyle='-', 
                color=colors[i], 
                markersize=5, 
                linewidth=2,
                label=f'Curve {i+1}')

    plt.xlabel('X-axis Label', fontsize=12)
    plt.ylabel('Y-axis Label', fontsize=12)
    plt.title('Population Trends', fontsize=14, pad=20)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 调整图例位置
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
    #         fontsize=10, title='Curves', title_fontsize=12)

    plt.tight_layout()
    #plt.show()

    # plt.imshow(record_popu_bigger[-1].reshape(num_x, num_y*2).T, cmap='Reds')
    # plt.show()

    # # with np.load('CountState_interaction1.00.npz') as files:
    # #     record_popu_bigger = files['record_popu_bigger']
    # #     t_list = files['t_list']
    
    # # import seaborn as sns
    # # Set Seaborn style for better aesthetics
    # # sns.set_theme(
    # #     style="whitegrid", 
    # #     rc={"axes.facecolor": "#fcfaf5"}
    # # )

    # # Create figure and axis objects
    # fig, ax = plt.subplots(figsize=(10, 6))

    # # --------------------------------------------------
    # # Custom color palette (adjust colors to enhance visibility)
    # #colors = sns.color_palette("husl", n_colors=len(t_list))

    # # ------------------------------
    # # Plotting each line with distinct styling
    # for idx, time_point in enumerate(t_list):

    #     # Convert time point to integer for coloring
    #     time_val = int(time_point)
        
    #     # Compute color based on normalized time value
    #     norm_time = time_val / max(int(k) for k in t_list)
    #     color = plt.cm.Blues(norm_time)  

    #     # Line properties (width, alpha)
    #     linewidth = (0.25 + 0.75 * norm_time)  # Wider lines for later times
        
    #     ax.plot(
    #         record_popu_bigger[idx], 'o--',
    #         label=f't={time_point}',
    #         color=color,
    #         linewidth=2.5,
    #         alpha=0.85,
    #     ) #             z_order=10 - time_val,  # Ensure earlier times are plotted first zorder=20 - time_val

    # #-----------------------------------------------------
    # # Configure axes & labels
    # ax.set_xlabel("X Position", fontsize=13, labelpad=10)
    # ax.set_ylabel(r"$\mathcal{P}(x)$", fontsize=14) # \,\,\,(a.u.)
    # ax.set_title("Time Evolution of Distribution", pad=30, fontsize=16)

    # # Use scientific notation where needed
    # if any(y >= 1e4 for y in ax.get_yticks()):
    #     plt.ticklabel_format(style='sci', scilimits=(0,0), useMathText=True)

    # # Add legend with improved placement & formatting
    # legend = ax.legend(
    #     title=r"Time Steps ($\Delta t$) ",
    #     loc='upper left',
    #     frameon=False,
    #     bbox_to_anchor=(0.97, 0.82),
    #     borderpad=1.5
    # )#.get_frame().set_linewidth(0) # frame_color="#fcfaf5",

    # # Enhance grid visibility
    # ax.grid(True, which='both', linestyle=':', linewidth=0.6)

    # # Improve colorbar integration (example for other plots)
    # # plt.colorbar(...)  # If a colorbar is used...

    # # Save the figure with tight layout and high resolution
    # plt.tight_layout()
    # plt.show()


def CalPopu_hist_corner():
    # 随机入射看演化。
    num_x = 5
    num_y = 5
    print('---L particle:---')
    QWZ2 = ParticleConservedSystem(num_x, num_y, 2)
    QWZ2.add_onsite_term(-1.5)
    QWZ2.add_NH(0.2)
    QWZ2.add_xhopping_term()
    QWZ2.add_yhopping_term()
    QWZ2.add_interaction_inter(0.2)
    print('---Get Hamiltonian with dimension---', QWZ2.dimension)
    ham2 = QWZ2.hamiltonian
    popu_indice = QWZ2.Calcu_Occup()
    t_list = np.linspace(0, 20, 11) # 当为2的时候会先减少。
    thresh_popu = 2/num_x
    evo_ham = expm(-1j*ham2*t_list[1])
    NumSamp = 1000
    record_popu_bigger = np.zeros((len(t_list), QWZ2.total_mode))
    from collections import defaultdict
    temp_list = [x for x in range(0, 2*num_x*num_y,2*num_y)] + [x for x in range(1, 2*num_x*num_y,2*num_y)]
    record_hist_dict = {x:[] for x in temp_list}#defaultdict(list)

    for temp_samp in range(NumSamp):
        [print('samp', temp_samp) if temp_samp % 10 == 0 else None]
        initial_state = np.zeros(QWZ2.dimension, dtype=complex)
        initial_state[np.random.randint(QWZ2.dimension)] = 1.
        for tx in range(len(t_list)):
            temp_state = np.dot(evo_ham, initial_state)
            temp_state = temp_state / np.linalg.norm(temp_state)
            for k in range(0, 2*num_x*num_y, 2*num_y): # 应该要加2。
                #print(temp_state[popu_indice[k]])
                temp_population = np.sum(np.abs(temp_state[popu_indice[k]])**2) # IPR()
                record_hist_dict[k].append(temp_population)
            for k in range(1, 2*num_x*num_y, 2*num_y): # 应该要加2。
                #print(temp_state[popu_indice[k]])
                temp_population = np.sum(np.abs(temp_state[popu_indice[k]])**2) # IPR()
                record_hist_dict[k].append(temp_population)
            initial_state = temp_state
    # 保存数据和画图。

    # 这个可以画在一个图中，然后不同的数据进行不同的label吗？
    data_for_hist = []
    labels = []
    plt.figure(figsize=(10, 6))
    for key, values in record_hist_dict.items():
        if len(values) > 0:  # 确保该键有数据
            data_for_hist.append(values)
            labels.append(f"Key {key}")
            plt.hist(data_for_hist, bins=20, label=labels, alpha=0.6)

    # 添加标签和图例
    plt.xlabel("Population")
    plt.ylabel("Frequency")
    plt.title("Histogram of Population Distribution")
    plt.legend()

    # 显示图形
    plt.show()

def DifInter(gamma):
    num_x = 5
    num_y = 5
    print('---One particle:---')
    QWZ1 = ParticleConservedSystem(num_x, num_y, 1)
    QWZ1.add_onsite_term(-1.5)
    # QWZ1.add_NH(0.2)
    QWZ1.add_xhopping_term()
    QWZ1.add_yhopping_term()
    print('---Get Hamiltonian---', QWZ1.dimension)
    ham1 = QWZ1.hamiltonian
    #print(ham1)
    print('---Two particle:---')
    QWZ2 = ParticleConservedSystem(num_x, num_y, 2)
    QWZ2.add_onsite_term(-1.5)
    # QWZ2.add_NH(0.2)
    QWZ2.add_xhopping_term()
    QWZ2.add_yhopping_term()
    QWZ2.add_interaction_inter(gamma)
    print('---Get Hamiltonian---', QWZ2.dimension)
    ham2 = QWZ2.hamiltonian
    #print(ham2)

    eigval1, eigvec1 = np.linalg.eig(ham1) 
    eigvec1 = eigvec1[:, np.argsort(np.abs(eigval1.real))]
    eigval1 = eigval1[np.argsort(np.abs(eigval1.real))]
    popu1 = QWZ1.Calcu_Occup(eigvec1[:,0])
    print('population:', popu1[::2]+popu1[1::2])
    eigval2, eigvec2 = np.linalg.eig(ham2)
    #print(eigval1, eigval2)
    new_eigval = []
    for i in range(len(eigval1)-1):
        for j in range(i+1, len(eigval1)):
            new_eigval.append(eigval1[i] + eigval1[j] )
    new_eigval =np.array(sorted(new_eigval))

    absolute_diff = np.abs(eigval2 - (eigval1[0]+eigval1[1]))
    popu2 = QWZ2.Calcu_Occup(eigvec2[:, np.argmin(absolute_diff)])
    return IPR(popu2)


def SuppleMater_Spectrum():
    """
    Fig2和Fig3中的结果
    """
    u = -1.5
    Lx,Ly = 7,7
    QWZ2 = ParticleConservedSystem(Lx, Ly, 2)
    QWZ2.add_onsite_term(u)
    QWZ2.add_NH(0.2)
    QWZ2.add_xhopping_term()
    QWZ2.add_yhopping_term()
    QWZ2.add_interaction_inter(1.)
    popu_indice = QWZ2.Calcu_Occup()
    hamiltonian = QWZ2.hamiltonian

    eigval2, eigvec2 = np.linalg.eig(hamiltonian)
    eigvec2 = eigvec2[:, np.argsort(np.imag(eigval2))]
    eigval2 = eigval2[np.argsort(np.imag(eigval2))]
    import h5py
    popu_eig = np.zeros(QWZ2.total_mode)
    for k in range(2*Lx*Ly): # 应该要加2。
        #print(temp_state[popu_indice[k]])
        popu_eig[k] = np.sum(np.abs(eigvec2[popu_indice[k],-1])**2)

    file = h5py.File('TwoExcitationEig%d.h5'%(Lx), 'w')
    file.create_dataset('eigval_NHMany', data=eigval2)
    file.create_dataset('eigvec_NHMany', data=eigvec2)
    file.create_dataset('popu_NHMany', data=popu_eig)
    file.close()

    fig, ax = plt.subplots(1, 3)
    ax[0].scatter(eigval2.real, eigval2.imag)
    ax[1].plot(np.abs(eigvec2[:,-1])**2)
    # ax[1].plot(np.abs(eigvec2[:,-1])**2)
    plt.show()
    

def LargerSystem():
    L_list = [10, 12, 14, 16, 18]
    u = -2.1
    import h5py
    file = h5py.File('TwoExcitationLargerSpectrum_NotTopo_u%.1f.h5'%u, 'w')
    for idx, temp_l in enumerate(L_list):
        print('System Size', temp_l)
        num_x = temp_l # 19600 for 5 and 4960 for 4. 3h for 5.
        num_y = temp_l
        QWZ2 = ParticleConservedSystem_Coo(num_x, num_y, 2)
        QWZ2.add_onsite_term(u)
        QWZ2.add_NH(0.2)
        QWZ2.add_xhopping_term()
        QWZ2.add_yhopping_term()
        QWZ2.add_interaction_inter(1.)
        popu_indice = QWZ2.Calcu_Occup()
        import scipy
        hamiltonian_sparse = scipy.sparse.coo_matrix((QWZ2.data, (QWZ2.row, QWZ2.col)), shape=(QWZ2.dimension, QWZ2.dimension))
        print('Processing interaction:')
        # eigval2, eigvec2 = np.linalg.eig(hamiltonian)  # 大概40min
        import scipy
        # hamiltonian_sparse = scipy.sparse.coo_matrix(hamiltonian)
        eigval2, eigvec2 = scipy.sparse.linalg.eigs(hamiltonian_sparse,k=3, which='LI')
        eigvec2 = eigvec2[:, np.argsort(np.imag(eigval2))]
        eigval2 = eigval2[np.argsort(np.imag(eigval2))]
        popu_eig2 = np.zeros(QWZ2.total_mode)
        for k in range(2*num_x*num_y): # 应该要加2。
            #print(temp_state[popu_indice[k]])
            popu_eig2[k] = np.sum(np.abs(eigvec2[popu_indice[k],-1])**2)
        
        file.create_dataset('eigval_NHMany%d'%(temp_l), data=eigval2)
        file.create_dataset('eigvec_NHMany%d'%(temp_l), data=eigvec2)
    file.close()

    # import h5py
    # L_list = [10, 12, 14, 16, 18]
    # fig, ax = plt.subplots(1, 2)
    # with h5py.File('TwoExcitationLargerSpectrum.h5', 'r') as files:
    #     for key in files.keys():
    #         print(key, files[key].name)
    #     for i in range(len(L_list)):
    #         eigval = files['eigval_NHMany%d'%(L_list[i])][:]
    #         ax[0].scatter(eigval.real, eigval.imag)
    #         ax[1].plot(np.abs(files['eigvec_NHMany%d'%(L_list[i])][:][:,-1])**2)   # eigenstate
    # plt.show()
    import h5py
    # L_list = [10, 12, 14, 16, 18]
    fig, ax = plt.subplots(1, 2)
    with h5py.File('ThreeExcitationLargerSpectrum.h5', 'r') as files:
        for key in files.keys():
            print(key, files[key].name)
        for i in range(len(L_list)):
            eigval = files['eigval_NHMany%d'%(L_list[i])][:]
            ax[0].scatter(eigval.real, eigval.imag)
            ax[1].plot(np.abs(files['eigvec_NHMany%d'%(L_list[i])][:][:,-1])**2)   # eigenstate
    plt.show()
    # np.savez('ThreeExcitationEig%d.npz'%num_x, eigval_NHMany=eigval2,eigvec_NHMany=eigvec2)

    # fig, ax = plt.subplots(1,2,figsize=(8, 4),sharex=True, sharey=True)
    # # ax[0].scatter(eigval.real, eigval.imag)
    # # ax[0].scatter(eigval[-1].real, eigval[-1].imag, color='r', s=100, marker='x')
    # ax[1].scatter(eigval2.real, eigval2.imag)
    # ax[1].scatter(eigval2[-1].real, eigval2[-1].imag, color='r', s=100, marker='x')
    # ax[0].set_xlabel('Re(E)')
    # ax[1].set_xlabel('Re(E)')
    # ax[0].set_ylabel('Im(E)')
    # # plt.show()

    # fig, ax = plt.subplots(1,2,figsize=(8, 4))
    # # ax[0].plot(np.abs(eigvec[:,-1])**2)
    # ax[1].plot(np.abs(eigvec2[:,-1])**2)
    # ax[0].set_xlabel('Number')
    # ax[1].set_xlabel('Number')
    # ax[0].set_ylabel(r'$\vert \phi_i(x,y)\vert^2$')
    # # plt.show()

    # fig, ax = plt.subplots(1,2,figsize=(8, 4))
    # # ax[0].imshow(popu_eig.reshape(num_x, 2*num_y).T, cmap='Reds')
    # ax[1].imshow(popu_eig2.reshape(num_x, 2*num_y).T, cmap='Reds')
    # ax[0].set_xlabel('x')
    # ax[1].set_xlabel('x')
    # ax[0].set_ylabel('y(A,B)')
    # # plt.show()
    # 176min

    print('Total time:', (time.time()-start_time)/60)


if __name__ == "__main__":
    start_time = time.time()
    # Comp()
    # Comp2()
    # FockSkin_InterNonHermitian()
    # FockSkin()
    # CalPopu()
    #CalPopu_t(1)
    #CalPopu_t(-0.2)
    # CalPopu_t(-1.)
    # CalPopu_inter_non(0., 20) # 尺寸是8*8
    # CalPopu_inter_non(0.2, 20) # 尺寸是8*8
    # CalPopu_inter_non(0., 60) # 尺寸是8*8
    # CalPopu_inter_non(0.2, 60) # 尺寸是8*8
    # plt.show()
    # CalPopu_inter_non(0.2, 20)
    # CalPopu_inter_non(0.0, 60) # 尺寸是8*8
    # CalPopu_inter_non(0.2, 60)

    # CalPopu_inter_non(1.)
    #CalPopu_t(-1.) # sample应该增加为更多,plt.show也应该取消。
    # CalPopu_hist_corner()

    # gamma_list = np.linspace(-0.2,0.2,41)
    # IPR_list = np.zeros(len(gamma_list))
    # for i in range(len(gamma_list)):
    #     IPR_list[i] = DifInter(gamma_list[i])

    # plt.plot(gamma_list, IPR_list, 'o-')
    # plt.xlabel('U')
    # plt.ylabel('IPR')
    # plt.show()

    # num_x = 7
    # num_y = 7

    # QWZ1 = ParticleConservedSystem(num_x, num_y, 1)
    # QWZ1.add_onsite_term(-1.5)
    # QWZ1.add_NH(0.2)
    # QWZ1.add_xhopping_term()
    # QWZ1.add_yhopping_term()
    # ham1 = QWZ1.hamiltonian
    # eigval1, eigvec1 = np.linalg.eig(ham1)
    # eigval1 = eigval1[np.argsort(np.abs(eigval1.real))]
    

    # QWZ2 = ParticleConservedSystem(num_x, num_y, 2)
    # QWZ2.add_onsite_term(-1.5)
    # QWZ2.add_NH(0.2)
    # QWZ2.add_xhopping_term()
    # QWZ2.add_yhopping_term()
    # # QWZ2.add_interaction_inter(-0.2)
    # print('---Get Hamiltonian---', QWZ2.dimension)
    # ham2 = QWZ2.hamiltonian
    # eigval2, eigvec2 = np.linalg.eig(ham2)
    # absolute_diff = np.abs(eigval2 - (np.sum(eigval1[:2])))
    # temp_ipr_list = np.zeros(len(eigval2))
    # for j in range(len(eigval2)):
    #     temp_ipr_list[j] = IPR(eigvec2[:, j])
    # # index_max = np.argsort(np.max(temp_ipr_list))
    # index_max = np.argmin(absolute_diff)
    # print('max ipr', np.max(temp_ipr_list), temp_ipr_list[index_max])
    # record_ipr = temp_ipr_list[index_max]
    # popu = QWZ2.Calcu_Occup(eigvec2[:, index_max])
    # print('IPR of the population:', IPR(popu, 'occ'))
    # popu = popu[::2]+popu[1::2]
    # popu = popu.reshape((num_x,num_y)).T
    # plt.imshow(popu, 'Reds')
    # plt.show()

    # plt.plot(np.abs(eigvec2[:,index_max]))
    # plt.plot(np.abs(eigvec2[:, np.argmax(temp_ipr_list)]))
    # plt.show()

    # temp_ipr_list = sorted(temp_ipr_list)
    # plt.plot(temp_ipr_list)
    # plt.hlines(record_ipr,xmin=0,xmax=len(eigval2))
    # plt.show()

    # LargerSystem()
    SuppleMater_Spectrum()


    # # sparse_density matrix. 47min for 7*7 .  194min for 8*8. 
    # # 对于厄米的非厄米的能算到17左右是没问题的。
    # num_x = 8 # 19600 for 5 and 4960 for 4. 3h for 5.
    # num_y = 8
    # QWZ2 = ParticleConservedSystem_Coo(num_x, num_y, 3)
    # QWZ2.add_onsite_term(-1.5)
    # QWZ2.add_NH(0.2)
    # QWZ2.add_xhopping_term()
    # QWZ2.add_yhopping_term()
    # QWZ2.add_interaction_inter(1.)
    # popu_indice = QWZ2.Calcu_Occup()
    # import scipy
    # hamiltonian_sparse = scipy.sparse.coo_matrix((QWZ2.data, (QWZ2.row, QWZ2.col)), shape=(QWZ2.dimension, QWZ2.dimension))
    # print('Processing interaction:')
    # # eigval2, eigvec2 = np.linalg.eig(hamiltonian)  # 大概40min
    # import scipy
    # # hamiltonian_sparse = scipy.sparse.coo_matrix(hamiltonian)
    # eigval2, eigvec2 = scipy.sparse.linalg.eigs(hamiltonian_sparse,k=3, which='LI')
    # eigvec2 = eigvec2[:, np.argsort(np.imag(eigval2))]
    # eigval2 = eigval2[np.argsort(np.imag(eigval2))]
    # popu_eig2 = np.zeros(QWZ2.total_mode)
    # for k in range(2*num_x*num_y): # 应该要加2。
    #     #print(temp_state[popu_indice[k]])
    #     popu_eig2[k] = np.sum(np.abs(eigvec2[popu_indice[k],-1])**2) 

    # np.savez('ThreeExcitationEig%d.npz'%num_x, eigval_NHMany=eigval2,eigvec_NHMany=eigvec2)

    # fig, ax = plt.subplots(1,2,figsize=(8, 4),sharex=True, sharey=True)
    # # ax[0].scatter(eigval.real, eigval.imag)
    # # ax[0].scatter(eigval[-1].real, eigval[-1].imag, color='r', s=100, marker='x')
    # ax[1].scatter(eigval2.real, eigval2.imag)
    # ax[1].scatter(eigval2[-1].real, eigval2[-1].imag, color='r', s=100, marker='x')
    # ax[0].set_xlabel('Re(E)')
    # ax[1].set_xlabel('Re(E)')
    # ax[0].set_ylabel('Im(E)')
    # # plt.show()

    # fig, ax = plt.subplots(1,2,figsize=(8, 4))
    # # ax[0].plot(np.abs(eigvec[:,-1])**2)
    # ax[1].plot(np.abs(eigvec2[:,-1])**2)
    # ax[0].set_xlabel('Number')
    # ax[1].set_xlabel('Number')
    # ax[0].set_ylabel(r'$\vert \phi_i(x,y)\vert^2$')
    # # plt.show()

    # fig, ax = plt.subplots(1,2,figsize=(8, 4))
    # # ax[0].imshow(popu_eig.reshape(num_x, 2*num_y).T, cmap='Reds')
    # ax[1].imshow(popu_eig2.reshape(num_x, 2*num_y).T, cmap='Reds')
    # ax[0].set_xlabel('x')
    # ax[1].set_xlabel('x')
    # ax[0].set_ylabel('y(A,B)')
    # # plt.show()
    # # 176min

    # print('Total time:', (time.time()-start_time)/60)
    # plt.show()
