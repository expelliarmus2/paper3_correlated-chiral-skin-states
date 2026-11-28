
import numpy as np
from scipy.linalg import eigh
from ComputationalPhysics.Package.BitOperation import ReadBit, FlipBit, PopCntBit, JointBit, PickBit
from itertools import combinations
import matplotlib.pyplot as plt
from scipy.linalg import expm

from tqdm import tqdm 

sigma_0 = np.array([[1., 0.], [0., 1.]], dtype='complex')
sigma_x = np.array([[0., 1.], [1., 0.]], dtype='complex')
sigma_y = np.array([[0., -1j], [1j, 0.]], dtype='complex')
sigma_z = np.array([[1., 0.], [0., -1.]], dtype='complex')
pseudo_sigma0 = np.tensordot(sigma_0, sigma_0, 0).transpose((0,2,1,3)).reshape(4,4)
pseudo_sigma1 = np.tensordot(sigma_x, sigma_x, 0).transpose((0,2,1,3)).reshape(4,4)
pseudo_sigma2 = np.tensordot(sigma_y, sigma_x, 0).transpose((0,2,1,3)).reshape(4,4)
pseudo_sigma3 = np.tensordot(sigma_z, sigma_x, 0).transpose((0,2,1,3)).reshape(4,4)
pseudo_sigma4 = np.tensordot(sigma_0, sigma_y, 0).transpose((0,2,1,3)).reshape(4,4)
# print(pseudo_sigma1, pseudo_sigma2, pseudo_sigma3, pseudo_sigma4)

def RealCSTI_ham(gamma, num_xsite, num_ysite, num_zsite, position=None, boundary='open', which_ham='non-hermitian'):
    # 计数的次序是(0,0,z),(0,1,z),...(1,0,z)
    ham = np.zeros((4*num_xsite*num_ysite*num_zsite, 4*num_xsite*num_ysite*num_zsite), dtype='complex')
    for i in range(num_xsite):
        for j in range(num_ysite):
            for k in range(num_zsite):
                row = i*4*num_ysite*num_zsite+j*4*num_zsite+k*4
                ham[row:row+4, row:row+4] = pseudo_sigma1 - 2*pseudo_sigma4
    ham_non = np.zeros((4 * num_xsite * num_ysite * num_zsite, 4 * num_xsite * num_ysite * num_zsite), dtype='complex')
    # 这里所有的i是不是应该在分子上面而不是分母上面，反正都是sin变换而来。
    for i in range(num_xsite):
        for j in range(num_ysite):
            for k in range(num_zsite):
                row = i*4*num_ysite*num_zsite+j*4*num_zsite+k*4
                if k!= num_zsite-1:
                    #col = i*4*num_ysite*num_zsite+j*4*num_zsite+(k+1)*4
                    ham_non[row+4:row+8,row:row+4] += pseudo_sigma3/(-2j) + pseudo_sigma4/2
                if j!= num_ysite-1:
                    ham_non[row+4*num_zsite:row+4*num_zsite+4, row:row+4] += \
                        -pseudo_sigma1/2 + pseudo_sigma2/(-2j) + pseudo_sigma4/2
                if i!= num_xsite-1:
                    ham_non[row+4*num_zsite*num_ysite:row+4*num_zsite*num_ysite+4,row:row+4] += \
                        pseudo_sigma1/(-2j) + pseudo_sigma4/2
    # (-2j) should be (2j)
    ham += ham_non + np.conj(ham_non.T)
    # add non-hermitian interaction

    ham_non = np.zeros((4 * num_xsite * num_ysite * num_zsite, 4 * num_xsite * num_ysite * num_zsite), dtype='complex')
    if boundary=='close_x':
        # close x boundary:
        for j in range(num_ysite):
            for k in range(num_zsite):
                col = j * 4 * num_zsite + k * 4
                site_constant = 4*(num_xsite-1)*num_ysite*num_zsite
                ham_non[col+site_constant:col+site_constant+4, col:col+4] += pseudo_sigma1/(2j) + pseudo_sigma4/2
        ham += ham_non + np.conj(ham_non.T)
    if boundary=='close_y':
        for i in range(num_xsite):
            for k in range(num_zsite):
                col = i * 4 * num_ysite * num_zsite + k * 4
                site_constant = 4*(num_ysite-1)*num_zsite
                ham_non[col + site_constant:col + site_constant + 4, col:col + 4] += \
                    -pseudo_sigma1/2 + pseudo_sigma2/(2j) + pseudo_sigma4/2
        ham += ham_non + np.conj(ham_non.T)
    if boundary=='close_xy':
        # close x boundary:
        for j in range(num_ysite):
            for k in range(num_zsite):
                col = j * 4 * num_zsite + k * 4
                site_constant = 4*(num_xsite-1)*num_ysite*num_zsite
                ham_non[col+site_constant:col+site_constant+4, col:col+4] += pseudo_sigma1/(2j) + pseudo_sigma4/2
        # close y
        for i in range(num_xsite):
            for k in range(num_zsite):
                col = i * 4 * num_ysite * num_zsite + k * 4
                site_constant = 4*(num_ysite-1)*num_zsite
                ham_non[col + site_constant:col + site_constant + 4, col:col + 4] += \
                    -pseudo_sigma1/2 + pseudo_sigma2/(2j) + pseudo_sigma4/2
        ham += ham_non + np.conj(ham_non.T)

    # add non-hermitian
    if which_ham=='non-hermitian':
        if position!=None:
            for site in range(len(position)):
                row = 4*position[site]
                ham[row:row+4, row:row+4] += 1j*gamma*pseudo_sigma0
        else:
            # apply dissipation on the boundary.
            for row in range(0, 4*num_xsite*num_ysite*num_zsite, 4*num_zsite):
                ham[row:row+4, row:row+4] += 1j*gamma*pseudo_sigma0
    return ham


class ParticleConservedSystem_CSTI:
    def __init__(self, num_xmodes: int, num_ymodes: int, num_zmodes: int, total_particles: int):
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
        self.num_z = num_zmodes
        self.total_particles = total_particles
        self.sublat = 4 # 一个元胞中包含A，B两个原子。
        self.total_mode = self.sublat*self.num_x*self.num_y*self.num_z
        
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

    def add_onsite_term(self):
        """添加局域能量项: 并不是onsite的，而是自己这个元胞到自己这个元胞"""
        #u = -1.5
        for idx, state in enumerate(self.basis): #第几个basis，矩阵的第几个。
            #print(idx, state, bin(state), ReadBit(state,0), ReadBit(state,1))
            #self._compute_fermion_sign(state, i, i+distance1)

            # print(np.sum([ReadBit(state, i) for i in range(0, 2*self.num_x*self.num_y, 2)]))
            for i in range(0, self.total_mode, 4): # from A sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 1 # mx A->mx+1，A
                    if ReadBit(state, i+distance1)==0:
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        self.hamiltonian[jdx1, idx] += -2j
                        self.hamiltonian[idx, jdx1] += 2j # to A lattice
                    distance2 = 3
                    if ReadBit(state, i+distance2)==0:
                        state1 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        self.hamiltonian[jdx2, idx] += 1 * sign
                        self.hamiltonian[idx, jdx2] += 1 * sign
            for i in range(1, self.total_mode, 4): # from B sublattice.
                distance1 = 1
                if ReadBit(state, i)==1 and ReadBit(state, i+distance1)==0:
                    state1 = FlipBit(FlipBit(state, i), i+distance1)
                    jdx1 = self.state_index[state1]
                    self.hamiltonian[jdx1, idx] += 1
                    self.hamiltonian[idx, jdx1] += 1
            for i in range(2, self.total_mode, 4): # from C sublattice.
                distance1 = 1
                if ReadBit(state, i)==1 and ReadBit(state, i+distance1)==0:
                    state1 = FlipBit(FlipBit(state, i), i+distance1)
                    jdx1 = self.state_index[state1]
                    self.hamiltonian[jdx1, idx] += -2j
                    self.hamiltonian[idx, jdx1] += 2j
                

    def add_NH(self, gamma: float, which_dissi:int=0):
        """在z=1的边界添加非厄米项: i * gamma * a†_m a_m
        which dissi:表示在y等于几加耗散。默认在y=1的边界
        """
        for idx, state in enumerate(self.basis): #第几个basis，矩阵的第几个。
            #print(idx, state, bin(state), ReadBit(state,0), ReadBit(state,1))
            # print(idx, bin(state),np.sum([ReadBit(state, i) for i in range(4*which_dissi+2, self.total_mode, 4*self.num_z)]) )
            self.hamiltonian[idx, idx] += 1j * gamma*np.sum([ReadBit(state, i) for i in range(4*which_dissi, self.total_mode, 4*self.num_z)]) 
            self.hamiltonian[idx, idx] +=  1j * gamma * np.sum([ReadBit(state, i) for i in range(4*which_dissi+1, self.total_mode, 4*self.num_z)]) 
            self.hamiltonian[idx, idx] += 1j * gamma * np.sum([ReadBit(state, i) for i in range(4*which_dissi+2, self.total_mode, 4*self.num_z)]) 
            self.hamiltonian[idx, idx] += 1j * gamma * np.sum([ReadBit(state, i) for i in range(4*which_dissi+3, self.total_mode, 4*self.num_z)])
            # 不嫩换行，不然就不对
    
    def add_NH_SingleSite(self, gamma: float, which_dissi:list=[]):
        """在y=1的边界添加非厄米项: i * gamma * a†_m a_m
        which dissi:表示在y等于几加耗散。默认在y=1的边界
        """
        for idx, state in enumerate(self.basis): #第几个basis，矩阵的第几个。
            #print(idx, state, bin(state), ReadBit(state,0), ReadBit(state,1))
            self.hamiltonian[idx, idx] += 1j * gamma*np.sum([ReadBit(state, 4*i) for i in which_dissi]) 
            + 1j * gamma * np.sum([ReadBit(state, 4*i+1) for i in which_dissi])
            + 1j * gamma * np.sum([ReadBit(state, 4*i+2) for i in which_dissi])
            + 1j * gamma * np.sum([ReadBit(state, 4*i+3) for i in which_dissi])

    def add_xhopping_term(self):
        """添加跃迁项: strength * a†_source a_target + h.c."""
        for idx, state in enumerate(self.basis):
            # from A particle
            for i in range(0, self.total_mode, 4): # A sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 4*self.num_y*self.num_z+1 # mx A->mx+1，B
                    if ReadBit(state, i+distance1)==0 and i+distance1<self.total_mode:
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        self.hamiltonian[jdx1, idx] += 1j/2 * sign
                        self.hamiltonian[idx, jdx1] += -1j/2 * sign
                    distance2 = 4*self.num_y*self.num_z + 3 # mx A->mx+1，D
                    if ReadBit(state, i+distance2)==0 and i+distance2<self.total_mode:
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        self.hamiltonian[jdx2, idx] += 1j/2 * sign
                        self.hamiltonian[idx, jdx2] += -1j/2 * sign

            # from B particle
            for i in range(1, self.total_mode, 4): # B sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 4*self.num_y*self.num_z-1 # mx B->mx+1，A
                    if ReadBit(state, i+distance1)==0 and i+distance1<self.total_mode:
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        self.hamiltonian[jdx1, idx] += -1j/2 * sign
                        self.hamiltonian[idx, jdx1] += 1j/2 * sign
                    distance2 = 4*self.num_y*self.num_z + 1 # mx B->mx+1，C
                    if ReadBit(state, i+distance2)==0 and i+distance2<self.total_mode:
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        self.hamiltonian[jdx2, idx] += 1j/2 * sign
                        self.hamiltonian[idx, jdx2] += -1j/2 * sign
            
            # from C particle
            for i in range(2, self.total_mode, 4): # C sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 4*self.num_y*self.num_z-1 # mx C->mx+1，B
                    if ReadBit(state, i+distance1)==0 and i+distance1<self.total_mode:
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        self.hamiltonian[jdx1, idx] += 1j/2 * sign
                        self.hamiltonian[idx, jdx1] += -1j/2 * sign
                    distance2 = 4*self.num_y*self.num_z + 1 # mx C->mx+1，D
                    if ReadBit(state, i+distance2)==0 and i+distance2<self.total_mode:
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        self.hamiltonian[jdx2, idx] += 1j/2 * sign
                        self.hamiltonian[idx, jdx2] += -1j/2 * sign
            
            # from D particle
            for i in range(3, self.total_mode, 4): # C sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 4*self.num_y*self.num_z-3 # mx D->mx+1，A
                    if ReadBit(state, i+distance1)==0 and i+distance1<self.total_mode:
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        self.hamiltonian[jdx1, idx] += 1j/2 * sign
                        self.hamiltonian[idx, jdx1] += -1j/2 * sign
                    distance2 = 4*self.num_y*self.num_z - 1 # mx D->mx+1，C
                    if ReadBit(state, i+distance2)==0 and i+distance2<self.total_mode:
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        self.hamiltonian[jdx2, idx] += -1j/2 * sign
                        self.hamiltonian[idx, jdx2] += 1j/2 * sign

    def add_yhopping_term(self):
        """添加跃迁项: strength * a†_source a_target + h.c."""
        for idx, state in enumerate(self.basis):
            # from A particle
            for i in range(0, self.total_mode, 4): # A sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 4*self.num_z+1 # mx A->mx+1，B
                    if ReadBit(state, i+distance1)==0 and i%(4*self.num_y*self.num_z)+4*self.num_z<4*self.num_y*self.num_z:
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        self.hamiltonian[jdx1, idx] += 1j/2 * sign
                        self.hamiltonian[idx, jdx1] += -1j/2 * sign
                    distance2 = 4*self.num_z + 3 # mx A->mx+1，D
                    if ReadBit(state, i+distance2)==0 and i%(4*self.num_y*self.num_z)+4*self.num_z<4*self.num_y*self.num_z:
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        self.hamiltonian[jdx2, idx] += -1 * sign
                        self.hamiltonian[idx, jdx2] += -1 * sign

            # from B particle
            for i in range(1, self.total_mode, 4): # B sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 4*self.num_z-1 # mx B->mx+1，A
                    if ReadBit(state, i+distance1)==0 and i%(4*self.num_y*self.num_z)+4*self.num_z<4*self.num_y*self.num_z:
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        self.hamiltonian[jdx1, idx] += -1j/2 * sign
                        self.hamiltonian[idx, jdx1] += 1j/2 * sign
                    distance2 = 4*self.num_z + 1 # mx B->mx+1，C
                    if ReadBit(state, i+distance2)==0 and i%(4*self.num_y*self.num_z)+4*self.num_z<4*self.num_y*self.num_z:
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        self.hamiltonian[jdx2, idx] += -1 * sign
                        self.hamiltonian[idx, jdx2] += -1 * sign
            
            # from C particle
            for i in range(2, self.total_mode, 4): # C sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance2 = 4*self.num_z + 1 # mx C->mx+1，D
                    if ReadBit(state, i+distance2)==0 and i%(4*self.num_y*self.num_z)+4*self.num_z<4*self.num_y*self.num_z:
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        self.hamiltonian[jdx2, idx] += 1j/2 * sign
                        self.hamiltonian[idx, jdx2] += -1j/2 * sign
            
            # from D particle
            for i in range(3, self.total_mode, 4): # C sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance2 = 4*self.num_z - 1 # mx D->mx+1，C
                    if ReadBit(state, i+distance2)==0 and i%(4*self.num_y*self.num_z)+4*self.num_z<4*self.num_y*self.num_z:
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        self.hamiltonian[jdx2, idx] += -1j/2 * sign
                        self.hamiltonian[idx, jdx2] += 1j/2 * sign

    def add_zhopping_term(self):
        """添加跃迁项: strength * a†_source a_target + h.c."""
        for idx, state in enumerate(self.basis):
            # from A particle
            for i in range(0, self.total_mode, 4): # A sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 4 + 1 # mz A->mx+1，B
                    if ReadBit(state, i+distance1)==0 and i not in list(range(4*self.num_z-4, self.total_mode, 4*self.num_z)):
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        self.hamiltonian[jdx1, idx] += 1j * sign
                        self.hamiltonian[idx, jdx1] += -1j * sign
            for i in range(3, self.total_mode, 4): # A sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance2 = 4 - 1 # mz D->mz+1，C
                    if ReadBit(state, i+distance2)==0 and i-3 not in list(range(4*self.num_z-4, self.total_mode, 4*self.num_z)):
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        self.hamiltonian[jdx2, idx] += -1j * sign
                        self.hamiltonian[idx, jdx2] += 1j * sign

    # def add_interaction_intra(self, U):
    #     """添加相互作用U n_{x,y, A} n_{x,y, B}，当一个元胞内有两个粒子能量变大，趋向于不呆在或者呆在一个元胞内。或者是相互之间的，这样chiral skin会让它趋向于局域，但是相互作用可能让它更局域或者弥散。"""
    #     self.inter = np.zeros_like(self.hamiltonian)
    #     for idx, state in enumerate(self.basis):
    #         # # U n_{x,y, A} n_{x,y, B}
    #         for i in range(0, self.total_mode, 2): # A sublattice.
    #             if ReadBit(state, i)==1 and ReadBit(state, i+1)==1:
    #                 self.inter[idx, idx] += U
    #     self.hamiltonian += self.inter
    #     return self.inter

    def add_interaction_inter(self, U):
        """添加相互作用U( n_{x,y, A}+n_{x,y, B}) * (n_{x+1,y, A}+n_{x+1,y, B}))，当近邻元胞有不同的电子数时，会出现相互作用项。这样chiral skin会让它趋向于局域，但是相互作用可能让它更局域或者弥散。但是现象好像是相互作用会使它更局域"""
        self.inter = np.zeros_like(self.hamiltonian)
        for idx, state in enumerate(self.basis):
            # X: (U n_{x,y, A}+n_{x,y, B}) * (U n_{x+1,y, A}+n_{x+1,y, B}). Y同样。
            distancey = 4*self.num_z
            distancex = 4*self.num_z*self.num_y
            for i in range(0, self.total_mode, 4): # A sublattice.
                # Z direction
                if i not in list(range(4*self.num_z-4, self.total_mode, 4*self.num_z)):
                    self.inter[idx, idx] += U*(ReadBit(state, i)+ReadBit(state,i+1)+ReadBit(state, i+2)+ReadBit(state,i+3))*(ReadBit(state, i+4)+ReadBit(state,i+5)+ReadBit(state, i+6)+ReadBit(state,i+7))
                # # y direction
                if i%(distancex)+distancey<distancex:
                    self.inter[idx, idx] += U*(ReadBit(state, i)+ReadBit(state,i+1)+ReadBit(state, i+2)+ReadBit(state,i+3))*(ReadBit(state, i+distancey)+ReadBit(state,i+distancey+1)+ReadBit(state, i+distancey+2)+ReadBit(state,i+distancey+3))
                # X direction
                if i+distancex<self.total_mode:
                    self.inter[idx, idx] += U*(ReadBit(state, i)+ReadBit(state,i+1)+ReadBit(state, i+2)+ReadBit(state,i+3))*(ReadBit(state, i+distancex)+ReadBit(state,i+distancex+1)+ReadBit(state, i+distancex+2)+ReadBit(state,i+distancex+3))
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

class ParticleConservedSystem_CSTI_Coo:
    def __init__(self, num_xmodes: int, num_ymodes: int, num_zmodes: int, total_particles: int):
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
        self.num_z = num_zmodes
        self.total_particles = total_particles
        self.sublat = 4 # 一个元胞中包含A，B两个原子。
        self.total_mode = self.sublat*self.num_x*self.num_y*self.num_z
        
        # 生成所有可能的量子态基矢
        self.basis = self._generate_basis()
        self.dimension = len(self.basis) # C_N^{n}
        #print(type(self.basis), self.basis)
        
        # 建立态向量到矩阵索引的映射
        self.state_index = {state: idx for idx, state in enumerate(self.basis)}  # 0: 00001111, 1:00010111 ....
        # print(self.state_index)
        # 初始化哈密顿矩阵
        # self.hamiltonian = np.zeros((self.dimension, self.dimension), dtype=np.complex128)
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

    def add_onsite_term(self):
        """添加局域能量项: 并不是onsite的，而是自己这个元胞到自己这个元胞"""
        #u = -1.5
        for idx, state in enumerate(self.basis): #第几个basis，矩阵的第几个。
            #print(idx, state, bin(state), ReadBit(state,0), ReadBit(state,1))
            #self._compute_fermion_sign(state, i, i+distance1)

            # print(np.sum([ReadBit(state, i) for i in range(0, 2*self.num_x*self.num_y, 2)]))
            for i in range(0, self.total_mode, 4): # from A sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 1 # mx A->mx+1，A
                    if ReadBit(state, i+distance1)==0:
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        # self.hamiltonian[jdx1, idx] += -2j
                        # self.hamiltonian[idx, jdx1] += 2j # to A lattice
                        self.row.append(jdx1)
                        self.col.append(idx)
                        self.data.append(-2j)
                        self.row.append(idx)
                        self.col.append(jdx1)
                        self.data.append(2j)
                    distance2 = 3
                    if ReadBit(state, i+distance2)==0:
                        state1 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        # self.hamiltonian[jdx2, idx] += 1 * sign
                        # self.hamiltonian[idx, jdx2] += 1 * sign
                        self.row.append(jdx2)
                        self.col.append(idx)
                        self.data.append(1 * sign)
                        self.row.append(idx)
                        self.col.append(jdx2)
                        self.data.append(1 * sign)
            for i in range(1, self.total_mode, 4): # from B sublattice.
                distance1 = 1
                if ReadBit(state, i)==1 and ReadBit(state, i+distance1)==0:
                    state1 = FlipBit(FlipBit(state, i), i+distance1)
                    jdx1 = self.state_index[state1]
                    # self.hamiltonian[jdx1, idx] += 1
                    # self.hamiltonian[idx, jdx1] += 1
                    self.row.append(jdx1)
                    self.col.append(idx)
                    self.data.append(-1)
                    self.row.append(idx)
                    self.col.append(jdx1)
                    self.data.append(1)
            for i in range(2, self.total_mode, 4): # from C sublattice.
                distance1 = 1
                if ReadBit(state, i)==1 and ReadBit(state, i+distance1)==0:
                    state1 = FlipBit(FlipBit(state, i), i+distance1)
                    jdx1 = self.state_index[state1]
                    # self.hamiltonian[jdx1, idx] += -2j
                    # self.hamiltonian[idx, jdx1] += 2j
                    self.row.append(jdx1)
                    self.col.append(idx)
                    self.data.append(-2j)
                    self.row.append(idx)
                    self.col.append(jdx1)
                    self.data.append(2j)
                

    def add_NH(self, gamma: float, which_dissi:int=0):
        """在z=1的边界添加非厄米项: i * gamma * a†_m a_m
        which dissi:表示在y等于几加耗散。默认在y=1的边界
        """
        for idx, state in enumerate(self.basis): #第几个basis，矩阵的第几个。
            #print(idx, state, bin(state), ReadBit(state,0), ReadBit(state,1))
            # print(idx, bin(state),np.sum([ReadBit(state, i) for i in range(4*which_dissi+2, self.total_mode, 4*self.num_z)]) )
            self.hamiltonian[idx, idx] += 1j * gamma*np.sum([ReadBit(state, i) for i in range(4*which_dissi, self.total_mode, 4*self.num_z)]) 
            self.hamiltonian[idx, idx] +=  1j * gamma * np.sum([ReadBit(state, i) for i in range(4*which_dissi+1, self.total_mode, 4*self.num_z)]) 
            self.hamiltonian[idx, idx] += 1j * gamma * np.sum([ReadBit(state, i) for i in range(4*which_dissi+2, self.total_mode, 4*self.num_z)]) 
            self.hamiltonian[idx, idx] += 1j * gamma * np.sum([ReadBit(state, i) for i in range(4*which_dissi+3, self.total_mode, 4*self.num_z)])
            # 不嫩换行，不然就不对
            self.row.append(idx)
            self.col.append(idx)
            self.data.append(1j * gamma*np.sum([ReadBit(state, i) for i in range(4*which_dissi, self.total_mode, 4*self.num_z)]) + 1j * gamma * np.sum([ReadBit(state, i) for i in range(4*which_dissi+1, self.total_mode, 4*self.num_z)]) + 1j * gamma * np.sum([ReadBit(state, i) for i in range(4*which_dissi+2, self.total_mode, 4*self.num_z)]) + 1j * gamma * np.sum([ReadBit(state, i) for i in range(4*which_dissi+3, self.total_mode, 4*self.num_z)]))
    
    def add_NH_SingleSite(self, gamma: float, which_dissi:list=[]):
        """在y=1的边界添加非厄米项: i * gamma * a†_m a_m
        which dissi:表示在y等于几加耗散。默认在y=1的边界
        """
        for idx, state in enumerate(self.basis): #第几个basis，矩阵的第几个。
            #print(idx, state, bin(state), ReadBit(state,0), ReadBit(state,1))
            self.hamiltonian[idx, idx] += 1j * gamma*np.sum([ReadBit(state, 4*i) for i in which_dissi]) 
            + 1j * gamma * np.sum([ReadBit(state, 4*i+1) for i in which_dissi])
            + 1j * gamma * np.sum([ReadBit(state, 4*i+2) for i in which_dissi])
            + 1j * gamma * np.sum([ReadBit(state, 4*i+3) for i in which_dissi])

    def add_xhopping_term(self):
        """添加跃迁项: strength * a†_source a_target + h.c."""
        for idx, state in enumerate(self.basis):
            # from A particle
            for i in range(0, self.total_mode, 4): # A sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 4*self.num_y*self.num_z+1 # mx A->mx+1，B
                    if ReadBit(state, i+distance1)==0 and i+distance1<self.total_mode:
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        # self.hamiltonian[jdx1, idx] += 1j/2 * sign
                        # self.hamiltonian[idx, jdx1] += -1j/2 * sign
                        self.row.append(jdx1)
                        self.col.append(idx)
                        self.data.append(-1j/2  *sign)
                        self.row.append(idx)
                        self.col.append(jdx1)
                        self.data.append(1j/2 * sign)
                    distance2 = 4*self.num_y*self.num_z + 3 # mx A->mx+1，D
                    if ReadBit(state, i+distance2)==0 and i+distance2<self.total_mode:
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        # self.hamiltonian[jdx2, idx] += 1j/2 * sign
                        # self.hamiltonian[idx, jdx2] += -1j/2 * sign
                        self.row.append(jdx2)
                        self.col.append(idx)
                        self.data.append(1j/2)
                        self.row.append(idx)
                        self.col.append(jdx2)
                        self.data.append(-1j/2)

            # from B particle
            for i in range(1, self.total_mode, 4): # B sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 4*self.num_y*self.num_z-1 # mx B->mx+1，A
                    if ReadBit(state, i+distance1)==0 and i+distance1<self.total_mode:
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        # self.hamiltonian[jdx1, idx] += -1j/2 * sign
                        # self.hamiltonian[idx, jdx1] += 1j/2 * sign
                        self.row.append(jdx1)
                        self.col.append(idx)
                        self.data.append(-1j/2 * sign)
                        self.row.append(idx)
                        self.col.append(jdx1)
                        self.data.append(1j/2 * sign)
                    distance2 = 4*self.num_y*self.num_z + 1 # mx B->mx+1，C
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
            
            # from C particle
            for i in range(2, self.total_mode, 4): # C sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 4*self.num_y*self.num_z-1 # mx C->mx+1，B
                    if ReadBit(state, i+distance1)==0 and i+distance1<self.total_mode:
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        # self.hamiltonian[jdx1, idx] += 1j/2 * sign
                        # self.hamiltonian[idx, jdx1] += -1j/2 * sign
                        self.row.append(jdx1)
                        self.col.append(idx)
                        self.data.append(1j/2 * sign)
                        self.row.append(idx)
                        self.col.append(jdx1)
                        self.data.append(-1j/2 * sign)
                    distance2 = 4*self.num_y*self.num_z + 1 # mx C->mx+1，D
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
            
            # from D particle
            for i in range(3, self.total_mode, 4): # C sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 4*self.num_y*self.num_z-3 # mx D->mx+1，A
                    if ReadBit(state, i+distance1)==0 and i+distance1<self.total_mode:
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        # self.hamiltonian[jdx1, idx] += 1j/2 * sign
                        # self.hamiltonian[idx, jdx1] += -1j/2 * sign
                        self.row.append(jdx1)
                        self.col.append(idx)
                        self.data.append(1j/2 * sign)
                        self.row.append(idx)
                        self.col.append(jdx1)
                        self.data.append(-1j/2 * sign)
                    distance2 = 4*self.num_y*self.num_z - 1 # mx D->mx+1，C
                    if ReadBit(state, i+distance2)==0 and i+distance2<self.total_mode:
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        # self.hamiltonian[jdx2, idx] += -1j/2 * sign
                        # self.hamiltonian[idx, jdx2] += 1j/2 * sign
                        self.row.append(jdx2)
                        self.col.append(idx)
                        self.data.append(-1j/2 * sign)
                        self.row.append(idx)
                        self.col.append(jdx2)
                        self.data.append(1j/2 * sign)

    def add_yhopping_term(self):
        """添加跃迁项: strength * a†_source a_target + h.c."""
        for idx, state in enumerate(self.basis):
            # from A particle
            for i in range(0, self.total_mode, 4): # A sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 4*self.num_z+1 # mx A->mx+1，B
                    if ReadBit(state, i+distance1)==0 and i%(4*self.num_y*self.num_z)+4*self.num_z<4*self.num_y*self.num_z:
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        # self.hamiltonian[jdx1, idx] += 1j/2 * sign
                        # self.hamiltonian[idx, jdx1] += -1j/2 * sign
                        self.row.append(jdx1)
                        self.col.append(idx)
                        self.data.append(1j/2 * sign)
                        self.row.append(idx)
                        self.col.append(jdx1)
                        self.data.append(-1j/2 * sign)
                    distance2 = 4*self.num_z + 3 # mx A->mx+1，D
                    if ReadBit(state, i+distance2)==0 and i%(4*self.num_y*self.num_z)+4*self.num_z<4*self.num_y*self.num_z:
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        # self.hamiltonian[jdx2, idx] += -1 * sign
                        # self.hamiltonian[idx, jdx2] += -1 * sign
                        self.row.append(jdx2)
                        self.col.append(idx)
                        self.data.append(-1 * sign)
                        self.row.append(idx)
                        self.col.append(jdx2)
                        self.data.append(-1 * sign)

            # from B particle
            for i in range(1, self.total_mode, 4): # B sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 4*self.num_z-1 # mx B->mx+1，A
                    if ReadBit(state, i+distance1)==0 and i%(4*self.num_y*self.num_z)+4*self.num_z<4*self.num_y*self.num_z:
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        # self.hamiltonian[jdx1, idx] += -1j/2 * sign
                        # self.hamiltonian[idx, jdx1] += 1j/2 * sign
                        self.row.append(jdx1)
                        self.col.append(idx)
                        self.data.append(-1j/2 * sign)
                        self.row.append(idx)
                        self.col.append(jdx1)
                        self.data.append(1j/2 * sign)
                    distance2 = 4*self.num_z + 1 # mx B->mx+1，C
                    if ReadBit(state, i+distance2)==0 and i%(4*self.num_y*self.num_z)+4*self.num_z<4*self.num_y*self.num_z:
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        # self.hamiltonian[jdx2, idx] += -1 * sign
                        # self.hamiltonian[idx, jdx2] += -1 * sign
                        self.row.append(jdx2)
                        self.col.append(idx)
                        self.data.append(-1 * sign)
                        self.row.append(idx)
                        self.col.append(jdx2)
                        self.data.append(-1 * sign)
            
            # from C particle
            for i in range(2, self.total_mode, 4): # C sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance2 = 4*self.num_z + 1 # mx C->mx+1，D
                    if ReadBit(state, i+distance2)==0 and i%(4*self.num_y*self.num_z)+4*self.num_z<4*self.num_y*self.num_z:
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
            
            # from D particle
            for i in range(3, self.total_mode, 4): # C sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance2 = 4*self.num_z - 1 # mx D->mx+1，C
                    if ReadBit(state, i+distance2)==0 and i%(4*self.num_y*self.num_z)+4*self.num_z<4*self.num_y*self.num_z:
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        # self.hamiltonian[jdx2, idx] += -1j/2 * sign
                        # self.hamiltonian[idx, jdx2] += 1j/2 * sign
                        self.row.append(jdx2)
                        self.col.append(idx)
                        self.data.append(-1j/2 * sign)
                        self.row.append(idx)
                        self.col.append(jdx2)
                        self.data.append(1j/2 * sign)

    def add_zhopping_term(self):
        """添加跃迁项: strength * a†_source a_target + h.c."""
        for idx, state in enumerate(self.basis):
            # from A particle
            for i in range(0, self.total_mode, 4): # A sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance1 = 4 + 1 # mz A->mx+1，B
                    if ReadBit(state, i+distance1)==0 and i not in list(range(4*self.num_z-4, self.total_mode, 4*self.num_z)):
                        state1 = FlipBit(FlipBit(state, i), i+distance1)
                        jdx1 = self.state_index[state1]
                        sign = self._compute_fermion_sign(state, i, i+distance1)
                        # self.hamiltonian[jdx1, idx] += 1j * sign
                        # self.hamiltonian[idx, jdx1] += -1j * sign
                        self.row.append(jdx1)
                        self.col.append(idx)
                        self.data.append(1j * sign)
                        self.row.append(idx)
                        self.col.append(jdx1)
                        self.data.append(-1j * sign)
            for i in range(3, self.total_mode, 4): # A sublattice.
                if ReadBit(state, i)==1: # 这个格点有粒子
                    distance2 = 4 - 1 # mz D->mz+1，C
                    if ReadBit(state, i+distance2)==0 and i-3 not in list(range(4*self.num_z-4, self.total_mode, 4*self.num_z)):
                        state2 = FlipBit(FlipBit(state, i), i+distance2)
                        jdx2 = self.state_index[state2]
                        sign = self._compute_fermion_sign(state, i, i+distance2)
                        # self.hamiltonian[jdx2, idx] += -1j * sign
                        # self.hamiltonian[idx, jdx2] += 1j * sign
                        self.row.append(jdx2)
                        self.col.append(idx)
                        self.data.append(-1j * sign)
                        self.row.append(idx)
                        self.col.append(jdx2)
                        self.data.append(1j * sign)

    # def add_interaction_intra(self, U):
    #     """添加相互作用U n_{x,y, A} n_{x,y, B}，当一个元胞内有两个粒子能量变大，趋向于不呆在或者呆在一个元胞内。或者是相互之间的，这样chiral skin会让它趋向于局域，但是相互作用可能让它更局域或者弥散。"""
    #     self.inter = np.zeros_like(self.hamiltonian)
    #     for idx, state in enumerate(self.basis):
    #         # # U n_{x,y, A} n_{x,y, B}
    #         for i in range(0, self.total_mode, 2): # A sublattice.
    #             if ReadBit(state, i)==1 and ReadBit(state, i+1)==1:
    #                 self.inter[idx, idx] += U
    #     self.hamiltonian += self.inter
    #     return self.inter

    def add_interaction_inter(self, U):
        """添加相互作用U( n_{x,y, A}+n_{x,y, B}) * (n_{x+1,y, A}+n_{x+1,y, B}))，当近邻元胞有不同的电子数时，会出现相互作用项。这样chiral skin会让它趋向于局域，但是相互作用可能让它更局域或者弥散。但是现象好像是相互作用会使它更局域"""
        # self.inter = np.zeros_like(self.hamiltonian)
        for idx, state in enumerate(self.basis):
            # X: (U n_{x,y, A}+n_{x,y, B}) * (U n_{x+1,y, A}+n_{x+1,y, B}). Y同样。
            distancey = 4*self.num_z
            distancex = 4*self.num_z*self.num_y
            for i in range(0, self.total_mode, 4): # A sublattice.
                # Z direction
                if i not in list(range(4*self.num_z-4, self.total_mode, 4*self.num_z)):
                    # self.inter[idx, idx] += U*(ReadBit(state, i)+ReadBit(state,i+1)+ReadBit(state, i+2)+ReadBit(state,i+3))*(ReadBit(state, i+4)+ReadBit(state,i+5)+ReadBit(state, i+6)+ReadBit(state,i+7))
                    self.row.append(idx)
                    self.col.append(idx)
                    self.data.append(U*(ReadBit(state, i)+ReadBit(state,i+1)+ReadBit(state, i+2)+ReadBit(state,i+3))*(ReadBit(state, i+4)+ReadBit(state,i+5)+ReadBit(state, i+6)+ReadBit(state,i+7)))
                # # y direction
                if i%(distancex)+distancey<distancex:
                    # self.inter[idx, idx] += U*(ReadBit(state, i)+ReadBit(state,i+1)+ReadBit(state, i+2)+ReadBit(state,i+3))*(ReadBit(state, i+distancey)+ReadBit(state,i+distancey+1)+ReadBit(state, i+distancey+2)+ReadBit(state,i+distancey+3))
                    self.row.append(idx)
                    self.col.append(idx)
                    self.data.append(U*(ReadBit(state, i)+ReadBit(state,i+1)+ReadBit(state, i+2)+ReadBit(state,i+3))*(ReadBit(state, i+distancey)+ReadBit(state,i+distancey+1)+ReadBit(state, i+distancey+2)+ReadBit(state,i+distancey+3)))
                # X direction
                if i+distancex<self.total_mode:
                    # self.inter[idx, idx] += U*(ReadBit(state, i)+ReadBit(state,i+1)+ReadBit(state, i+2)+ReadBit(state,i+3))*(ReadBit(state, i+distancex)+ReadBit(state,i+distancex+1)+ReadBit(state, i+distancex+2)+ReadBit(state,i+distancex+3))
                    self.row.append(idx)
                    self.col.append(idx)
                    self.data.append(U*(ReadBit(state, i)+ReadBit(state,i+1)+ReadBit(state, i+2)+ReadBit(state,i+3))*(ReadBit(state, i+distancex)+ReadBit(state,i+distancex+1)+ReadBit(state, i+distancex+2)+ReadBit(state,i+distancex+3)))
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



def CompHam():
    # 比较得到的单比特哈密顿量是否与精确的相等。
    num_x = 2
    num_y = 2
    num_z = 2
    ham1 = RealCSTI_ham(0.2, num_x, num_y, num_z)
    CSTI = ParticleConservedSystem_CSTI(num_x, num_y, num_z, 1)
    CSTI.add_onsite_term()
    CSTI.add_xhopping_term()
    CSTI.add_yhopping_term()
    CSTI.add_zhopping_term()
    CSTI.add_interaction_inter(1.)
    CSTI.add_NH(0.2)
    ham2 = CSTI.hamiltonian
    # print(ham1, ham2)
    print(np.diag(ham1), np.diag(ham2))
    print(np.linalg.norm(ham1-ham2))
    from scipy.sparse import coo_matrix
    spar_ham1 = coo_matrix(ham1)
    spar_ham2 = coo_matrix(ham2)
    print('Hermitian:', np.linalg.norm(ham1 - np.conj(ham1.T)))
    # print(len(spar_ham1.row), len(spar_ham2.row), spar_ham1.row, spar_ham2.row)
    print(np.linalg.norm(spar_ham1.row-spar_ham2.row),np.linalg.norm(spar_ham1.col-spar_ham2.col))

def CompTwoExcit():
    num_x = 3
    num_y = 3
    num_z = 6
    CSTI = ParticleConservedSystem_CSTI(num_x, num_y, num_z, 2)
    CSTI.add_onsite_term()
    CSTI.add_xhopping_term()
    CSTI.add_yhopping_term()
    CSTI.add_zhopping_term()
    # CSTI.add_interaction_inter(1.)
    CSTI.add_NH(0.2)
    ham1 = CSTI.hamiltonian

    CSTI2 = ParticleConservedSystem_CSTI(num_x, num_y, num_z, 2)
    CSTI2.add_onsite_term()
    CSTI2.add_xhopping_term()
    CSTI2.add_yhopping_term()
    CSTI2.add_zhopping_term()
    CSTI2.add_interaction_inter(1.)
    CSTI2.add_NH(0.2)
    ham2 = CSTI2.hamiltonian

    print('Get the Hamiltonian: with total mode', CSTI.dimension)

    eigval_NH, eigvec_NH = np.linalg.eig(ham1)
    eigvec_NH = eigvec_NH[:, np.argsort(np.abs(eigval_NH.imag))]
    eigval_NH = eigval_NH[np.argsort(np.abs(eigval_NH.imag))]
    print('Finish the first one:')
    eigval_NHMany, eigvec_NHMany = np.linalg.eig(ham2)
    eigvec_NHMany = eigvec_NHMany[:, np.argsort(np.abs(eigval_NHMany.imag))]
    eigval_NHMany = eigval_NHMany[np.argsort(np.abs(eigval_NHMany.imag))]

    np.savez('3DCSTI_Spectrum_x%d_y%d_z%d.npz'%(num_x, num_y, num_z), eigval_NH=eigval_NH, eigvec_NH=eigvec_NH, eigval_NHMany=eigval_NHMany, eigvec_NHMany=eigvec_NHMany)

    fig, ax = plt.subplots(1,2,figsize=(8, 4),sharex=True, sharey=True)
    ax[0].scatter(eigval_NH.real, eigval_NH.imag)
    ax[0].scatter(eigval_NH[-1].real, eigval_NH[-1].imag, color='r', s=100, marker='x')
    ax[1].scatter(eigval_NHMany.real, eigval_NHMany.imag)
    ax[1].scatter(eigval_NHMany[-1].real, eigval_NHMany[-1].imag, color='r', s=100, marker='x')
    ax[0].set_xlabel('Re(E)')
    ax[1].set_xlabel('Re(E)')
    ax[0].set_ylabel('Im(E)')
    plt.show()

    fig, ax = plt.subplots(1,2,figsize=(8, 4))
    ax[0].plot(np.abs(eigvec_NH[:,-1])**2)
    ax[1].plot(np.abs(eigvec_NHMany[:,-1])**2)
    ax[0].set_xlabel('Number')
    ax[1].set_xlabel('Number')
    ax[0].set_ylabel(r'$\vert \phi_i(x,y)\vert^2$')
    plt.show()

    # fig, ax = plt.subplots(1,2,figsize=(8, 4))
    # ax[0].imshow(popu_eig.reshape(num_x, 2*num_y))
    # ax[1].imshow(popu_eig2.reshape(num_x, 2*num_y))
    # ax[0].set_xlabel('x')
    # ax[1].set_xlabel('x')
    # ax[0].set_ylabel('y(A,B)')
    # plt.show()

    eigen1 = CSTI.Calcu_Occup(eigvec_NH[:,-1]).reshape((num_x, num_y, 4*num_z))
    eigen_non = CSTI2.Calcu_Occup(eigvec_NHMany[:,-1]).reshape((num_x, num_y, 4*num_z))

    dx, dy, dz = [], [], []
    color_string = 'plasma'
    for x0 in range(num_x):
        for y0 in range(num_y):
            for z0 in range(4*num_z):
                dx.append(x0 + 1)
                dy.append(y0 + 1)
                dz.append(z0 + 1)
    fig = plt.figure()
    ax1 = fig.add_subplot(211, projection='3d')
    im1 = ax1.scatter(dx,dy,dz, c=eigen1, cmap=color_string)
    ax2 = fig.add_subplot(212, projection='3d')
    im2 = ax2.scatter(dx,dy,dz, c=eigen_non, cmap=color_string)
    cbar1 = plt.colorbar(im1, ax=ax1)  # 关联到第一个axes
    cbar2 = plt.colorbar(im2, ax=ax2)  # 关联到第一个axes
    # ax1.set_xlabel(r'$x$', fontsize=font_x)
    # ax1.set_ylabel(r'$y$', fontsize=font_y)
    # ax1.set_zlabel(r'$z$', fontsize=font_z, labelpad=label_pad)
    # ax1.view_init(elev=elev, azim=azim)
    plt.show()

    # 其实也可以搞二维的，
    eigen1 = CSTI.Calcu_Occup(eigvec_NH[:,-1]).reshape((num_x*num_y, 4*num_z))
    eigen_non = CSTI2.Calcu_Occup(eigvec_NHMany[:,-1]).reshape((num_x*num_y, 4*num_z))
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    im1 = ax[0].imshow(eigen1, origin='lower', cmap=color_string)
    im2 = ax[1].imshow(eigen_non, origin='lower', cmap=color_string)
    ax[0].set_xticks(ticks=[0,4,8,12], labels=['1A','2A','3A','4A'])
    ax[0].set_yticks(ticks=list(range(0,num_x*num_y,num_y)), labels=['(%d,1)'%x for x in range(1, num_x+1, 1)])
    ax[0].set_ylabel('(x,y)')
    ax[0].set_xlabel('z(A,B,C,D)')
    ax[1].set_xlabel('z(A,B,C,D)')
    ax[1].set_xticks(ticks=[0,4,8,12], labels=['1A','2A','3A','4A'])
    cbar1 = plt.colorbar(im1, ax=ax[0])  # 关联到第一个axes
    cbar2 = plt.colorbar(im2, ax=ax[1])  # 关联到第一个axes
    
    plt.show()


if __name__=="__main__":
    # CompHam()
    CompTwoExcit()

