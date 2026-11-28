from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator
from scipy.linalg import expm
import numpy as np
import matplotlib.pyplot as plt

import quafu

from quafu import User
user = User("<your API token>")
user.save_apitoken()

# qc = QuantumCircuit(3)
# # qc.ccx(0.1, 0, 1, 2)  # 原始 Toffoli 门
# transpiled_qc = transpile(qc, routing_method='sabre', basis_gates=['rx', 'ry', 'cx'], optimization_level=3) # , 'ccx'
# print(transpiled_qc)

qc = quafu.QuantumCircuit(5)
qc.x(0)
qc.x(1)
qc.cnot(2, 1)
# qc.ry(1, np.pi/2)
qc.rx(2, np.pi)
qc.rz(3, 0.1)
qc.cz(2, 3)

# qc.draw_circuit()
qc.plot_circuit(title='A Quantum Circuit')
plt.show()

res = quafu.simulate(qc)
res.plot_probabilities()


# 第一种方式
from pyquafu import Circuit
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import OneQubitEvoGate

# 步骤1：使用Qiskit等工具预编译幺正矩阵
U = np.array([[0,1], [1,0]])  # 示例幺正矩阵
qc = QuantumCircuit(1)
qc.unitary(U, [0])  # Qiskit自动分解
decomposed_gates = qc.decompose().data  # 获取基础门序列

# 步骤2：转换为PyQuafu线路
pyq_circ = Circuit(1)
for gate in decomposed_gates:
    if gate.operation.name == 'rx':
        pyq_circ.rx(0, gate.operation.params[0])
    # 添加其他门类型转换...

# 步骤3：提交任务
task = Task(backend="ScQ-P10")
task.send(pyq_circ)

# 第二种方式
from pyquafu import CustomGate

# 定义自定义门（需提前进行门分解）
def build_unitary_gate():
    # 示例：将2x2矩阵分解为Rz-Rx-Rz序列
    theta, phi, lam = 1.57, 0.5, -0.5  # 通过矩阵分解计算获得
    circ = Circuit(1)
    circ.rz(0, lam)
    circ.rx(0, theta)
    circ.rz(0, phi)
    return circ

U_gate = CustomGate(build_unitary_gate())
main_circ = Circuit(2)
main_circ.add_gate(U_gate, [0])  # 应用到目标比特