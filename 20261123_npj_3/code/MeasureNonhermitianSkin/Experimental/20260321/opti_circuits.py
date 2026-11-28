import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import state_fidelity
from qiskit_aer import Aer
from scipy.optimize import minimize


def real_ham_qwz_dissipation(u, a, gamma, num_xsite, num_ysite):
    """Build the single-particle non-Hermitian QWZ Hamiltonian."""
    ham = np.zeros((2 * num_xsite * num_ysite, 2 * num_xsite * num_ysite), dtype="complex")

    for delta_x in range(num_xsite):
        for delta_y in range(num_ysite):
            row = 2 * (delta_x * num_ysite + delta_y)
            ham[row : row + 2, row : row + 2] = [[u, 0], [0, -u]]

    for delta_x in range(num_xsite - 1):
        for delta_y in range(num_ysite):
            delta_col = 2 * (delta_x * num_ysite + delta_y)
            delta_row = 2 * ((delta_x + 1) * num_ysite + delta_y)
            ham[delta_row : delta_row + 2, delta_col : delta_col + 2] = [[1 / 2, a * 1j / 2], [a * 1j / 2, -1 / 2]]
            ham[delta_col : delta_col + 2, delta_row : delta_row + 2] = [[1 / 2, -a * 1j / 2], [-a * 1j / 2, -1 / 2]]

    for delta_x in range(num_xsite):
        for delta_y in range(num_ysite - 1):
            delta_col = 2 * (delta_x * num_ysite + delta_y)
            delta_row = 2 * (delta_x * num_ysite + delta_y + 1)
            ham[delta_row : delta_row + 2, delta_col : delta_col + 2] = [[1 / 2, 1 / 2], [-1 / 2, -1 / 2]]
            ham[delta_col : delta_col + 2, delta_row : delta_row + 2] = [[1 / 2, -1 / 2], [1 / 2, -1 / 2]]

    # Dissipation on y=0 boundary.
    for delta_y in range(num_xsite):
        row = 2 * (delta_y * num_ysite + 0)
        ham[row : row + 2, row : row + 2] += [[1j * gamma, 0.0], [0.0, 1j * gamma]]
    return ham


def get_target_state():
    """Return the target eigenstate used in optimization."""
    u = -1.5
    gamma = 0.2
    num_xsite = 2
    num_ysite = 4
    ham = real_ham_qwz_dissipation(u, 1, gamma, num_xsite, num_ysite)
    eigval, eigvec = np.linalg.eig(ham)
    order = np.argsort(np.abs(eigval.real))
    eigvec = eigvec[:, order]
    return eigvec[:, 0].reshape(-1, 1)


TARGET_STATE = get_target_state()
NUM_QUBITS = 4
SIMULATOR = Aer.get_backend("statevector_simulator")

# Control CNOT count per layer, e.g. 5 or 6.
CNOT_COUNT_PER_LAYER = 5

# Use a middle 1q block between two CNOT segments.
USE_MIDDLE_1Q_BLOCK = True

# Forward + reverse nearest-neighbor pattern (cycled if count is larger).
CNOT_PAIR_POOL = [(0, 1), (1, 2), (2, 3)]


def params_per_layer(num_qubits):
    """3 params per qubit in multiple single-qubit blocks."""
    num_single_qubit_blocks = 3 if USE_MIDDLE_1Q_BLOCK else 2
    return 3 * num_qubits * num_single_qubit_blocks


def add_single_qubit_block(qc, angles, num_qubits):
    """Apply generic single-qubit U gates."""
    for qi in range(num_qubits):
        theta = angles[3 * qi]
        phi = angles[3 * qi + 1]
        lam = angles[3 * qi + 2]
        qc.u(theta, phi, lam, qi)


def create_layer(num_qubits, layer_params):
    """Create one ansatz layer with extra parameterized flexibility."""
    qc = QuantumCircuit(num_qubits)

    block_size = 3 * num_qubits
    front_angles = layer_params[:block_size]
    if USE_MIDDLE_1Q_BLOCK:
        middle_angles = layer_params[block_size : 2 * block_size]
        back_angles = layer_params[2 * block_size : 3 * block_size]
    else:
        middle_angles = None
        back_angles = layer_params[block_size : 2 * block_size]

    add_single_qubit_block(qc, front_angles, num_qubits)

    split_cnot = CNOT_COUNT_PER_LAYER // 2 if USE_MIDDLE_1Q_BLOCK else CNOT_COUNT_PER_LAYER
    for idx in range(split_cnot):
        q1, q2 = CNOT_PAIR_POOL[idx % len(CNOT_PAIR_POOL)]
        qc.cz(q1, q2)

    if USE_MIDDLE_1Q_BLOCK:
        add_single_qubit_block(qc, middle_angles, num_qubits)
        for idx in range(split_cnot, CNOT_COUNT_PER_LAYER):
            q1, q2 = CNOT_PAIR_POOL[idx % len(CNOT_PAIR_POOL)]
            qc.cz(q1, q2)

    add_single_qubit_block(qc, back_angles, num_qubits)
    
    return qc


def cost_function(params, num_layers, num_qubits, record_loss=None, type_optimize=None):
    """Compute infidelity to the target state."""
    if type_optimize == "Optimize_All":
        ppl = params_per_layer(num_qubits)
        num_full_layers = len(params) // ppl
        fixed_prev_params = [params[i * ppl : (i + 1) * ppl] for i in range(max(0, num_full_layers - 1))]
        current_layer_params = params[-ppl:]
    else:
        fixed_prev_params = []
        current_layer_params = params

    qc = QuantumCircuit(num_qubits)
    for layer_param in fixed_prev_params:
        qc.compose(create_layer(num_qubits, layer_param), inplace=True)
    qc.compose(create_layer(num_qubits, current_layer_params), inplace=True)

    result = SIMULATOR.run(qc).result()
    final_state = result.get_statevector(qc)
    fid = state_fidelity(final_state, TARGET_STATE)
    loss = 1.0 - fid

    if record_loss is not None:
        record_loss[num_layers].append(loss)
    return loss


def opt_vqe(max_layers=1, threshold=0.01, num_restarts=5):
    """Layer-wise growth while optimizing all current parameters."""
    all_loss = defaultdict(list)
    best_params_all = np.array([], dtype=float)
    ppl = params_per_layer(NUM_QUBITS)

    for num_layers in range(1, max_layers + 1):
        print(f"\nOptimizing layer {num_layers}...")
        candidate_best_error = np.inf
        candidate_best_params = None

        for _ in range(num_restarts):
            initial_params = np.random.uniform(0.0, 2.0 * np.pi, size=ppl)
            init_all = np.append(best_params_all, initial_params)
            result = minimize(
                fun=lambda p: cost_function(p, num_layers, NUM_QUBITS, record_loss=all_loss, type_optimize="Optimize_All"),
                x0=init_all,
                method="L-BFGS-B", # "COBYLA", # 
                options={"maxiter": 10000},
            )
            if result.fun < candidate_best_error:
                candidate_best_error = result.fun
                candidate_best_params = result.x

        best_params_all = candidate_best_params
        current_error = candidate_best_error
        print(f"Layer {num_layers} achieved error: {current_error:.6f}")
        if current_error < threshold:
            print(f"Converged at layer {num_layers} with error {current_error:.6f}")
            break

    split_params = [best_params_all[i * ppl : (i + 1) * ppl] for i in range(len(best_params_all) // ppl)]
    return split_params, num_layers, all_loss


def check_opt():
    """Run optimization, evaluate fidelity, and visualize populations."""
    optimal_params_per_layer, _, loss = opt_vqe(max_layers=1)

    final_circuit = QuantumCircuit(NUM_QUBITS)
    for layer_params in optimal_params_per_layer:
        final_circuit.compose(create_layer(NUM_QUBITS, layer_params), inplace=True)

    # Draw the optimized circuit.
    try:
        fig = final_circuit.draw(output="mpl", fold=-1)
        fig.savefig("optimized_circuit.png", dpi=200, bbox_inches="tight")
        plt.show()
        print("Saved optimized circuit figure: optimized_circuit.png")
    except Exception:
        # Fallback if matplotlib drawer dependencies are unavailable.
        print("Matplotlib circuit drawer is unavailable, fallback to text view:")
        print(final_circuit.draw(output="text", fold=-1))

    result = SIMULATOR.run(final_circuit).result()
    final_state = result.get_statevector(final_circuit)
    fidelity = state_fidelity(final_state, TARGET_STATE)
    print(f"\nFinal state fidelity: {fidelity:.6f}")

    for i in range(1, 15):
        if len(loss[i]) > 0:
            plt.plot(loss[i], label=f"layer {i}")
    plt.title("Loss history")
    plt.xlabel("Iteration")
    plt.ylabel("Infidelity")
    plt.show()

    num_xsite, num_ysite = 2, 4
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    popu_final = np.zeros(2 * num_xsite * num_ysite)
    popu_target = np.zeros(2 * num_xsite * num_ysite)

    final_data = final_state.data if hasattr(final_state, "data") else np.asarray(final_state)
    for k in range(2 * num_xsite * num_ysite):
        popu_final[k] = np.abs(final_data[k]) ** 2
        popu_target[k] = np.abs(TARGET_STATE[k, 0]) ** 2

    popu_final = (popu_final[::2] + popu_final[1::2]).reshape(num_xsite, num_ysite).T
    popu_target = (popu_target[::2] + popu_target[1::2]).reshape(num_xsite, num_ysite).T

    vmin = min(popu_final.min(), popu_target.min())
    vmax = max(popu_final.max(), popu_target.max())

    im1 = ax[0].imshow(popu_final, origin="lower", cmap="gist_heat_r", vmin=vmin, vmax=vmax)
    ax[0].set_title("Final State")

    im2 = ax[1].imshow(popu_target, origin="lower", cmap="gist_heat_r", vmin=vmin, vmax=vmax)
    ax[1].set_title("Target State")

    fig.suptitle(f"Fidelity: {fidelity:.6f}")
    plt.colorbar(im2, ax=ax[1])
    plt.tight_layout()
    plt.show()


    # [Start of the visualization section]
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    color_string = 'gist_heat_r'
    u = -1.5 # u 太大之后使topological trivial的。
    gamma = 0.2
    num_xsite = 2
    num_ysite = 4
    
    # Prepare the population data
    popu_eig1 = np.zeros(16)
    for k in range(2*num_xsite*num_ysite):
        popu_eig1[k] = np.sum(np.abs(final_state.data[k])**2)
    popu_eig1 = (popu_eig1[::2] + popu_eig1[1::2]).reshape(num_xsite, num_ysite).T

    popu_eig2 = np.zeros(16)
    for k in range(2*num_xsite*num_ysite):
        popu_eig2[k] = np.sum(np.abs(TARGET_STATE[k,0])**2)
    popu_eig2 = (popu_eig2[::2] + popu_eig2[1::2]).reshape(num_xsite, num_ysite).T

    vmin = min(popu_eig1.min(), popu_eig2.min())
    vmax = max(popu_eig1.max(), popu_eig2.max())

    # Helper function to annotate the heatmap
    def annotate_heatmap(data, axis):
        rows, cols = data.shape
        for i in range(rows):
            for j in range(cols):
                # Calculate text color based on background intensity for readability
                text_color = "white" if data[i, j] > (vmax + vmin) / 2 else "black"
                axis.text(j, i, f'{data[i, j]:.4f}', 
                          ha="center", va="center", color=text_color, fontsize=9)

    # Plot 1: Final State
    im1 = ax[0].imshow(popu_eig1, origin='lower', cmap=color_string, vmin=vmin, vmax=vmax)
    annotate_heatmap(popu_eig1, ax[0])
    ax[0].set_title('Final State')

    # Plot 2: Target State
    im2 = ax[1].imshow(popu_eig2, origin='lower', cmap=color_string, vmin=vmin, vmax=vmax)
    annotate_heatmap(popu_eig2, ax[1])
    ax[1].set_title('Target State')

    fig.suptitle(f'Fidelity: {fidelity:.4f}')
    plt.colorbar(im2, ax=ax[1])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    start_time = time.time()
    check_opt()
    print((time.time() - start_time) / 60)
