import time
import logging
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

# Use fewer two-qubit gates per layer.
CNOT_COUNT_PER_LAYER = 3

# Use more single-qubit blocks to compensate expressibility.
SINGLE_QUBIT_BLOCKS_PER_LAYER = 4

# Forward + reverse nearest-neighbor pattern (cycled if count is larger).
CNOT_PAIR_POOL = [(0, 1), (1, 2), (2, 3)]

# Layer-wise stack: first layer = full prep + CZ sandwich; later layers = only CZ + 2 U per CZ (6 U total).
ENABLE_LAYERWISE_ANSATZ = True
ANSATZ_DEPTH = 2

# First layer: U on all qubits, then for each pair: CZ + U on that pair (same as before).
PARAMS_FIRST_LAYER = 3 * NUM_QUBITS + 3 * (3 * 2)  # 12 + 18 = 30
# Tail layers: no leading full block; only CZ then U on the two qubits, repeated for each pair -> 6 U gates.
PARAMS_TAIL_LAYER = 3 * 2 * len(CNOT_PAIR_POOL)  # 18


def total_params_for_depth(num_stack_layers: int) -> int:
    """Total trainable params for a stack of `num_stack_layers` (>=1)."""
    if num_stack_layers < 1:
        return 0
    return PARAMS_FIRST_LAYER + (num_stack_layers - 1) * PARAMS_TAIL_LAYER


def params_per_layer(num_qubits):
    """Deprecated name: first-layer param count (for single-layer mode)."""
    _ = num_qubits
    return PARAMS_FIRST_LAYER


def add_single_qubit_block(qc, angles, num_qubits):
    """Apply generic single-qubit U gates."""
    for qi in range(num_qubits):
        theta = angles[3 * qi]
        phi = angles[3 * qi + 1]
        lam = angles[3 * qi + 2]
        qc.u(theta, phi, lam, qi)

def add_single_qubit_cz(qc, angles, qubits):
    """Apply generic single-qubit U gates."""
    for qi in range(len(qubits)):
        theta = angles[3 * qi]
        phi = angles[3 * qi + 1]
        lam = angles[3 * qi + 2]
        qc.u(theta, phi, lam, qubits[qi])


def create_layer_first(num_qubits, layer_params):
    """First stack layer: full single-qubit block, then CZ + 2 U per bond."""
    if len(layer_params) != PARAMS_FIRST_LAYER:
        raise ValueError(f"First layer expects {PARAMS_FIRST_LAYER} params, got {len(layer_params)}.")
    qc = QuantumCircuit(num_qubits)

    block_size = 3 * num_qubits
    cz_size = 3 * 2
    angle_blocks = []
    angle_blocks.append(layer_params[0:block_size])
    for bidx in range(len(CNOT_PAIR_POOL)):
        begin = bidx * cz_size + block_size
        end = (bidx + 1) * cz_size + block_size
        angle_blocks.append(layer_params[begin:end])

    add_single_qubit_block(qc, angle_blocks[0], num_qubits)
    for i in range(0, len(CNOT_PAIR_POOL), 1):
        q1, q2 = CNOT_PAIR_POOL[i]
        qc.cz(q1, q2)
        add_single_qubit_cz(qc, angle_blocks[i + 1], [q1, q2])

    return qc


def create_layer_tail(num_qubits, layer_params):
    """Later stack layers: only CZ then two U on that pair (6 single-qubit gates total)."""
    if len(layer_params) != PARAMS_TAIL_LAYER:
        raise ValueError(f"Tail layer expects {PARAMS_TAIL_LAYER} params, got {len(layer_params)}.")
    qc = QuantumCircuit(num_qubits)
    offset = 0
    for i in range(len(CNOT_PAIR_POOL)):
        q1, q2 = CNOT_PAIR_POOL[i]
        qc.cz(q1, q2)
        block = layer_params[offset : offset + 6]
        offset += 6
        add_single_qubit_cz(qc, block, [q1, q2])
    return qc


# Backward-compatible alias
def create_layer(num_qubits, layer_params):
    return create_layer_first(num_qubits, layer_params)


def split_params_into_layers(flat_params):
    """Split flat vector into [first_layer, tail, tail, ...]."""
    fp = np.asarray(flat_params, dtype=float).ravel()
    n = len(fp)
    if n < PARAMS_FIRST_LAYER:
        raise ValueError(f"Need at least {PARAMS_FIRST_LAYER} params, got {n}.")
    chunks = [fp[:PARAMS_FIRST_LAYER]]
    rem = n - PARAMS_FIRST_LAYER
    if rem % PARAMS_TAIL_LAYER != 0:
        raise ValueError(
            f"After first layer ({PARAMS_FIRST_LAYER} params), remainder {rem} "
            f"must be a multiple of tail size {PARAMS_TAIL_LAYER}."
        )
    for j in range(rem // PARAMS_TAIL_LAYER):
        a = PARAMS_FIRST_LAYER + j * PARAMS_TAIL_LAYER
        chunks.append(fp[a : a + PARAMS_TAIL_LAYER])
    return chunks


def combine_layers_to_circuit(num_qubits, layer_chunks):
    qc = QuantumCircuit(num_qubits)
    if not layer_chunks:
        return qc
    qc.compose(create_layer_first(num_qubits, layer_chunks[0]), inplace=True)
    for c in layer_chunks[1:]:
        qc.compose(create_layer_tail(num_qubits, c), inplace=True)
    return qc


def build_ansatz_from_params(params, num_qubits):
    """Build circuit from flat params (first + optional tails)."""
    return combine_layers_to_circuit(num_qubits, split_params_into_layers(params))


def cost_function(params, num_layers, num_qubits, record_loss=None, type_optimize=None):
    """Compute infidelity to the target state."""
    p = np.asarray(params, dtype=float).ravel()

    if not ENABLE_LAYERWISE_ANSATZ:
        qc = create_layer_first(num_qubits, p[:PARAMS_FIRST_LAYER])
    else:
        expected = total_params_for_depth(num_layers)
        if len(p) != expected:
            raise ValueError(f"For stack depth {num_layers}, expected {expected} params, got {len(p)}.")
        qc = build_ansatz_from_params(p, num_qubits)

    result = SIMULATOR.run(qc).result()
    final_state = result.get_statevector(qc)
    fid = state_fidelity(final_state, TARGET_STATE)
    loss = 1.0 - fid

    if record_loss is not None:
        record_loss[num_layers].append(loss)
    return loss


def opt_vqe(max_layers=1, threshold=0.01, num_restarts=5):
    """Layer-wise growth while optimizing all current parameters."""
    if not ENABLE_LAYERWISE_ANSATZ:
        max_layers = 1
    all_loss = defaultdict(list)
    best_params_all = np.array([], dtype=float)

    for num_layers in range(1, max_layers + 1):
        print(f"\nOptimizing layer {num_layers}...")
        candidate_best_error = np.inf
        candidate_best_params = None

        new_block = PARAMS_FIRST_LAYER if num_layers == 1 else PARAMS_TAIL_LAYER

        for _ in range(num_restarts):
            initial_params = np.random.uniform(0.0, 2.0 * np.pi, size=new_block)
            init_all = np.append(best_params_all, initial_params)
            result = minimize(
                fun=lambda p: cost_function(p, num_layers, NUM_QUBITS, record_loss=all_loss, type_optimize="Optimize_All"),
                x0=init_all,
                method= "L-BFGS-B", # "COBYLA", # 
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

    split_params = split_params_into_layers(best_params_all)
    return split_params, num_layers, all_loss


def preview_layer_stack(num_layers=3, seed=0, save_path="preview_ansatz_layers.png"):
    """
    Visual-only: stack first layer + (num_layers-1) tail layers with random U parameters.
    Use this to see what e.g. 3 layers look like without running optimization.
    """
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(NUM_QUBITS)
    lp_first = rng.uniform(0.0, 2.0 * np.pi, size=PARAMS_FIRST_LAYER)
    qc.compose(create_layer_first(NUM_QUBITS, lp_first), inplace=True)
    for _ in range(num_layers - 1):
        lp = rng.uniform(0.0, 2.0 * np.pi, size=PARAMS_TAIL_LAYER)
        qc.compose(create_layer_tail(NUM_QUBITS, lp), inplace=True)
    total_p = total_params_for_depth(num_layers)
    print(
        f"Preview: {num_layers} stacked layers (1 first + {num_layers - 1} tail), "
        f"total params={total_p} (first={PARAMS_FIRST_LAYER}, tail={PARAMS_TAIL_LAYER})."
    )
    try:
        fig = qc.draw(output="mpl", fold=-1)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.show()
        print(f"Saved preview figure: {save_path}")
    except Exception:
        print(qc.draw(output="text", fold=-1))
    return qc


def check_opt():
    """Run optimization, evaluate fidelity, and visualize populations."""
    max_layers = ANSATZ_DEPTH if ENABLE_LAYERWISE_ANSATZ else 1
    optimal_params_per_layer, _, loss = opt_vqe(max_layers=max_layers)

    final_circuit = combine_layers_to_circuit(NUM_QUBITS, optimal_params_per_layer)

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
    # Set PREVIEW_ONLY=True to only draw a multi-layer ansatz (no optimization).
    PREVIEW_ONLY = True
    if PREVIEW_ONLY:
        preview_layer_stack(num_layers=ANSATZ_DEPTH, seed=0)
        check_opt()
    else:
        check_opt()
    print((time.time() - start_time) / 60)
