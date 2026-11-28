"""Small open-boundary two-fermion tests; run from any directory."""
from run_interaction_controls import model, observables, ROOT, np, plt, csv
from scipy.linalg import eig, expm
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import csr_matrix
from MeasureNonhermitianSkin.Experimental.ManyBodyQWZ import ParticleConservedSystem

OUT = ROOT / "results" / "reviewer_followup"
OUT.mkdir(parents=True, exist_ok=True)


def save(name, rows):
    with (OUT / (name + ".csv")).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def initial(m, indices):
    psi = np.zeros(m.dimension, complex)
    psi[m.state_index[sum(1 << i for i in indices)]] = 1
    return psi


def natural(m, psi):
    a = np.zeros((m.total_mode, m.total_mode), complex)
    for basis, amplitude in zip(m.basis, psi):
        i, j = [k for k in range(m.total_mode) if (basis >> k) & 1]
        a[i, j], a[j, i] = amplitude, -amplitude
    n = np.linalg.svd(a, compute_uv=False)**2
    assert np.isclose(n.sum(), 2) and n.max() < 1 + 1e-9
    return n


def scan(ax):
    rows, occupations = [], []
    for L in (3, 4):
        prev = None
        for U in np.linspace(0, 2, 9):
            m = model(L, -1.5, .2, U)
            e, v = eig(m.hamiltonian)
            order = np.argsort(e.imag)
            k = order[-1]
            gap = e[k].imag - e[order[-2]].imag
            psi = v[:, k]
            assert np.linalg.norm(m.hamiltonian @ psi - e[k]*psi) < 1e-9
            dyn = expm_multiply(-10j*csr_matrix(m.hamiltonian), initial(m,[1,2*L+1]))
            dyn /= np.linalg.norm(dyn)
            for name, state in [('t10', dyn), ('dominant', psi)]:
                if name == 'dominant' and gap < 1e-9:
                    continue
                n = natural(m,state)
                defect = np.sum(n*(1-n))/2
                if U == 0:
                    assert abs(defect) < 1e-8
                obs = observables(m,state,L)[0]
                rows.append(dict(L=L,U=U,state=name,gap=gap,slater_defect=defect,
                                 t10_fidelity=abs(np.vdot(psi,dyn))**2 if gap>1e-9 else np.nan,
                                 previous_dominant_overlap=abs(np.vdot(prev,psi))**2 if prev is not None and gap>1e-9 else np.nan,**obs))
                occupations.append(dict(L=L,U=U,state=name,occupations=' '.join(map(str,n))))
            prev = psi if gap>1e-9 else None
        for name,style in [('t10','--'),('dominant','-')]:
            r = [x for x in rows if x['L']==L and x['state']==name]
            ax.plot([x['U'] for x in r],[x['slater_defect'] for x in r],style,marker='.',label=f'L={L} {name}')
    ax.set(xlabel='Interaction U',ylabel='Slater defect')
    save('dominant_vs_time',rows)
    save('natural_occupations',occupations)


def reversal(ax):
    rows = []
    L = 4
    for u in (-1.5,-2.5):
        m = model(L,u,0,1)
        h = m.hamiltonian.copy()
        m.add_NH(.2)
        gain = m.hamiltonian-h
        psi = initial(m,[2*L+1,4*L+1])
        for direction in (1,-1):
            gen = (h if direction==1 else h.conj())+gain
            times = np.linspace(0,4,33)
            states = expm_multiply(-1j*csr_matrix(gen),psi,start=0,stop=4,num=len(times))
            signal = []
            for t,state in zip(times,states):
                density = m.Calcu_Occup(state/np.linalg.norm(state)).reshape(L,L,2).sum(2)
                edge = density[:,0]
                moment = np.dot(np.arange(L)-1.5,edge)/2
                rows.append(dict(mass=u,interaction=1,gamma=.2,direction=direction,time=t,edge_moment=moment))
                signal.append(moment)
            ax.plot(times,signal,label=f'u={u}, sign={direction:+d}')
    ax.set(xlabel='Time',ylabel='Signed edge moment (cells)')
    save('chirality_reversal',rows)


def kraus(ax):
    rows = []
    m = ParticleConservedSystem(2,4,2)
    m.add_onsite_term(-1.5)
    m.add_xhopping_term()
    m.add_yhopping_term()
    m.add_interaction_inter(1)
    h = m.hamiltonian.copy()
    m.add_NH(.2)
    edge_gain = np.diag(m.hamiltonian-h).imag
    psi = initial(m,[1,9])
    for dt in (.4,.2,.1,.05):
        steps = round(4/dt)
        # Ancilla starts and ends in |1>: empty mode gets exp(-gamma dt).
        layer = np.exp(dt*(edge_gain-4*.2))[:,None]*expm(-1j*h*dt)
        state = psi.copy()
        for step in range(1,steps+1):
            state = layer@state
            t = step*dt
            exact = expm_multiply(-1j*t*csr_matrix(m.hamiltonian),psi)
            success = np.linalg.norm(state)**2
            fidelity = abs(np.vdot(state,exact))**2/(success*np.linalg.norm(exact)**2)
            rows.append(dict(dt=dt,time=t,layers=step,success=success,infidelity=1-fidelity,
                             mean_shots_for_1000=1000/success))
        trace = [r for r in rows if r['dt']==dt]
        assert np.all(np.diff([r['success'] for r in trace]) <= 1e-10)
        ax.semilogy([r['time'] for r in trace],[r['success'] for r in trace],label=f'dt={dt}')
    ax.set(xlabel='Time',ylabel='Kraus success probability')
    save('kraus_steps',rows)


if __name__ == '__main__':
    fig, axes = plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for func,ax in zip((scan,reversal,kraus),axes):
        func(ax)
        ax.legend(fontsize=8)
    for ext in ('png','pdf'):
        fig.savefig(OUT/f'followup.{ext}',dpi=180)
