"""Small-system reviewer benchmarks. Reuses the unmodified original QWZ model."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys, json, csv, argparse
from pathlib import Path
from itertools import combinations
import numpy as np
import scipy
from scipy.linalg import eig, expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from MeasureNonhermitianSkin.Experimental.ManyBodyQWZ import ParticleConservedSystem


def model(L, u, gamma, U, particles=2):
    m = ParticleConservedSystem(L, L, particles)
    m.add_onsite_term(u)
    m.add_xhopping_term()
    m.add_yhopping_term()
    m.add_NH(gamma)
    m.add_interaction_inter(U)
    return m


def entropy(psi, basis, cut):
    # A is a contiguous prefix in fermionic mode ordering: no mode permutation.
    mask = (1 << cut) - 1
    a = [int(b) & mask for b in basis]
    b = [int(b) >> cut for b in basis]
    ua, ub = sorted(set(a)), sorted(set(b))
    ia, ib = {v:i for i,v in enumerate(ua)}, {v:i for i,v in enumerate(ub)}
    matrix = np.zeros((len(ua), len(ub)), complex)
    for k in range(len(basis)):
        matrix[ia[a[k]], ib[b[k]]] = psi[k]
    p = np.linalg.svd(matrix, compute_uv=False)**2
    p = p[p > 1e-14]
    return float(-np.sum(p*np.log(p)))


def observables(m, psi, L):
    psi = psi/np.linalg.norm(psi)
    occ = np.array([[(int(b)>>j)&1 for j in range(m.total_mode)] for b in m.basis], float)
    p = abs(psi)**2
    density = p @ occ
    pair = occ.T @ (p[:,None]*occ)
    connected = pair - np.outer(density, density)
    cell = density.reshape(L,L,2).sum(2)
    edge = cell[:,0].sum()/2
    adjacent = sum(pair[i,j] for i,j in combinations(range(m.total_mode),2)
                   if abs(i//(2*L)-j//(2*L))+abs((i//2)%L-(j//2)%L)==1)
    rms_y = np.sqrt(np.sum(cell*np.arange(L)[None,:]**2)/2)
    off = connected.copy(); np.fill_diagonal(off,0)
    return dict(edge_weight=float(edge),rms_y=float(rms_y),adjacent_pair=float(adjacent),
                connected_norm=float(np.linalg.norm(off)),
                spatial_entropy=entropy(psi,m.basis,2*L*(L//2))), cell, connected


def checks():
    m = model(2,-1.5,0,0)
    assert np.allclose(m.hamiltonian,m.hamiltonian.conj().T)
    assert m.dimension == 28
    single = model(2,-1.5,.2,0,1)
    es = np.linalg.eigvals(single.hamiltonian)
    sums = np.array([a+b for a,b in combinations(es,2)])
    from scipy.optimize import linear_sum_assignment
    em = np.linalg.eigvals(model(2,-1.5,.2,0).hamiltonian)
    cost = abs(em[:,None]-sums)
    r,c=linear_sum_assignment(cost)
    assert cost[r,c].max()<1e-9
    psi = np.zeros(m.dimension,complex); psi[0]=1
    assert abs(entropy(psi,m.basis,4))<1e-12
    basis=[3,48]; psi=np.ones(2)/np.sqrt(2)
    assert abs(entropy(psi,basis,4)-np.log(2))<1e-12
    return {'hermiticity':True,'noninteracting_pair_spectrum_max_error':float(cost[r,c].max()),
            'product_entropy':0,'bell_entropy':float(np.log(2))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--sizes',nargs='+',type=int,default=[3,4])
    ap.add_argument('--time',type=float,default=10.);ap.add_argument('--points',type=int,default=9)
    ap.add_argument('--output',type=Path,default=ROOT/'results'/'reviewer_experiments')
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    validation=checks(); rows=[]; snapshots={}
    cases=[('topological_gain',-1.5,.2),('trivial_gain',-2.5,.2),('topological_hermitian',-1.5,0.)]
    for L in args.sizes:
        for name,u,gamma in cases:
            for U in np.linspace(0,2,args.points):
                m=model(L,u,gamma,U)
                init=np.zeros(m.dimension,complex)
                init[m.state_index[(1<<1)|(1<<(2*L+1))]]=1
                psi=expm(-1j*m.hamiltonian*args.time)@init
                obs,cell,conn=observables(m,psi,L)
                rows.append(dict(L=L,case=name,U=float(U),time=args.time,**obs))
                if L==max(args.sizes) and name=='topological_gain' and (np.isclose(U,0) or np.isclose(U,1)):
                    e,v=eig(m.hamiltonian)
                    k=np.argmax(e.imag); dominant=v[:,k]/np.linalg.norm(v[:,k])
                    residual=np.linalg.norm(m.hamiltonian@dominant-e[k]*dominant)
                    assert residual<1e-8
                    snapshots[float(U)]=(e,cell,conn)
                    np.savez_compressed(out/f'snapshot_L{L}_U{U:.2f}.npz',energies=e,
                         density=cell,connected=conn,state=psi/np.linalg.norm(psi),
                         dominant_state=dominant,basis=m.basis,residual=residual)
                print(f'L={L} {name} U={U:.2f}',flush=True)
    with (out/'scan.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    metrics=['edge_weight','rms_y','adjacent_pair','connected_norm','spatial_entropy']
    labels=['Gain-edge particle fraction','RMS distance from gain edge (cells)',
            'Adjacent-cell pair probability','Off-diagonal connected norm','Spatial mode entropy (nats)']
    for ax,metric,label in zip(axes.flat,metrics,labels):
        for L in args.sizes:
            for name,_,_ in cases:
                r=[r for r in rows if r['L']==L and r['case']==name]
                ax.plot([x['U'] for x in r],[x[metric] for x in r],marker='.',label=f'{name}, L={L}')
        ax.set(xlabel='Interaction U',ylabel=label);ax.grid(alpha=.2)
    axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles,labels,loc='center',fontsize=8)
    fig.suptitle(f'Two particles; normalized evolution at t={args.time:g}; same product initial state')
    for ext in ['png','pdf']:fig.savefig(out/f'interaction_controls.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for row,U in enumerate(sorted(snapshots)):
        e,cell,conn=snapshots[U]
        axes[row,0].scatter(e.real,e.imag,s=5);axes[row,0].set(xlabel='Re E',ylabel='Im E',title=f'U={U:g}, full spectrum')
        im=axes[row,1].imshow(cell.T,origin='lower',vmin=0);fig.colorbar(im,ax=axes[row,1])
        axes[row,1].set(title='Finite-time cell occupation',xlabel='x (zero-based)',ylabel='y (zero-based)')
        scale=max(abs(conn).max(),1e-12)
        im=axes[row,2].imshow(conn,cmap='RdBu_r',vmin=-scale,vmax=scale);fig.colorbar(im,ax=axes[row,2])
        axes[row,2].set(title='Connected density correlation',xlabel='Mode j',ylabel='Mode i')
    for ext in ['png','pdf']:fig.savefig(out/f'spectrum_density_correlations.{ext}',dpi=180)
    metadata=dict(sizes=args.sizes,time=args.time,U=list(np.linspace(0,2,args.points)),cases=cases,
        numpy=np.__version__,scipy=scipy.__version__,validation=validation,
        note='Finite-time conditional states, not assumed stationary. RMS is not an exponential localization length. Spatial entropy is not particle entanglement or proof of topological protection.')
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')

if __name__=='__main__':main()
