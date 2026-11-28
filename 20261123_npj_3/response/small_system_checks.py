"""Small open QWZ benchmarks for responses 5 and 7; run with Python from any cwd."""
from pathlib import Path
import sys
import csv
import numpy as np
from scipy.linalg import eig
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/workflows/reviewer_experiments'))
from run_interaction_controls import model

def main():
    rng=np.random.default_rng(1128)
    rows=[]
    for n in (2,3):
        m=model(3,-1.5,.2,1,n)
        bits=np.array([[(b>>i)&1 for i in range(18)] for b in m.basis])
        cells=bits.reshape(-1,3,3,2).sum(3)
        # Fraction of unordered particle pairs occupying nearest-neighbour cells.
        adj=((cells[:,:-1]*cells[:,1:]).sum((1,2))+
             (cells[:,:,:-1]*cells[:,:,1:]).sum((1,2)))/(n*(n-1)/2)
        for kind in (['clean','onsite','bond','sublattice'] if n==2 else ['clean']):
            for run in range(12 if kind in ['onsite','bond'] else 1):
                h=m.hamiltonian.copy()
                if kind=='onsite':
                    h[np.diag_indices_from(h)]+=bits@rng.uniform(-.1,.1,18)
                if kind=='sublattice':
                    # Local real potential on just the first A mode.
                    h[np.diag_indices_from(h)]+=.1*bits[:,0]
                if kind=='bond':
                    # Share each real fractional bond error across all spectator configurations.
                    scale={}
                    for a,b in zip(*np.nonzero(np.triu(h,1))):
                        key=m.basis[a]^m.basis[b]
                        if key not in scale: scale[key]=1+rng.uniform(-.1,.1)
                        h[a,b]*=scale[key]
                        h[b,a]*=scale[key]
                e,v=eig(h)
                order=np.argsort(e.imag)
                k=order[-1]
                gap=e[k].imag-e[order[-2]].imag
                assert gap>1e-9 and np.linalg.norm(h@v[:,k]-e[k]*v[:,k])<1e-9
                p=abs(v[:,k])**2
                rows.append(dict(n=n,kind=kind,run=run,gap=gap,edge=p@cells[:,:,0].sum(1)/n,adjacent=p@adj))
    dest=Path(__file__).with_name('small_system_checks.csv')
    with dest.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    for n,kind in sorted({(r['n'],r['kind']) for r in rows}):
        a=np.array([[r['edge'],r['adjacent']] for r in rows if (r['n'],r['kind'])==(n,kind)])
        print(n,kind,a.mean(0),a.std(0,ddof=1) if len(a)>1 else '')
if __name__=='__main__': main()
