# Copied 2026-11-02 08:59 CET from the numerical-stability review (written 08:53-08:55 CET); unchanged below this line.
# MC error of the 20-draw corpus-bearing control AUROC, same-length LODO contrast (as bayes.py bb_cell), all corpora x models
import csv, gzip, sys, numpy as np, glob, os
R=os.environ.get('OUTROOT', 'out_clean')  # folder with features_<model>.csv.gz and per-corpus subfolders
def auc(pos,neg):
    v=np.r_[pos,neg]; r=np.argsort(np.argsort(v))+1.0; n1=len(pos); n0=len(neg)
    return (r[:n1].sum()-n1*(n1+1)/2)/(n1*n0)
rows=[]
for corp,d in [('nobel',R),('wiki',R+'/wiki'),('gutenberg',R+'/gutenberg'),('turing',R+'/turing'),('fields',R+'/fields')]:
    for f in sorted(glob.glob(d+'/features_*.csv.gz')):
        m=os.path.basename(f)[9:-7]
        with gzip.open(f,'rt') as fh:
            rd=csv.DictReader(fh); A=[];B=[]
            for row in rd:
                if row['variant']=='lodo' and row['cell']=='Q250|C750': A.append(row)
                elif row['variant']=='lodo_dens' and row['cell']=='Q250|C250': B.append(row)
        if not A or 'longbar_enull00' not in A[0] or A[0]['longbar_enull00']=='': continue
        for ft in ('w1','longbar'):
            real=auc(np.array([float(r[ft]) for r in B]),np.array([float(r[ft]) for r in A]))
            nul=np.array([auc(np.array([float(r[f'{ft}_enull{k:02d}']) for r in B]),np.array([float(r[f'{ft}_enull{k:02d}']) for r in A])) for k in range(20)])
            rows.append((corp,m,ft,len(A),len(B),real-nul.mean(),nul.std(ddof=1),nul.std(ddof=1)/np.sqrt(20)))
for ft in ('w1','longbar'):
    x=np.array([r[5:] for r in rows if r[2]==ft])
    print(ft,'n=',len(x),'gain median',np.median(x[:,0]).round(3),'single-draw SD median/max',np.median(x[:,1]).round(4),x[:,1].max().round(4),'MC SE of 20-mean median/max',np.median(x[:,2]).round(4),x[:,2].max().round(4))
    print('   |gain|/MCSE min', np.min(np.abs(x[:,0])/x[:,2]).round(1), ' gains within 2 MCSE of 0:', int(np.sum(np.abs(x[:,0])<2*x[:,2])))
