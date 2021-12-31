from sys import argv
import pandas as pd
import numpy as np
import sklearn
import operator
from pathlib import Path
import os
from utils import map_
from sklearn.metrics import confusion_matrix,classification_report
import warnings
pd.options.mode.chained_assignment = None

def main():


    #fx = lambda a : a.split("test")[1].split("reg")[0]
    fx = lambda a : a.split("train")[1].split("reg")[0]
    #fx = lambda a : a.split("train_val")[1].split("reg")[0]
    #print(path2.split(os.sep)[2])
    savefold = Path("sizes")
    savefold.mkdir(parents=True, exist_ok=True)

    base = 900
    savename = Path(savefold,"ivdsag.csv")
    p=pd.read_csv('/data/users/mathilde/ccnn/CDA/data/all_sagittal/train/sizes_all_classesGT.csv')
    q = pd.read_csv('/data/users/mathilde/ccnn/CDA/data/all_sagittal/val/sizes_all_classesGT.csv')
    q = pd.concat([p,q])
    qvals = list(q.val_ids)
    print(qvals[1:10])
    qvals.sort()

    q=q.sort_values(by=['val_ids'])
    #print(qvals[1:10])

    # dumb predictor : each class its mean + tag
    median_vec = np.empty((2))
    for nclass in range(0,2):
        print(nclass)
        gtsize = list(q.val_gt_size)
        gtsize = [eval(f)[nclass] for f in gtsize]
        gtclass = [i>0 for i in gtsize]



        median_pos = np.round(np.median((np.array([g for g in gtsize if g >0]))))
        if nclass >0:
            mean_pos = base
        else:
            mean_pos = 256*256-base 

        median_vec[nclass] = median_pos
        dumbpredgt = list(map(operator.mul, gtclass, np.repeat(mean_pos, len(gtclass))))  # DUMBPREDGT : MEAN
        dumbpredgt2 = list(map(operator.mul, gtclass, np.repeat(median_pos, len(gtclass))))
        if nclass ==0:
            dumbpredgtvec = [[dumbpredgt[i]] for i in range(0,len(dumbpredgt))]
            dumbpredgtvec2 = [[dumbpredgt2[i]] for i in range(0,len(dumbpredgt2))]
            gtvec = [[gtsize[i]] for i in range(0,len(dumbpredgt))]
        else:
            dumbpredgtvec = [list(np.append(dumbpredgtvec[i],dumbpredgt[i])) for i in range(0,len(dumbpredgt))]
            dumbpredgtvec2 = [list(np.append(dumbpredgtvec2[i],dumbpredgt2[i])) for i in range(0,len(dumbpredgt2))]
            gtvec = [list(np.append(gtvec[i],gtsize[i])) for i in range(0,len(dumbpredgt))]

        print("median vec",median_vec)
        #bb = [[ np.float(vps[i][j]>median_vec[j]/10) for j in range(0,5)] for i in range(0,len(vps))] 
        #q['cutpred']= [list(np.multiply(bb[i],vps[i])) for i in range(0,len(vps))]
        q['dumbpredwtags'] = dumbpredgtvec
        q['dumbpredwtags2'] = dumbpredgtvec2
    dumbpredgtvec = np.array(dumbpredgtvec)
    gtvec = np.array(gtvec)
    diffd= np.mean([np.mean(np.abs(dumbpredgtvec[i] -gtvec[i])) for i in range(0,len(gtsize))])

    q["dumbpredwtags"] = map_(lambda x:[256*256-x[1],x[1]] , q.dumbpredwtags) 

    q.to_csv(savename,index=False)
    print("saving",savename)
    print(q[['val_gt_size','dumbpredwtags']][500:510])
    print(q[['val_gt_size','dumbpredwtags']][600:610])
    print(q[['val_gt_size','dumbpredwtags']][700:710])
    #print(q.head(3))
    return



if __name__ == "__main__":
    main()
