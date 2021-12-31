import numpy as np
import pandas as pd
from sys import argv




def run(exp_type):
    dsc=[]
    best_epc=[]
    run_epc=[]
    if exp_type =="sad" :
        tar = "/IMG"
        base = "results/prostate/sfdanotag_notrain"
        exp = ["02","05","07","08","12","17","20","22","26"]
        paper_dsc = "79.4"
    elif exp_type =="whs":
        base = 'results/whs/sfdanotag_notrain'
        exp = ['1019/', '1014/', '1008/', '1003/']
        tar = "/IMG"
        paper_dsc = "75.7"
    elif exp_type =="whs_19":
        base = 'results/whs/sfdanotag_notrain1019_'
        exp = ['c1bis_tmp/', 'c2_tmp/', 'c3_tmp/', 'c4_tmp/']
        tar = "/IMG"
        paper_dsc = "75.7"
    elif exp_type =="whs_select":
        base = 'results/whs/'
        exp = ['sfda_proselect3/', 'sfda_proselect_th2_tmp/', 'sfda_proselect_th2bis_tmp/', 'sfda_proselect_th3/',
               'sfda_proselect_th4','sfda_proselect_th4bis_tmp',
               'sfda_proselect_th4ter_tmp','sfda_proselect_th4_update_tmp']
        tar = "/IMG"
        paper_dsc = "75.7"
    else:
        tar = "/Inn"
        base = "results/ivd/sfdanotag_zero_train_subj"
        exp = ["0bis","15bis","5bis"]
        paper_dsc = "70.9"
    for x in exp:
        with open(base+x+'/3dbestepoch.txt') as f:
            lines = f.readlines()
            dsc.append(eval(lines[0])[1])
            print(x,dsc)
            best_epc.append(eval(lines[0])[0])
        p = pd.read_csv(base+x+tar+'_target_metrics.csv',sep=',')
        run_epc.append(len(p))

    print("current dice",np.round(np.round(np.mean(dsc*100),3)*100,2),"paper dice",paper_dsc)

    print("current mean best epoch",np.round(np.mean((best_epc))))
    print("current run epoch",np.round(np.mean((run_epc))))

def main():

    exp_type = argv[1]

    run(exp_type)

if __name__ == "__main__":
    main()