import numpy as np
import pandas as pd
from sys import argv
import random as random

def run(exp_type,th):
    if exp_type =="sad" :
        p=pd.read_csv("sizes/prostate_proposal.csv",sep=";");q=pd.read_csv("sizes/prostate.csv",sep=",");p=p.sort_values(by=['val_ids'])
        select = [int(u.split(",")[1].split(']')[0])>0.02*384*384 for u in p.proposal_size]
        neg_select = [int(u.split(",")[1].split(']')[0])==0 for u in p.proposal_size]
        dumbpredselect=[q.dumbprednotags[i] if select[i] else "[0, 0]" for i in range(0,len(select))]
        dumbpredselect=[dumbpredselect[i] if not neg_select[i] else "[148225, 0]" for i in range(0,len(neg_select))]
        q["dumbpredselect"]=dumbpredselect
        q.to_csv("sizes/prostate.csv",sep=",",index=False)

    elif exp_type =="whs":
        p = pd.read_csv("sizes/whs_proposal50.csv",sep=";");q=pd.read_csv("sizes/whs.csv",sep=";");p=p.sort_values(by=['val_ids']);s=q[0:26221]
        # select = [np.sum(eval(u)[1:])>0.06*256*256 for u in p.proposal_size];neg_select = [np.sum(eval(u)[1:])==0 for u in p.proposal_size]
        size_vec = [4428, 4996, 4487, 3706]
        select1 = [int(np.sum(eval(u)[1])>size_vec[0]/th) for u in p.proposal_size]
        select2 = [int(np.sum(eval(u)[2])>size_vec[1]/th) for u in p.proposal_size]
        select3 = [int(np.sum(eval(u)[3])>size_vec[2]/th) for u in p.proposal_size]
        select4 = [int(np.sum(eval(u)[4])>size_vec[3]/th) for u in p.proposal_size]
        # select = [np.sum(eval(u)[1])>0.06*256*256 for u in p.proposal_size]

        # Method 1 : Select Confident Slices
        zipped_list = zip(select1, select2, select3,select4)
        sum = [x + y + z + t for (x, y, z, t) in zipped_list]
        # print(sum[1:100])
        select = [s>2 for s in sum]
        neg_select = [np.sum(eval(u)[1:])==0 and random.random() > 0.5 for u in p.proposal_size]
        print("nb positive selected slices",np.sum(select),"nb neg selected slices",np.sum(neg_select))
        dumbpredselect = [q.dumbprednotags[i] if select[i] else "[0,0,0,0,0]" for i in range(0,len(select1))]
        dumbpredselect = [dumbpredselect[i] if not neg_select[i] else "[65536,0,0,0,0]" for i in range(0,len(neg_select))]
        s["dumbpredselect"] = dumbpredselect

        # Method 2 : Keep all slices, estimate Tag with proposal tag
        proposalselect = [get_sizepred_line(q, select1, select2, select3, select4, i) for i in range(0, len(select1))]
        is_eq = [proposalselect[i]==eval(q.dumbpredwtags[i]) for i in range(0, len(select1))]
        print("% of well predicted tag for all classes",np.sum(is_eq)/len(select1))
        s["proposalselect"]=proposalselect
        s.to_csv("sizes/whs_select_th"+str(th)+"_prop50.csv",sep=";",index=False)

    elif exp_type == "whs_byclass":
        cc = [1,2,3,4]
        for c in cc:
            size_vec=[4428, 4996, 4487,3706]
            p=pd.read_csv("sizes/whs_proposal.csv",sep=";");q=pd.read_csv("sizes/whs.csv",sep=";");p=p.sort_values(by=['val_ids']);s=q[0:26221]
            select = [np.sum(eval(u)[c])>size_vec[c-1]/2 for u in p.proposal_size];neg_select = [np.sum(eval(u)[1:])==0 for u in p.proposal_size]
            dumbpredselect=[q.dumbprednotags[i] if select[i] else "[0,0,0,0,0]" for i in range(0,len(select))]
            dumbpredselect=[dumbpredselect[i] if not neg_select[i] else "[65536,0,0,0,0]" for i in range(0,len(neg_select))]
            s["dumbpredselect"]=dumbpredselect
            s.to_csv("sizes/whs_select_class"+str(c)+".csv",sep=";",index=False)

    elif exp_type == "ivd":
        p=pd.read_csv("sizes/ivd_proposal.csv",sep=";");q=pd.read_csv("sizes/ivd.csv",sep=";");p=p.sort_values(by=['val_ids'])
        select = [np.sum(eval(u)[1:]) > 100 for u in p.proposal_size];neg_select = [np.sum(eval(u)[1:])==0 for u in p.proposal_size]
        print(np.sum(select)/(len(select)-np.sum(neg_select)))
        dumbpredselect=[q.dumbprednotags[i] if select[i] else "[0,0]" for i in range(0,len(select))]
        dumbpredselect=[dumbpredselect[i] if not neg_select[i] else "[65536,0]" for i in range(0,len(neg_select))]
        q["dumbpredselect"]=dumbpredselect
        q.to_csv("sizes/ivd_select2.csv",sep=";",index=False)

    elif exp_type == "ivdsag":
        p=pd.read_csv("sizes/ivdsag_proposal.csv",sep=";");q=pd.read_csv("sizes/ivdsag.csv",sep=",");p=p.sort_values(by=['val_ids'])
        select = [np.sum(eval(u)[1:]) > 200 for u in p.proposal_size];neg_select = [np.sum(eval(u)[1:])==0 for u in p.proposal_size]
        print(np.sum(select)/(len(select)-np.sum(neg_select)))
        dumbpredselect=[q.dumbprednotags2[i] if select[i] else "[0,0]" for i in range(0,len(select))]
        dumbpredselect=[dumbpredselect[i] if not neg_select[i] else "[65536,0]" for i in range(0,len(neg_select))]
        q["dumbpredselect"]=dumbpredselect
        q.to_csv("sizes/ivdsag_select.csv",sep=";",index=False)


def get_sizepred_line(q,select1,select2,select3,select4,i):
    c1=select1[i]*eval(q.dumbprednotags[i])[1]
    c2=select2[i]*eval(q.dumbprednotags[i])[2]
    c3=select3[i] * eval(q.dumbprednotags[i])[3]
    c4=select4[i]*eval(q.dumbprednotags[i])[4]
    c0=256*256-(c1+c2+c3+c4)
    return [c0,c1,c2,c3,c4]

def main():

    exp_type = argv[1]
    th = int(argv[2])

    run(exp_type,th)

if __name__ == "__main__":
    main()