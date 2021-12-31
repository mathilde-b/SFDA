import pandas as pd
from pathlib import Path
pd.options.mode.chained_assignment = None

def main():

    savefold = Path("sizes")
    savefold.mkdir(parents=True, exist_ok=True)
    savename = Path(savefold,"ivdsag_proposal.csv")
    p = pd.read_csv("results/ivdsag/cesourceim2/val0sizes.csv")
    q = pd.read_csv("results/ivdsag/cesourceim/val0sizes.csv")
    q = pd.concat([p,q])
    q = q.reset_index()
    vec = [get_size_class(q.proposal_size[i], 2) for i in range(0, len(q))]
    q["proposal_size"]=vec
    q.to_csv(savename,index=False,sep=";")
    print("saving",savename)
    return


def get_size_class(str,n_cls):
    vec=[]
    for cls in range(0,n_cls):
        vec.append(int(str.split("tensor(")[cls+1].split('.)')[0]))
    return vec


if __name__ == "__main__":
    main()
