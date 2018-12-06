#!/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:30:40 2018

@author: aph516
"""

import pandas as pd
from subprocess import run, Popen, PIPE
from pathlib import Path

path = Path("/Users/aph516/GitHub/NAPS/")
#path = Path("C:/Users/kheyam/Documents/GitHub/NAPS/")

testset_df = pd.read_table(path/"data/testset/testset.txt", header=None, names=["ID","PDB","BMRB","Resolution","Length"])
testset_df["obs_file"] = [path/x for x in "data/testset/simplified_BMRB/"+testset_df["BMRB"].astype(str)+".txt"]
testset_df["preds_file"] = [path/x for x in "data/testset/shiftx2_results/"+testset_df["ID"]+"_"+testset_df["PDB"]+".cs"]
testset_df["out_name"] = testset_df["ID"]+"_"+testset_df["BMRB"].astype(str)
testset_df.index = testset_df["ID"]


#%%
#### Test all proteins in the testset

for i in testset_df.index:
    print(testset_df.loc[i, "obs_file"])
    args = [(path/"python/NAPS.py").as_posix(), testset_df.loc[i, "obs_file"].as_posix(), testset_df.loc[i, "preds_file"].as_posix(), 
            (path/("output/testset/"+testset_df.loc[i, "out_name"]+".txt")).as_posix()]
    run(args)

#%%
#### Check out accurate the assignments are
def check_assignment_accuracy(data_dir, ranks=[1], prefix=""):
    # Nb. make sure data_dir ends with a forward slash

    assigns = None
    for i in testset_df.index:
        tmp = pd.read_csv(data_dir/(prefix+testset_df.loc[i, "out_name"]+".txt"), sep="\t", index_col=0)
        if "Rank" in tmp.columns:
            tmp = tmp.loc[tmp["Rank"].isin(ranks),:]
        tmp["ID"] = i
        tmp = tmp[["ID","Res_N","Res_type","Res_name","SS_name","Dummy_SS","Dummy_res"]]
        # Convert Res_N column to integer
        tmp["Res_N"] = tmp["Res_N"].fillna(-999)
        tmp["Res_N"] = tmp["Res_N"].astype(int)
        # Cysteines in disulphide bridges are often named B in this dataset. 
        # We don't need to know this, so change to C
        mask = tmp["Res_type"]=="B"
        tmp.loc[mask,"Res_type"] = "C"
        tmp.loc[mask, "Res_name"] = tmp.loc[mask,"Res_N"].astype(str)+tmp.loc[mask, "Res_type"]
        # Add a column saying whether a match exists for the spin system 
        # (ie whether a correct assignment is possible)
        tmp["SS_in_pred"] = tmp["SS_name"].isin(tmp["Res_name"])
        tmp["Pred_in_SS"] = tmp["Res_name"].isin(tmp["SS_name"])
        
        if type(assigns)!=type(tmp):
            assigns = tmp
        else:
            assigns = pd.concat([assigns, tmp], ignore_index=True)
    
    # Determine which spin systems were correctly assigned
    assigns["Correct"] = False
    assigns["Status"] = ""
    
    mask = assigns["SS_in_pred"] & ~assigns["Dummy_SS"]
    assigns.loc[mask & (assigns["SS_name"]==assigns["Res_name"]), "Correct"] = True
    assigns.loc[mask & (assigns["SS_name"]==assigns["Res_name"]), "Status"] = "Correctly assigned"
    assigns.loc[mask & (assigns["SS_name"]!=assigns["Res_name"]), "Status"] = "Misassigned"
    assigns.loc[mask & assigns["Dummy_res"], "Status"] = "Wrongly unassigned"
    
    mask = ~assigns["SS_in_pred"] & ~assigns["Dummy_SS"]
    assigns.loc[mask & (assigns["SS_name"]!=assigns["Res_name"]), "Status"] = "Wrongly assigned"
    assigns.loc[mask & assigns["Dummy_res"], "Correct"] = True
    assigns.loc[mask & assigns["Dummy_res"], "Status"] = "Correctly unassigned"
    
    assigns.loc[assigns["Dummy_SS"], "Status"] = "Dummy SS"
    
    assigns.groupby(["ID","Status"])["Status"].count()
    
    summary = assigns.groupby(["ID","Status"])["Status"].count().unstack(1)
    summary = summary.fillna(0).astype(int)
    summary["N"] = summary.apply(sum, axis=1)
    summary["N_SS"] = summary["N"] - summary["Dummy SS"]    
    summary["Pc_correct"] = (summary["Correctly assigned"]+summary["Correctly unassigned"]) / summary["N_SS"]
    
    return([assigns, summary])
    
assigns, summary = check_assignment_accuracy(path/"output/testset/")
#%%
#### Test effect of accounting for correlated errors
for i in testset_df.index:
    print(testset_df.loc[i, "obs_file"])    
    args = [(path/"python/NAPS.py").as_posix(), testset_df.loc[i, "obs_file"].as_posix(), testset_df.loc[i, "preds_file"].as_posix(), 
            (path/("output/delta_correlation/"+testset_df.loc[i, "out_name"]+".txt")).as_posix(), "--delta_correlation"]
    run(args)
    
assigns2, summary2 = check_assignment_accuracy(path/"output/delta_correlation/")

#%%
#### Test alternative assignments
for i in testset_df.index:
    print(testset_df.loc[i, "out_name"])    
    args = [(path/"python/NAPS.py").as_posix(), testset_df.loc[i, "obs_file"].as_posix(), testset_df.loc[i, "preds_file"].as_posix(), 
            (path/("output/testset/alt_"+testset_df.loc[i, "out_name"]+".txt")).as_posix(), "--delta_correlation", "--alt_assignments", "2"]
    p = Popen(args)
    result = p.communicate()
    
assigns3, summary3 = check_assignment_accuracy(path/"output/testset/", prefix="alt_")
