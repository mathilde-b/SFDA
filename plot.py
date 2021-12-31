#!/usr/bin/env python3.6

import argparse
from typing import List
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
#from scipy.interpolate import spline
from scipy.interpolate import make_interp_spline, BSpline
import os
from utils import map_
import pandas as pd

def run(args: argparse.Namespace) -> None:
    colors = ["darkorange","salmon","mediumpurple", "mediumturquoise", 'grey', 'chartreuse', 'coral','olive','mediumseagreen','darksalmon','limegreen','mediumpurple']
    #colors = ["darkgreen", "firebrick", "darkorange", "navy", "dodgerblue", 'grey', 'chartreuse', 'coral','olive','mediumseagreen','darksalmon','limegreen','mediumpurple']
    colors = ["dodgerblue","darkgreen", "firebrick", "darkorange", "grey","green"]
    #colors = ["dodgerblue","darkgreen", "firebrick"]
    #colors = ["firebrick", "darkorange","dodgerblue"]
    colors=["darkgreen","firebrick"]
    #colors.reverse()
    colors = colors+colors+colors+colors+colors+colors+colors+colors+colors
    #colors = ["c", "r", "g", "b", "m", 'grey', 'chartreuse', 'coral','olive','mediumseagreen','darksalmon','limegreen','mediumpurple']
    styles = ['--','-','-','--']
    styles = ['--','-','-','-','--','--','-','--','-','--']
    styles = ['--','--','--','-','-','-']
    styles = ['-','--','-','-','-','-']
    #styles.reverse()
    #styles = ['-','-','-','-']
    #styles = ['--', '-.', ':']
    folders = [args.base_folder + m for m in args.methods]
    print(folders)
    names: List[np.ndarray] = args.methodnames
    print(names)
    #names = [names[i] for i in range(0,len(names)) if len(os.listdir(folders[i])) > 2]
    #print(names)
    #folders = [f for f in folders if len(os.listdir(f)) > 2]
    print(folders, names)
    assert len(folders) <= len(colors)
    assert (1 + len(args.plot_column)) <= len(styles)
    paths: List[Path] = [Path(f, args.filename) for f in folders]
    arrays: List[np.ndarray] = [pd.read_csv(str(p)) for p in paths]
    #assert len(set(a.shape for a in arrays)) == 1
    #if len(arrays[0].shape) == 2:
    #    arrays = map_(lambda a: a[..., np.newaxis], arrays)
    n_epoch =min([arrays[i].shape[0] for i in range(0,len(arrays))]) 
    #n_epoch = min(n_epoch,100)
    #n_epoch = min(args.max_epc,arrays[0].shape[0])
    print(n_epoch)
    fig = plt.figure(figsize=(14, 9))
    ax = fig.gca()
    #ax.set_ylim([0,1])
    #if 'loss' in args.plot_column[0]:
    #	ax.set_ylim([0, 0.01])
    ax.set_xlim([0, n_epoch - 1])
    ax.set_xlabel("Epoch" , fontsize=22)
    #ax.set_ylabel("DSC" , fontsize=22)
    #ax.set_ylabel(Path(args.filename).stem)
    ax.grid(True, axis='y')
    #ax.set_title(f"{paths[0].stem} over epochs")

    xnew = np.linspace(0, n_epoch, n_epoch * 4)
    epcs = np.arange(n_epoch)
    for i, (a, c, p, n,ss) in enumerate(zip(arrays, colors, paths, names,styles)):
        #mean_a = a.mean(axis=1)

        # if len(args.metrics) > 1:
        #     mean_column = a[:, args.columns].mean(axis=1)
        #     ax.plot(epcs, mean_column, color=c, linestyle='-', label=f"{p.parent.name}-mean", linewidth=2)

        for s in styles[0:len(args.plot_column)]:
            #values = mean_a[..., k]
            for index,column in enumerate(args.plot_column):
                #print(column)
                #print(a)
                #epcs = a[args.epc_column]
                #plot_style = styles[i]
                plot_style = styles[index]
                print(styles,plot_style)
                metric = a[column]
                values = metric[0:n_epoch]
                xnew = np.linspace(0, n_epoch, round(n_epoch/3)) 
                spl = make_interp_spline(range(0,n_epoch),values, k=3)  # type: BSpline
                smoothed = spl(xnew)
                #smoothed = spline(epcs, values, xnew)
                #ax.plot(xnew, smoothed, linestyle=plot_style, color=c, label=n, linewidth=2)
                #ax.plot(range(0,n_epoch), values, linestyle=plot_style, color=c, label=f"{p.parent.name}-{column}", linewidth=1.5)
                plt.rcParams['font.size'] = '6'
                #ax.plot(range(0,n_epoch), values, linestyle=plot_style, color=c, label=n, linewidth=2)
                ax.plot(range(0,n_epoch), values, linestyle=plot_style, color=c, linewidth=2)
                #ax.legend(prop={"size":6},loc=1)
                plt.xticks(fontsize=18)
                plt.yticks(fontsize=18)
                #plt.yticks(np.arange(0.35,0.8,0.05))

                plt.show()
                #ax.legend()
                print(f"{Path(p).parents[0]}, metric {column} min epc {values.argmin()} : {values.min():.04f} max epc {values.argmax()} : {values.max():.04f}")
                #fig.savefig("plots/dices_"+'_'.join(args.methods)+column+'.png')
                #plt.close()
                
    if args.hline:
        for v, l, s in zip([args.hline], [args.l_line], styles):
            ax.plot([0, n_epoch], [v, v], linestyle=s, linewidth=2, color='dodgerblue', label='baseline')

    ax.legend(prop={"size":28},loc='lower right')

    fig.tight_layout()
    if args.savefig:
        namefig = "plots/"+args.data+"_"+'_'.join(args.methods)+'_'+'_'.join(args.plot_column)+'.png'
        namefigs = "plots/"+args.data+"_"+'_'.join(args.methods)+'_'+'*.png'
        fig.savefig(namefig)
        print("saving as "+ namefig)
        #print("scp -r livia:PycharmProjects/CDA/"+ namefig + ' plots/')
        print("scp -r AP50860@koios.logti.etsmtl.ca:../../../data/users/mathilde/ccnn/CDA/SFDAA/"+ namefigs + ' plots/')
        #print("scp -r AP50860@phoebe.logti.etsmtl.ca:../../../data/users/mathilde/ccnn/CDA/"+ namefig + ' plots/')
    #if not args.headless:
    plt.show()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Plot data over time')
    parser.add_argument('--folders', type=str, required=True, nargs='+', help="The folders containing the file")
    parser.add_argument('--filename', type=str, required=True)
    parser.add_argument('--columns', type=int, nargs='+', default=0, help="Which columns of the third axis to plot")
    parser.add_argument('--max_epc', type=int, nargs='+', default=0, help="epoch max")
    parser.add_argument("--savefig", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--hline", type=float, nargs='*')
    parser.add_argument("--l_line", type=str, nargs='*')
    args = parser.parse_args()

    print(args)

    return args


if __name__ == "__main__":
    plot_source_metrics = False
    #t_data = "ct"
    #t_data = "uinn"
    data="Inn"
    t_data="ivd"
    #data="Inn"
    #t_data = 'srct'
    result_fold = 'results/'+t_data+'/'
    methods = next(os.walk(result_fold))[1]
    methods = ['sfdasubj10','adaent_subj10']
    methodnames= methods


    #metrics =['val_dice_3d']
    #run(argparse.Namespace(data=t_data,base_folder=result_fold, methods= methods, savefig = True, max_epc=100, plot_column=metrics, epc_column = ["epoch"],  methodnames= methodnames,
                           #hline=0.36, filename=data+'_target_metrics.csv', hgadless=False,l_line='-'))
                           #hline=0.4675, filename='Inn_target_metrics.csv', hgadless=False,l_line='-'))
                           #hline=0.6562, filename='Inn_target_metrics.csv', hgadless=False,l_line='-'))
 
    metrics =['tra_loss_tot','val_loss_tot']
    run(argparse.Namespace(data=t_data,base_folder=result_fold, methods= methods, savefig = True, max_epc=150, plot_column=metrics, epc_column = ["epoch"],methodnames= methodnames,
                           hline='', filename=data+'_target_metrics.csv', hgadless=False,l_line='-'))
 
    metrics =['val_loss_tot']
    run(argparse.Namespace(data=t_data,base_folder=result_fold, methods= methods, savefig = True, max_epc=150, plot_column=metrics, epc_column = ["epoch"],methodnames= methodnames,
                           hline='', filename=data+'_target_metrics.csv', hgadless=False,l_line='-'))

    if 1<0 :                       
        metrics =['val_hd95_3d']
        run(argparse.Namespace(data=t_data,base_folder=result_fold, methods= methods, savefig = True, max_epc=150, plot_column=metrics, epc_column = ["epoch"],  methodnames= methodnames,
                           hline='', filename=data+'_target_metrics.csv', hgadless=False,l_line='-'))

    metrics =['val_dice']
    if plot_source_metrics:
    	run(argparse.Namespace(data=s_data,base_folder=result_fold, methods= methods, savefig = True, max_epc=150, plot_column=metrics, epc_column = ["epoch"],methodnames= methodnames,
                           hline='', filename=s_data+'_source_metrics.csv', hgadless=False,l_line='-'))

    run(argparse.Namespace(data=t_data,base_folder=result_fold, methods= methods, savefig = True, max_epc=100, plot_column=metrics, epc_column = ["epoch"],methodnames= methodnames,
                           hline='', filename=data+'_target_metrics.csv', hgadless=False,l_line='-'))

    metrics =['tra_dice']
    run(argparse.Namespace(data=t_data,base_folder=result_fold, methods= methods, savefig = True, max_epc=100, plot_column=metrics, epc_column = ["epoch"],methodnames= methodnames,
                           hline='', filename=data+'_target_metrics.csv', hgadless=False,l_line='-'))

    metrics =['tra_size_mean']
    run(argparse.Namespace(data=t_data,base_folder=result_fold, methods= methods, savefig = True, max_epc=100, plot_column=metrics, epc_column = ["epoch"],  methodnames= methodnames,
                           hline=34, filename=data+'_target_metrics.csv', hgadless=False,l_line='-'))

    metrics =['val_size_mean']
    run(argparse.Namespace(data=t_data,base_folder=result_fold, methods= methods, savefig = True, max_epc=100, plot_column=metrics, epc_column = ["epoch"],  methodnames= methodnames,
                           hline=34, filename=data+'_target_metrics.csv', hgadless=False,l_line='-')) # 118 IVD, 1039 WHS

    metrics =['val_size_mean_pos']
    run(argparse.Namespace(data=t_data,base_folder=result_fold, methods= methods, savefig = True, max_epc=100, plot_column=metrics, epc_column = ["epoch"],  methodnames= methodnames,
                           hline=415, filename=data+'_target_metrics.csv', hgadless=False,l_line='-'))


    '''
    metrics =['tra_loss_inf']
    if plot_source_metrics:
    	run(argparse.Namespace(data=s_data,base_folder=result_fold, methods= methods, savefig = True, max_epc=100, plot_column=metrics, epc_column = ["epoch"],methodnames= methodnames,
                                         hline='', filename=s_data+'_source_metrics.csv', hgadless=False,l_line='-'))

    run(argparse.Namespace(data=t_data,base_folder=result_fold, methods= methods, savefig = True, max_epc=100, plot_column=metrics, epc_column = ["epoch"],methodnames= methodnames,
                           hline='',filename=t_data+'_target_metrics.csv', hgadless=False,l_line='-'))
    '''
