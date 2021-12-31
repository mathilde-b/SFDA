CC = python
SHELL = bash
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all view plot report

CFLAGS = -O
#DEBUG = --debug

#the regex of the slices in the target dataset
#for the prostate
G_RGX = Case\d+_\d+

TT_DATA = [('IMG', nii_transform2, False), ('GT', nii_gt_transform2, False), ('GT', nii_gt_transform2, False)]
#TT_DATA = [('IMG', nii_transform2, False), ('GTNew', nii_gt_transform2, False), ('GTNew', nii_gt_transform2, False)]
L_OR = [('CrossEntropy', {'idc': [0,1], 'weights':[1,1]}, None, None, None, 1)]
NET = UNet

# the folder containing the target dataset - site A is the target dataset and site B is the source one
T_FOLD = data/prostate/SA
T_FOLD = /data/users/mathilde/ccnn/CDA/data/SAD
#T_FOLD = /data/users/mathilde/ccnn/SAML/data/SA/

#the network weights used as initialization of the adaptation
M_WEIGHTS_ul = results/prostate/cesource/last.pkl

#run the main experiment
TRN = results/prostate/sfda_notag2
#TRN = results/prostate/cesource_onsource
TRN = results/prostate/sfdanotag_notrain08 results/prostate/sfdanotag_notrain12 results/prostate/sfdanotag_notrain15
TRN =  results/prostate/sfdanotag_notrain17 results/prostate/sfdanotag_notrain20
TRN =  results/prostate/sfdanotag_notrain22 results/prostate/sfdanotag_notrain26

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-CSize.tar.gz

all: pack
plot: $(PLT)

pack: $(PACK) report
$(PACK): $(TRN) $(INF_0) $(TRN_1) $(INF_1) $(TRN_2) $(TRN_3) $(TRN_4)
	mkdir -p $(@D)
	tar cf - $^ | pigz > $@
	chmod -w $@
# tar -zc -f $@ $^  # Use if pigz is not available

# first train on the source dataset only:
results/prostate/cesourcen: OPT =  --target_losses="$(L_OR)" --target_dataset "/data/users/mathilde/ccnn/SAML/data/SB" \
	     --network UNet --model_weights="" --lr_decay 1 \
	    
# full supervision
results/prostate/fs: OPT =  --target_losses="$(L_OR)" \
	     --network UNet --lr_decay 1 \

# SFDA. Remove --saveim True --entmap --do_asd 1 --do_hd 1 to speed up
results/prostate/sfdaCase28: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 --train_grp_regex Case28*.nii \

results/prostate/sfda_proposal: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'proposal_size','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 \

results/prostate/sfda_notag2: OPT = --target_losses="[('EntKLPropNoTag', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbprednotags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate2.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 \

results/prostate/sfda_proselect2: OPT = --target_losses="[('EntKLPropSelect', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredselect','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate2.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 \

results/prostate/sfdanotag_notrain17: OPT = --target_losses="[('EntKLPropSelect', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredselect','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate2.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 --valonly --specific_subj Case17 --tta --train_case_nb=1 --target_folders="$(TT_DATA)" --lr_decay_epoch 600 --n_epoch 1800\

results/prostate/sfdanotag_notrain20: OPT = --target_losses="[('EntKLPropSelect', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredselect','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate2.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 --valonly --specific_subj Case20 --tta --train_case_nb=1 --target_folders="$(TT_DATA)" --lr_decay_epoch 600 --n_epoch 1800\

results/prostate/sfdanotag_notrain22: OPT = --target_losses="[('EntKLPropSelect', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredselect','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate2.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 --valonly --specific_subj Case22 --tta --train_case_nb=1 --target_folders="$(TT_DATA)" --lr_decay_epoch 600 --n_epoch 1800\

results/prostate/sfdanotag_notrain26: OPT = --target_losses="[('EntKLPropSelect', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredselect','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate2.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 --valonly --specific_subj Case26 --tta --train_case_nb=1 --target_folders="$(TT_DATA)" --lr_decay_epoch 600 --n_epoch 1800\

results/prostate/sfdanotag_notrain12: OPT = --target_losses="[('EntKLPropSelect', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredselect','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate2.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 --valonly --specific_subj Case02 --tta --train_case_nb=1 --target_folders="$(TT_DATA)" --lr_decay_epoch 600 --n_epoch 1800\

results/prostate/sfdanotag_notrain15: OPT = --target_losses="[('EntKLPropSelect', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredselect','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate2.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 --valonly --specific_subj Case02 --tta --train_case_nb=1 --target_folders="$(TT_DATA)" --lr_decay_epoch 600 --n_epoch 1800\

results/prostate/sfdaNew: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':',','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]" \
           --l_rate 0.000001 \

#inference mode : saves the segmentation masks for a specific model saved as pkl file (ex. "results/prostate/cesource/last.pkl" below):
results/prostate/cesourceimright: OPT =  --target_losses="$(L_OR)" \
	   --do_asd 1 --saveim True --pprint --mode makeim  --batch_size 1  --l_rate 0 --model_weights="$(M_WEIGHTS_ul)" --pprint --n_epoch 1\

results/prostate/sfdaim: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim --saveim True --entmap --do_asd 1 --do_hd 1  --batch_size 1  --l_rate 0 --model_weights="results/prostate/sfda/best_3d.pkl" --pprint --n_epoch 1 --saveim True --entmap\

results/prostate/sfda_onsource: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim --l_rate 0 --model_weights="results/prostate/sfda3/best_3d.pkl" --pprint --n_epoch 1 --val_target_folders="$(S_DATA)"  \

results/prostate/cesource_onsource: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim --l_rate 0  --pprint --n_epoch 1 --val_target_folders="$(S_DATA)"  \


$(TRN) :
	$(CC) $(CFLAGS) main_sfda.py --batch_size 8 --n_class 2 --workdir $@_tmp --target_dataset "$(T_FOLD)" \
                --wh 384 --metric_axis 1  --n_epoch 150 --dice_3d --l_rate 5e-4 --weight_decay 1e-4 --grp_regex="$(G_RGX)" --network=$(NET) --val_target_folders="$(TT_DATA)"\
                  --lr_decay 0.9  --model_weights="$(M_WEIGHTS_ul)"  --target_folders="$(TT_DATA)" $(OPT) $(DEBUG)
	mv $@_tmp $@


