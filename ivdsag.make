CC = python
SHELL = bash
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all view plot report

CFLAGS = -O
#CFLAGS = -m pdb
#DEBUG = --debug

#the regex of the subjects in the target dataset
#for the ivdsag
G_RGX = Subj_\d+

TT_DATA = [('Inn', png_transform, False), ('GT', gtpng_transform, False),('GT', gtpng_transform, False)]
TAUG_DATA = [('Innaug', png_transform, False), ('GTaug', gtpng_transform, False),('GTaug', gtpng_transform, False)]
S_DATA = [('Wat', png_transform, False), ('GT', gtpng_transform, False),('GT', gtpng_transform, False)]
S_DATAAUG = [('Wataug', png_transform, False), ('GTaug', gtpng_transform, False),('GTaug', gtpng_transform, False)]
L_OR = [('CrossEntropy', {'idc': [0,1], 'weights':[1,1]}, None, None, None, 1)]
NET = UNet

# the folder containing the datasets
B_FOLD = /data/users/mathilde/ccnn/CDA/data/all_sagittal/

#the network weights used as initialization of the adaptation
M_WEIGHTS_ul = results/ivdsag/cesource/last.pkl

NBSUBJ = 7

#run the main experiment
TRN = results/ivdsag/sfda_aug
TRN = results/ivdsagsag/sfdasubj$(NBSUBJ) 
#results/ivdsag/sfda_augsubj$(NBSUBJ)
#TRN = results/ivdsagsag/sfdasubj2 results/ivdsagsag/sfdasubj3 results/ivdsagsag/sfdasubj4 results/ivdsagsag/sfdasubj5
#TRN = results/ivdsag/adaent_subj$(NBSUBJ)
#TRN = results/ivdsag/cesourceaugim
TRN = results/ivdsag/cesource_onsource
TRN = results/ivdsag/sfda_zero_train_subj15
TRN = results/ivdsag/sfdaim
TRN = results/ivdsag/sfda_select2
TRN = results/ivdsag/sfdanotag_zero_train_subj15bis
TRN = results/ivdsag/cesourceim
#TRN = results/ivdsag/sfda_select

REPO = $(shell basename `git rev-parse --show-toplevel`)
DATE = $(shell date +"%y%m%d")
HASH = $(shell git rev-parse --short HEAD)
HOSTNAME = $(shell hostname)
PBASE = archives
PACK = $(PBASE)/$(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-CSize.tar.gz

all: pack
plot: $(PLT)

pack: $(PACK) report
$(PACK): $(TRN)
	mkdir -p $(@D)
	tar cf - $^ | pigz > $@
	chmod -w $@
# tar -zc -f $@ $^  # Use if pigz is not available


# first train on the source dataset only:
results/ivdsag/cesourceaug: OPT =  --target_losses="$(L_OR)" --target_folders="$(S_DATAAUG)" --val_target_folders="$(S_DATA)" \
	     --network UNet --model_weights="" --lr_decay 1 --l_rate 5e-4 \
	    
# full supervision
results/ivdsag/fs: OPT =  --target_losses="$(L_OR)" \
	     --network UNet --model_weights="$(M_WEIGHTS_uce)" --lr_decay 1 \

# SFDA. Remove --saveim True --entmap --do_asd 1 --do_hd 1 to speed up
results/ivdsag/sfda4: OPT = --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.2 \
          #--saveim True --entmap --do_asd 1 --do_hd 1  \

results/ivdsag/sfda_select: OPT = --target_losses="[('EntKLPropSelect', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredselect','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivdsag_select.csv'},'norm_soft_size',1)]" --lr_decay 0.2 \


results/ivdsag/sfdanotag: OPT = --target_losses="[('EntKLPropNoTag', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbprednotags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.2 \


results/ivdsag/sfda_zero_train_subj_gtsize: OPT = --train_case_nb=1 --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'val_gt_size','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.2 \
	--valonly --specific_subj Subj_0 --tta --target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" --lr_decay_epoch 600 --n_epoch 1800 \

results/ivdsagsag/sfda_zero_train_subj_gtsize: OPT = --train_case_nb=1 --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'val_gt_size','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.2 \
	--valonly --specific_subj Subj_0 --tta --target_folders="$(TAUG_DATA)" --val_target_folders="$(TT_DATA)" --lr_decay_epoch 600 --n_epoch 1800 \
          #--saveim True --entmap --do_asd 1 --do_hd 1  \

results/ivdsag/sfdanotag_zero_train_subj15bis: OPT =  --target_losses="[('EntKLPropSelect', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredselect','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivdsag_select.csv'},'norm_soft_size',1)]" --lr_decay 0.2 \
	--valonly --specific_subj Subj_15 --tta --train_case_nb=1 --target_folders="$(TT_DATA)" --lr_decay_epoch 600 --n_epoch 1800 --val_target_folders="$(TT_DATA)" \

results/ivdsag/sfda_zero_train_subj15: OPT = --train_case_nb=1 --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.2 \
	--valonly --specific_subj Subj_15 --tta --target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" --lr_decay_epoch 600 --n_epoch 1800 \

results/ivdsag/sfda_zero_train_subj: OPT = --train_case_nb=1 --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.2 \
	--valonly --specific_subj Subj_0 --tta --target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)" --lr_decay_epoch 600 --n_epoch 1800 \

results/ivdsagsag/sfdasubj$(NBSUBJ): OPT = --train_case_nb=$(NBSUBJ) --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.2 \
	--target_folders="$(TAUG_DATA)" --val_target_folders="$(TT_DATA)" \
          #--saveim True --entmap --do_asd 1 --do_hd 1  \

results/ivdsag/adaent_subj$(NBSUBJ): OPT =  --train_case_nb=$(NBSUBJ) --target_losses="[('AdaEntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.2 \


results/ivdsag/sfda_aug: OPT = --augment --target_losses="[('EntKLProp', {'lamb_se':1, 'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'dumbpredwtags','power': 1, 'mode':'percentage','sep':';','sizefile':'sizes/ivdsag.csv'},'norm_soft_size',1)]" --lr_decay 0.2 \
          #--saveim True --entmap --do_asd 1 --do_hd 1  \


#inference mode : saves the segmentation masks + entropy masks for a specific model saved as pkl file (ex. "$(M_WEIGHTS_ul)" below):
results/ivdsag/cesourceim: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim  --l_rate 0 --pprint --n_epoch 1 --saveim True --entmap \

results/ivdsag/cesourceaugim: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim --model_weights="results/ivdsag/cesourceaug/epoch_120.pkl"  --l_rate 0 --pprint --n_epoch 1 --saveim True --entmap \

results/ivdsag/sfda_onsource: OPT =  --target_losses="$(L_OR)" \
	   --val_target_folders="$(S_DATA)"  --mode makeim   --l_rate 0 --pprint --n_epoch 1 --model_weights="results/ivdsag/sfda4/best_3d.pkl" \

results/ivdsag/cesource_onsource: OPT =  --target_losses="$(L_OR)" \
	   --val_target_folders="$(S_DATA)"  --mode makeim   --l_rate 0 --pprint --n_epoch 1 \

results/ivdsag/sfdaim: OPT =  --target_losses="$(L_OR)" \
	   --val_target_folders="$(T_DATA)" --mode makeim  --batch_size 32  --l_rate 0 --pprint --n_epoch 1 \


$(TRN) :
	$(CC) $(CFLAGS) main_sfda.py --batch_size 20 --n_class 2 --workdir $@_tmp --target_dataset "$(B_FOLD)"  \
                --grp_regex="$(G_RGX)"  --target_folders="$(TT_DATA)" --val_target_folders="$(TT_DATA)"\
                --model_weights="$(M_WEIGHTS_ul)" --network=$(NET) \
                --lr_decay 0.9 --metric_axis 1  --n_epoch 150 --dice_3d --l_rate 3e-5 --lr_decay_epoch 50 --weight_decay 1e-4 $(OPT) $(DEBUG)\

	mv $@_tmp $@


