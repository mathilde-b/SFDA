CC = python
SHELL = bash
PP = PYTHONPATH="$(PYTHONPATH):."

.PHONY: all view plot report

CFLAGS = -O
#DEBUG = --debug

#the regex of the slices in the target dataset
#for the heart
G_RGX1 = slice\d+_1

TT_DATA = [('IMG', nii_transform2, False), ('GT', nii_gt_transform, False), ('GT', nii_gt_transform, False)]
L_OR = [('CrossEntropy', {'idc': [0,1,2,3,4], 'weights':[1,1,1,1,1]}, None, None, None, 1)]
NET = UNet

#the network weights used as initialization of the adaptation
M_WEIGHTS_ul = results/whs/cesource/last.pkl

#run the main experiments
TRN = results/whs/cesource results/whs/sfda


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
results/whs/cesource: OPT =  --target_losses="$(L_OR)" --target_dataset "data/mr" \
	     --network UNet --model_weights="" --lr_decay 1 \
	    
# full supervision
results/whs/fs: OPT =  --target_losses="$(L_OR)" \
	     --network UNet --model_weights="$(M_WEIGHTS_uce)" --lr_decay 1 \

# SFDA. Put --saveim False and remove --entmap and --do_hd 90 to speed up
results/whs/sfda: OPT = --target_losses="[('EntKLProp', {'curi':True,'lamb_se':1, 'lamb_consprior':1, 'ivd':False,'weights_se':[0.02, 0.27, 0.18, 0.21, 0.32],'idc_c': [1,2,3,4],'power': 1},'PredictionBounds', \
        {'margin':0,'dir':'high','idc':[1],'predcol':'meansourcewtags', 'power': 1,'fake':False, 'mode':'percentage','prop':False,'sizefile':'sizes/whs.csv'},'norm_soft_size',1)]"\
          --batch_size 22  --model_weights="$(M_WEIGHTS_uce)"  --ontest --l_rate 0.000001 --lr_decay 0.9 --weight_decay 1e-3 \

#inference mode : saves the segmentation masks for a specific model saved as pkl file (ex. "results/sa/cesource/last.pkl" below):
results/whs/cesourceim: OPT =  --target_losses="$(L_OR)" \
	   --mode makeim  --batch_size 1  --l_rate 0 --model_weights="results/whs/cesource/last.pkl" --pprint --lr_decay 1 --n_epoch 1 --saveim True\

$(TRN) :
	$(CC) $(CFLAGS) main_sfda.py --batch_size 24 --n_class 5 --workdir $@_tmp --target_dataset "data/ct" \
                --metric_axis 1  --n_epoch 100 --dice_3d --l_rate 5e-4 --weight_decay 1e-4 --grp_regex="$(G_RGX)" --network=$(NET) --val_target_folders="$(TT_DATA)"\
                  --lr_decay 0.9  --model_weights="$(M_WEIGHTS_uce)"  --target_folders="$(TT_DATA)" $(OPT) $(DEBUG)
	mv $@_tmp $@


