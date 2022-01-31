# Source-Free Domain Adaptation
We introduce Source-Free Domain Adaptation for Image Segmentation.

[Mathilde Bateson](https://github.com/mathilde-b), [Hoel Kervadec](https://github.com/HKervadec), [Jose Dolz](https://github.com/josedolz), Hervé Lombaert, Ismail Ben Ayed @ETS Montréal

Code of our submission at [MICCAI 2020](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_48) and its ongoing journal extension. Video of the MICCAI talk is available: 
https://www.youtube.com/watch?v=ALYaa5xrxbQ&ab_channel=MB

* [MICCAI 2020 Proceedings](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_48)
* [arXiv preprint](https://arxiv.org/abs/2005.03697)

Please cite our paper if you find it useful for your research.

```

@inproceedings{BatesonSFDA,
	Author = {Bateson, Mathilde and Kervadec, Hoel and Dolz, Jose and Lombaert, Herv{\'e} and Ben Ayed, Ismail},
	Booktitle = {Medical Image Computing and Computer Assisted Intervention -- MICCAI 2020},
	Pages = {490--499},
	Publisher = {Springer International Publishing},
	Title = {Source-Relaxed Domain Adaptation for Image Segmentation},
	Year = {2020}
    Address = {Cham}}

```

![Visual comparison](seg_pro3.png)


## Requirements
Non-exhaustive list:
* python3.6+
* Pytorch 1.0
* nibabel
* Scipy
* NumPy
* Matplotlib
* Scikit-image
* zsh
* tqdm
* pandas
* scikit-image

## Data scheme
### datasets
For instance
```
data
    prostate_source/
	    train/
		IMG/
		    Case10_0.png
		    ...
		GT/
		    Case10_0.png
		    ...
		...
	    val/
		IMG/
		    Case11_0.png
		    ...
		GT/
		    Case11_0.png
		    ...
		...
    prostate_target/
	    train/
		IMG/
		    Case10_0.png
		    ...
		GT/
		    Case10_0.png
		    ...
		...
	    val/
		IMG/
		    Case11_0.png
		    ...
		GT/
		    Case11_0.png
		    ...
		...
```
The network takes png or nii files as an input. The gt folder contains gray-scale images of the ground-truth, where the gray-scale level is the number of the class (0,1,...K).

### Class-ratio (sizes) prior
The class-ratio prior is estimated from anatomical knowledge for each application. In our implementation, is it estimated for each slice in the target domain training and validation sets. It estimated once, before the start of the adaptation phase, and saved in a csv file. 

Scheme
```
sizes/
    prostate.csv
    whs.csv
    ivd.csv
```
The size csv file should be organized as follows:

| val_ids | dumbpredwtags
| ------------- | ------------- |
| Case00_0.nii | [Estimated_Size_class0, Estimated_Size_class1, ..., Estimated_Size_classk]

Sample from sizes/prostate.csv :

| val_ids  | val_gt_size | dumbpredwtags
| ------------- | ------------- |------------- |
| Case00_0.nii  | [147398.0, 827.0]  | [140225, 6905]
| Case00_1.nii  | [147080.0, 1145.0]  | [140225, 6905]
| Case00_14.nii  | [148225.0, 0.0] | [148225, 0]

NB 1 : there should be no overlap between names of the slices in the training and validation sets (Case00_0.nii,...).

NB 2: in our implementation, the csv file contains the sizes priors in pixels, and the KL Divergence loss divides the size in pixels by (w*h) the height and weight of the slice, to obtain the class-ratio prior.

NB 3: Estimated_Size_class0 + Estimated_Size_class1 + ... + Estimated_Size_classk = w*h

NB 4: the true val_gt_size is unknown, so it is not directly used in our proposed SFDA. However, in our framework an image-level annotation is available for the target training dataset: the "Tag" of each class k, indicating the presence or absence of class k in the slice. Therefore, Estimated_Size_classk=0 if val_gt_size_k = 0 and Estimated_Size_classk>0 if val_gt_size_k > 0

NB 5: To have an idea of the capacity of the SFDA model in the ideal case where the ground truth class-ratio prior is known, it is useful to run the upper bound model SFDA_TrueSize choosing the column "val_gt_size" instead of "dumbpredwtags". This can be changed in the makefile :

```
results/sa/SFDA_TrueSize: OPT = --target_losses="[('EntKLProp', {'lamb_se':1,'lamb_consprior':1,'ivd':True,'weights_se':[0.1,0.9],'idc_c': [1],'curi':True,'power': 1},'PredictionBounds', \
      {'margin':0,'dir':'high','idc':[0,1],'predcol':'val_gt_size','power': 1, 'mode':'percentage','sizefile':'sizes/prostate.csv'},'norm_soft_size',1)]" \
           --val_target_folders="$(TT_DATA)"  --l_rate 0.000001 --n_epoch 100 --lr_decay 0.9 --batch_size 10 --target_folders="$(TT_DATA)" --model_weights="$(M_WEIGHTS_ul)" \
```

NB 6 : If you change the name of the columns (val_ids, dumbpredwtags) in the size file, you should change them in the [`bounds.py`](bounds.py) file as well as in the [`prostate.make`](prostate.make) makefile. 

### results
```
results/
    prostate/
        fs/
            best_epoch_3d/
                val/
                    Case11_0.png
                    ...
            iter000/
                val/
            ...
        sfda/
            ...
        params.txt # saves all the argparse parameters of the model 
	best_3d.pkl # best model saved
	last.pkl # last epoch
        IMG_target_metrics.csv # metrics over time, csv
        3dbestepoch.txt # number and 3D Dice of the best epoch 
        ...
    whs/
        ...
archives/
    $(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-sfda.tar.gz
    $(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-prostate.tar.gz
```
## Interesting bits
The losses are defined in the [`losses.py`](losses.py) file. 


## Running our main experiment
Once you have downladed the data and organized it such as in the scheme above, run the main experiment as follows:
```
make -f prostate.make 
```
This will first run the source training model, which will be saves in results/cesource, and then the SFDA model, which will be saved in results/sfda.

## Cool tricks
Remove all assertions from the code to speed up. Usually done after making sure it does not crash for one complete epoch:
```
make -f prostate.make <anything really> CFLAGS=-O
```

Use a specific python executable:
```
make -f prostate.make <super target> CC=/path/to/the/executable
```

Train for only 5 epochs, with a dummy network, and only 10 images per data loader. Useful for debugging:
```
make -f prostate.make <anything really> NET=Dimwit EPC=5 DEBUG=--debug
```

Rebuild everything even if already exist:
```
make -f prostate.make <a> -B
```

Only print the commands that will be run (useful to check recipes are properly defined):
```
make -f prostate.make <a> -n
```


## Related Implementation and Dataset
* [Mathilde Bateson](https://github.com/mathilde-b), [Hoel Kervadec](https://github.com/HKervadec), [Jose Dolz](https://github.com/josedolz), Hervé Lombaert, Ismail Ben Ayed. Constrained Domain Adaptation for Image Segmentation. In IEEE Transactions on Medical Imaging, 2021. [[paper]](https://ieeexplore.ieee.org/document/9382339) [[implementation]](https://github.com/mathilde-b/CDA) 
* [Hoel Kervadec](https://github.com/HKervadec), [Jose Dolz](https://github.com/josedolz), Meng Tang, Eric Granger, Yuri Boykov, Ismail Ben Ayed. Constrained-CNN losses for weakly supervised segmentation. In Medical Image Analysis, 2019. [[paper]](https://www.sciencedirect.com/science/article/pii/S1361841518306145?via%3Dihub) [[code]](https://github.com/LIVIAETS/SizeLoss_WSS)
* Prostate Dataset and details: https://liuquande.github.io/SAML/. The SA site dataset was used a target domain, the SB site was used as source domain. For both datasets, we use 20 scans for training, and the remaining 10 scans for validation.
* Heart Dataset and details: We used the preprocessed dataset from Dou et al. : https://github.com/carrenD/Medical-Cross-Modality-Domain-Adaptation. The data is in tfs records, it should be transformed to nii or png before running the makefile.
* Spine Dataset and details: https://ivdm3seg.weebly.com/ . From the original coronal view, we transposed the slices to transverse view in our experiments. We set the water modality (Wat) as the source and the in-phase (IP) modality as the target domain. From this dataset, 13 scans are used for training, and the remaining 3 scans for validation.


## Note
The model and code are available for non-commercial research purposes only.
 
