# SFDA
Source-Free Domain Adaptation
by [Mathilde Bateson](https://github.com/mathilde-b), [Hoel Kervadec](https://github.com/HKervadec), [Jose Dolz](https://github.com/josedolz), Hervé Lombaert, Ismail Ben Ayed

Code of our submission at [MICCAI 2020](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_48) and its ongoing journal extension. Video of the MICCAI talk is available: 
https://www.youtube.com/watch?v=ALYaa5xrxbQ&ab_channel=MB

* [MICCAI 2020 Proceedings](https://link.springer.com/chapter/10.1007/978-3-030-59710-8_48)
* [arXiv preprint](https://arxiv.org/abs/2005.03697)

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

## Data scheme
### datasets
For instance
```
MICCAI/
    train/
        img/
            case_10_0_0.png
            ...
        gt/
            case_10_0_0.png
            ...
        random/
            ...
        ...
    val/
        img/
            case_10_0_0.png
            ...
        gt/
            case_10_0_0.png
            ...
        random/
            ...
        ...
```
The network takes png or nii files as an input. The gt folder contains gray-scale images of the ground-truth, where the gray-scale level are the number of the class (namely, 0 and 1).
### results
```
results/
    prostate/
        fs/
            best_epoch_3d/
                val/
                    case_10_0_0.png
                    ...
            iter000/
                val/
            ...
        sfda/
            ...
        best.pkl # best model saved
        metrics.csv # metrics over time, csv
        best_epoch.txt # number of the best epoch
        val_dice.npy # log of all the metric over time for each image and class
        val_dice.png # Plot over time
        ...
    whs/
        ...
archives/
    $(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-sfda.tar.gz
    $(REPO)-$(DATE)-$(HASH)-$(HOSTNAME)-prostate.tar.gz
```
## Interesting bits
The losses are defined in the [`losses.py`](losses.py) file. 

## Cool tricks
Remove all assertions from the code. Usually done after making sure it does not crash for one complete epoch:
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

