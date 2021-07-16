#!/usr/bin/env python3.6

from typing import Any, List

import torch
from torch import Tensor
import pandas as pd
from utils import eq


class ConstantBounds():
    def __init__(self, **kwargs):
        self.C: int = kwargs['C']
        self.const: Tensor = torch.zeros((self.C, 1, 2), dtype=torch.float32)

        for i, (low, high) in kwargs['values'].items():
            self.const[i, 0, 0] = low
            self.const[i, 0, 1] = high

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        return self.const



class PredictionBoundswTags():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        self.mode: str = kwargs['mode']
        self.sizefile: float = kwargs['sizefile']
        self.sizes = pd.read_csv(self.sizefile,index_col=0)
        self.prop: bool = kwargs['prop']
        self.fake: bool = kwargs['fake']
        self.final: bool = kwargs['final']
        self.tags: bool = kwargs['tags']
        #self.orsizes = pd.read_csv('results/trainval_size_Inn/ls_winr2_40/trainvalreg_metrics_C2.csv',index_col=0)
        
        # Do it on CPU to avoid annoying the main loop
        #self.net: Callable[Tensor, [Tensor]] = torch.load(kwargs['net'], map_location='cpu')

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        #with torch.no_grad():
        #    value: Tensor = self.net(image[None, ...])[0].type(torch.float32)[..., None]  # cwh and not bcwh
        c,w,h=target.shape
        if self.prop:
            pred_size_col = 'val_pred_prop'
        else:
            pred_size_col = 'val_pred_size'
        gt_size_col = 'val_gt_size'
        #pred_size_col = 'val_gt_prop'
        try:
            #print(filename,self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0])
            value = eval(self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0])
        except:
              print("An exception occurred", filename)
        #value = self.sizes.at[filename,pred_size_col]
        #print(sum(self.sizes.val_ids == filename),"should be 1")
        #print(len(self.sizes.loc[self.sizes.val_ids == filename, pred_size_col]), "should be 1")
        value = eval(self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0])
        gt_value = eval(self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0])
        bool_isnotzero= float(gt_value!=0)
        print('value',value)
        value = value*bool_isnotzero
        if self.prop:
            value = [i * 65536 for i in value]
        #value = self.sizes.at[filename, "val_gt_size"]
        #value_gt = self.orsizes.at[filename,"val_gt_size"]
        #print(value,"this is value",value_gt,"this is value_gt")
        value = torch.tensor([value])
        value = torch.transpose(value, 0, 1)
        #print(value.shape, "value.shape")

        with_margin: Tensor = torch.stack([value, value], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        res = torch.max(with_margin, torch.zeros(*value.shape, 2)).type(torch.float32)
        return res


class PredictionBoundswClasses():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        self.dir: str = kwargs['dir']
        self.mode = "percentage" 
        self.sizefile: float = kwargs['sizefile']
        self.sizes = pd.read_csv(self.sizefile)
        self.predcol: bool = kwargs['predcol']
        self.classcol: bool = kwargs['classcol']
        self.classfile: float = kwargs['classfile']
        self.classes = pd.read_csv(self.classfile)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        c, w, h=target.shape
        # get classes for specific slice
        #print(self.classes.columns) 
        cols = list(self.classes.columns)[1:]
        #print(cols)
        #print(filename,self.classes.loc[self.classes.ids == filename])
        #print(filename,self.classes.loc[self.classes.ids == filename,cols].values[0])
        try:
            #cls = eval(self.classes.loc[self.classes.ids == filename, self.classcol].values[0])
            cls = self.classes.loc[self.classes.ids == filename,cols].values[0]
        except:
            print('not eval',filename)
            #print(self.classes.val_ids)
            #cls should be smthng like : NA 1 NA 1 NA
            cls = self.classes.loc[self.classes.ids == filename,cols].values[0]
        #print(value.shape, "value.shape")
        cls = torch.tensor([cls]).squeeze(0).type(torch.float32)
        
        # get the sizes for specific slice 
        pred_size_col = self.predcol
        try:
            #print(filename)
            #print(filename,self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0])
            value = eval(self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0])
        except:
            print('not eval sizes')
            #print(self.sizes.val_ids)
            print(filename)
            #print(self.sizes.val_ids == filename)
            value = self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0]
        value = torch.tensor([value]).squeeze(0)
        margin: Tensor
        if self.mode == "percentage":
            margin = value * self.margin
        else:
            raise ValueError("mode")

        if self.dir == "both":
            with_margin: Tensor = torch.stack([value - margin, value + margin], dim=-1)
        elif self.dir == "high":
            with_margin: Tensor = torch.stack([value, value + margin], dim=-1)
        elif self.dir == "low":
            with_margin: Tensor = torch.stack([value-margin, value], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        res = torch.max(with_margin, torch.zeros(*value.shape, 2).type(torch.long).type(torch.float32)).type(torch.float32)
        #res = torch.max(with_margin, torch.zeros(*value.shape, 2).type(torch.long))
        #print(res)
        res = torch.einsum("cm,c->cm", [res,cls]).type(torch.float32)
        #print(res)
        return res


class PredictionBounds():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        self.dir: str = kwargs['dir']
        self.mode = "percentage" 
        self.sizefile: float = kwargs['sizefile']
        self.sizes = pd.read_csv(self.sizefile)
        self.predcol: bool = kwargs['predcol']
        
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        #with torch.no_grad():
        #    value: Tensor = self.net(image[None, ...])[0].type(torch.float32)[..., None]  # cwh and not bcwh
        c,w,h=target.shape
        '''
        if self.prop:
            pred_size_col = 'val_pred_prop'
        else:
            pred_size_col = 'val_pred_size'
        if self.final:    
            pred_size_col = 'val_finalpred_size'
        if self.tags:
            pred_size_col = 'val_tagpred_size'
        if self.fake:
            pred_size_col = 'val_gt_size'
        '''    
        pred_size_col = self.predcol
        #print(self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0])
        try:
            #print(filename)
            #print(filename,self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0])
            value = eval(self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0])
        except:
            print('not eval')
            print(self.sizes[pred_size_col])
            print(self.sizes.columns)
            print(filename)
            #print(self.sizes.val_ids == filename)
            #print(self.sizes.loc[self.sizes.val_ids == filename, pred_size_col])
            value = self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0]
            #value = torch.tensor([value]).unsqueeze(1)
            #value = torch.transpose(value, 0, 1)
        #print(type(eval(value)),'value')
        #value = self.sizes.at[filename,pred_size_col]
        #print(sum(self.sizes.val_ids == filename),"should be 1")
        #print(len(self.sizes.loc[self.sizes.val_ids == filename, pred_size_col]), "should be 1")
        #value = self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0]
        #if self.prop:
        #    value = [i * 65536 for i in value]
        #value = self.sizes.at[filename, "val_gt_size"]
        #value_gt = self.orsizes.at[filename,"val_gt_size"]
        #print(value,"this is value",value_gt,"this is value_gt")
        #print(value)
        #value = eval(self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0])
        #value = torch.tensor([value]).unsqueeze(1)
        #print(value)
        #value = torch.transpose(value, 0, 1)
        #print(value.shape, "value.shape")
        value = torch.tensor([value]).squeeze(0)
        #with_margin: Tensor = torch.stack([value, value], dim=-1)
        #assert with_margin.shape == (*value.shape, 2), with_margin.shape

        margin: Tensor
        if self.mode == "percentage":
            margin = value * self.margin
        else:
            raise ValueError("mode")

        if self.dir == "both":
            with_margin: Tensor = torch.stack([value - margin, value + margin], dim=-1)
        elif self.dir == "high":
            with_margin: Tensor = torch.stack([value, value + margin], dim=-1)
        elif self.dir == "low":
            with_margin: Tensor = torch.stack([value-margin, value], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        #res = torch.max(with_margin, torch.zeros(*value.shape, 2)).type(torch.float32)
        res = torch.max(with_margin, torch.zeros(*value.shape, 2).type(torch.long)).type(torch.float32)
        #print(res.shape,'res.shape')
        return res


class TagBounds(ConstantBounds):
    def __init__(self, **kwargs):
        super().__init__(C=kwargs['C'], values=kwargs["values"])  # We use it as a dummy

        self.idc: List[int] = kwargs['idc']
        self.idc_mask: Tensor = torch.zeros(self.C, dtype=torch.uint8)  # Useful to mask the class booleans
        self.idc_mask[self.idc] = 1

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        positive_class: Tensor = torch.einsum("cwh->c", [target]) > 0
        #weak_positive_class: Tensor = torch.einsum("cwh->c", [weak_target]) > 0
        c,w,h = target.shape
        masked_positive: Tensor = torch.einsum("c,c->c", [positive_class, self.idc_mask]).type(torch.float32)  # Keep only the idc
        #masked_weak: Tensor = torch.einsum("c,c->c", [weak_positive_class, self.idc_mask]).type(torch.float32)
        #assert eq(masked_positive, masked_weak), f"Unconsistent tags between labels: {filename}"
        if masked_positive.sum() ==0: # only background
            print("negative image",filename)
            res =  torch.zeros((self.C, 1, 2), dtype=torch.float32)
            res[0,0,1] = w*h
            res[0,0,0] = w*h
        else:    
            #print("positive image",filename)
            res: Tensor = super().__call__(image, target, weak_target, filename)
            res = torch.einsum("cki,c->cki", [res, masked_positive])
        #print(res)
        return res



class TagBoundsPos(ConstantBounds):
    def __init__(self, **kwargs):
        super().__init__(C=kwargs['C'], values=kwargs["values"])  # We use it as a dummy

        self.idc: List[int] = kwargs['idc']
        self.idc_mask: Tensor = torch.zeros(self.C, dtype=torch.uint8)  # Useful to mask the class booleans
        self.idc_mask[self.idc] = 1

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        positive_class: Tensor = torch.einsum("cwh->c", [target]) > 0
        weak_positive_class: Tensor = torch.einsum("cwh->c", [weak_target]) > 0

        masked_positive: Tensor = torch.einsum("c,c->c", [positive_class, self.idc_mask]).type(torch.float32)  # Keep only the idc
        masked_weak: Tensor = torch.einsum("c,c->c", [weak_positive_class, self.idc_mask]).type(torch.float32)
        #assert eq(masked_positive, masked_weak), f"Unconsistent tags between labels: {filename}"

        res: Tensor = super().__call__(image, target, weak_target, filename)
        masked_res = torch.einsum("cki,c->cki", [res, masked_positive])

        return masked_res


class PreciseBounds():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        self.mode: str = kwargs['mode']
        self.namefun: str = kwargs['fn']
        self.power: int = kwargs['power']
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        if self.namefun == "norm_soft_size":
            value: Tensor = self.__fn__(target[None, ...].type(torch.float32), self.power)[0].type(torch.float32)  # cwh and not bcwh
        else:
            #value: Tensor = self.__fn__(target[None, ...])[0].type(torch.float32)  # cwh and not bcwh
            value: Tensor = self.__fn__(target[None, ...].type(torch.float32), self.power)[0].type(torch.float32)  # cwh and not bcwh
        margin: Tensor
        if self.mode == "percentage":
            margin = value * self.margin
        elif self.mode == "abs":
            margin = torch.ones_like(value) * self.margin
        else:
            raise ValueError("mode")

        with_margin: Tensor = torch.stack([value - margin, value + margin], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        res = torch.max(with_margin, torch.zeros(*value.shape, 2)).type(torch.float32)
        #print(res.shape)
        return res

class PreciseBoundsOnWeak():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        self.mode: str = kwargs['mode']
        self.namefun: str = kwargs['fn']
        self.power: int = kwargs['power']
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        if self.namefun == "norm_soft_size":
            value: Tensor = self.__fn__(weak_target[None, ...].type(torch.float32), self.power)[0].type(torch.float32)  # cwh and not bcwh
        else:
            value: Tensor = self.__fn__(weak_target[None, ...])[0].type(torch.float32)  # cwh and not bcwh
        margin: Tensor
        if self.mode == "percentage":
            margin = value * self.margin
        elif self.mode == "abs":
            margin = torch.ones_like(value) * self.margin
        else:
            raise ValueError("mode")

        with_margin: Tensor = torch.stack([value - margin, value + margin], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        res = torch.max(with_margin, torch.zeros(*value.shape, 2)).type(torch.float32)
        return res


class PreciseBoundsOnWeakWTags():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        self.mode: str = kwargs['mode']
        self.namefun: str = kwargs['fn']
        self.power: int = kwargs['power']
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        if self.namefun == "norm_soft_size":
            value: Tensor = self.__fn__(weak_target[None, ...].type(torch.float32), self.power)[0].type(torch.float32)  # cwh and not bcwh
            value_gt: Tensor = self.__fn__(target[None, ...].type(torch.float32), self.power)[0].type(torch.float32)  # cwh and not bcwh
        else:
            #value: Tensor = self.__fn__(weak_target[None, ...])[0].type(torch.float32)  # cwh and not bcwh
            #value_gt: Tensor = self.__fn__(target[None, ...])[0].type(torch.float32)  # cwh and not bcwh
            value: Tensor = self.__fn__(weak_target[None, ...].type(torch.float32), self.power)[0].type(torch.float32)  # cwh and not bcwh
            value_gt: Tensor = self.__fn__(target[None, ...].type(torch.float32), self.power)[0].type(torch.float32)  # cwh and not bcwh
        if value_gt[1]*0 == value_gt[1]:
            #print("gt size is 0")
            value = value_gt
        else:
            if value[1]*0 == value[1]: 
                #print("inf size is 0")
                value[1] = torch.ones_like(value[1]).type(torch.float32)
            #else:
                #print("both are >0")
        margin: Tensor
        if self.mode == "percentage":
            margin = value * self.margin
        elif self.mode == "abs":
            margin = torch.ones_like(value) * self.margin
        else:
            raise ValueError("mode")

        with_margin: Tensor = torch.stack([value - margin, value + margin], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        res = torch.max(with_margin, torch.zeros(*value.shape, 2)).type(torch.float32)
        #print(res.shape)
        return res

class PreciseTags(PreciseBounds):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.neg_value: List = kwargs['neg_value']

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        positive_class: Tensor = torch.einsum("cwh->c", [target]) > 0

        res = super().__call__(image, target, weak_target, filename)

        masked = res[...]
        masked[positive_class == 0] = torch.Tensor(self.neg_value)

        return masked


class BoxBounds():
    def __init__(self, **kwargs):
        self.margins: Tensor = torch.Tensor(kwargs['margins'])
        assert len(self.margins) == 2
        assert self.margins[0] <= self.margins[1]

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        c = len(weak_target)
        box_sizes: Tensor = torch.einsum("cwh->c", [weak_target])[..., None].type(torch.float32)

        bounds: Tensor = box_sizes * self.margins

        res = bounds[:, None, :]
        assert res.shape == (c, 1, 2)
        assert (res[..., 0] <= res[..., 1]).all()
        return res


'''class PredictionBounds():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        self.mode: str = kwargs['mode']
        self.sizefile: float = kwargs['sizefile']
        self.sizes = pd.read_csv(self.sizefile,index_col=0)
        # Do it on CPU to avoid annoying the main loop
        #self.net: Callable[Tensor, [Tensor]] = torch.load(kwargs['net'], map_location='cpu')

        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        #with torch.no_grad():
        #    value: Tensor = self.net(image[None, ...])[0].type(torch.float32)[..., None]  # cwh and not bcwh
        c,w,h=target.shape
        pred_size_col = 'val_pred_size'
        value = self.sizes.at[filename,pred_size_col]
        value = torch.tensor([w*h - value, value]).unsqueeze(1)
        margin: Tensor
        if self.mode == "percentage":
            margin = value * self.margin
        elif self.mode == "abs":
            margin = torch.ones_like(value) * self.margin
        else:
            raise ValueError("mode")

        with_margin: Tensor = torch.stack([value - margin, value + margin], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        res = torch.max(with_margin, torch.zeros(*value.shape, 2)).type(torch.float32)
        return res

'''

def CheckBounds(**kwargs):
        sizefile: float = kwargs['sizefile']
        sizes = pd.read_csv(sizefile)
        predcol: str = kwargs['predcol']
        #print(predcol, 'pred_size_col')
        #print(sizes.columns, 'self.sizes.columns')
        if predcol in sizes.columns:
            return True
        else:
            print('size pred not in file')
            return False



class PredictionBoundsold():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        #self.dir: float = kwargs['dir']
        self.mode: str = kwargs['mode']
        self.sizefile: float = kwargs['sizefile']
        #self.sizes = pd.read_csv(self.sizefile,index_col=0)
        self.sizes = pd.read_csv(self.sizefile)
        self.idc: List[int] = kwargs['idc']
        self.predcol: str = kwargs['predcol']
        #self.indb: List[int] = kwargs['indb']
        # Do it on CPU to avoid annoying the main loop
        #self.net: Callable[Tensor, [Tensor]] = torch.load(kwargs['net'], map_location='cpu')

        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        c, w, h = target.shape
        #pred_size_col = 'val_finalpredsize'
        pred_size_col = self.predcol
        #pred_size_col = 'val_gt_size'
        #print(self.sizes.head())
        #print(self.sizes[pred_size_col])
        #print(filename)
        #value_vec = eval(self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0])
        #print(self.sizes.columns)
        value= self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0]
        #print(value)
        #value = [eval(f)[self.idc] for f in value]
        value = eval(value)
        print(value)
        #print(self.sizes[self.sizes.val_ids == filename])
        #value = self.sizes.at[filename,pred_size_col]
        #print(value,'value')
        #value = torch.tensor([w*h - value, value]).unsqueeze(1)
        #value = torch.tensor(value_vec).unsqueeze(1)
        margin: Tensor
        if self.mode == "percentage":
            margin = value * self.margin
        elif self.mode == "abs":
            margin = torch.ones_like(value) * self.margin
        else:
            raise ValueError("mode")
        #print('sizeloss',filename,value)
        with_margin: Tensor = torch.stack([value - margin, value + margin], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        res = torch.max(with_margin.type(torch.float32), torch.zeros(*value.shape, 2)).type(torch.float32)
        return res


class PredictionBoundswTagsold():
    def __init__(self, **kwargs):
        self.margin: float = kwargs['margin']
        #self.dir: float = kwargs['dir']
        self.mode: str = kwargs['mode']
        self.sizefile: float = kwargs['sizefile']
        self.sizes = pd.read_csv(self.sizefile,index_col=0)
        self.idc: List[int] = kwargs['idc']
        self.predcol: str = kwargs['predcol']
        #self.indb: List[int] = kwargs['indb']
        # Do it on CPU to avoid annoying the main loop
        #self.net: Callable[Tensor, [Tensor]] = torch.load(kwargs['net'], map_location='cpu')

        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        c, w, h = target.shape
        pred_size_col = 'val_pred_size'
        pred_size_col = 'val_dumbpred'
        pred_size_col = self.predcol
        #pred_size_col = 'val_dumbpredmax'
        gt_size_col = "val_gt_size"
        #value = eval(self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0])*256*256
        #value_gt = eval(self.sizes.loc[self.sizes.val_ids == filename, gt_size_col].values[0])*256*256
        value = self.sizes.at[filename,pred_size_col]
        value_gt = self.sizes.at[filename,gt_size_col]
       #print(value_gt)
        #if value_gt*0 == value_gt:
        if value_gt == 0.0:
            value = value_gt
        #print(len(value))    
        value = torch.tensor([w*h - value, value]).unsqueeze(1)
        margin: Tensor
        if value_gt == 0.0:
        #if value_gt*0 == value_gt:
        #    print("neg im")
            margin = torch.zeros_like(value)
        else:
            if self.mode == "percentage":
                margin = value * self.margin
            elif self.mode == "abs":
                margin = torch.ones_like(value) * self.margin
            else:
                raise ValueError("mode")
        #print('sizeloss',filename,value)
        with_margin: Tensor = torch.stack([value - margin, value + margin], dim=-1)
        assert with_margin.shape == (*value.shape, 2), with_margin.shape

        res = torch.max(with_margin.type(torch.float32), torch.zeros(*value.shape, 2)).type(torch.float32)
        return res


class PredictionValues():
    def __init__(self, **kwargs):
        self.sizefile: float = kwargs['sizefile']
        self.sizes = pd.read_csv(self.sizefile,index_col=0)

    def __call__(self, image: Tensor, target: Tensor, weak_target: Tensor, filename: str) -> Tensor:
        #print(weak_target.shape,'weak shape',filename)
        c,w,h=target.shape
        pred_size_col = 'val_gt_prop'
        gt_size_col="val_gt_prop"
        #print(self.sizes.at[filename,pred_size_col])
        value = eval(self.sizes.loc[self.sizes.val_ids == filename, pred_size_col].values[0])*256*256
        value_gt = eval(self.sizes.loc[self.sizes.val_ids == filename, gt_size_col].values[0])*256*256
        #value = self.sizes.at[filename,pred_size_col]
        value_gt = self.sizes.at[filename,gt_size_col]
        if value_gt ==0:
            value = value_gt
        #print("proploss",filename,value)
        value = value /(w*h)
        res = torch.tensor([1-value, value])        
        #print("res shape",res.shape)
        return res

