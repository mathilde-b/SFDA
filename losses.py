#!/usr/env/bin python3.6

from typing import List, Tuple
# from functools import reduce
from operator import add
from functools import reduce
import numpy as np
import torch
from torch import einsum
from torch import Tensor
import pandas as pd

from utils import simplex, sset, probs2one_hot
import torch.nn.modules.padding
from torch.nn import BCEWithLogitsLoss

class DiceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        #self.nd: str = kwargs["nd"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        intersection: Tensor = einsum(f"bcwh,bcwh->bc", pc, tc)
        union: Tensor = (einsum(f"bkwh->bk", pc) + einsum(f"bkwh->bk", tc))

        divided: Tensor = torch.ones_like(intersection) - (2 * intersection + 1e-10) / (union + 1e-10)

        loss = divided.mean()

        return loss

class AdaEntKLProp():
    """
    Entropy minimization with KL proportion regularisation
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]

    def __call__(self, probs: Tensor, target: Tensor, bounds) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop_mask = self.__fn__(predicted_mask,self.power).squeeze(2)
        est_prop: Tensor = self.__fn__(probs,self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:,:,0]
                bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            gt_prop = gt_prop[:,:,0]
        else:
            gt_prop: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
        if not self.curi:
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        log_est_prop: Tensor = (est_prop + 1e-10).log()

        #log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        #log_est_prop_mask: Tensor = (est_prop_mask + 1e-10).log()
        loss_cons_prior = - torch.einsum("bc,bc->", [gt_prop, log_est_prop])
        # Adding division by batch_size to normalise
        loss_cons_prior /= b

        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10

        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se*loss_se, self.lamb_consprior*loss_cons_prior,est_prop


class EntKLPropNoTag():
    """
    Entropy minimization with KL proportion regularisation
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]

    def __call__(self, probs: Tensor, target: Tensor, bounds) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop_mask = self.__fn__(predicted_mask,self.power).squeeze(2)
        est_prop: Tensor = self.__fn__(probs,self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:,:,0] 
                bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            gt_prop = gt_prop[:,:,0]
        else:
            gt_prop: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
        if not self.curi:
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        #print(est_prop_mask)
        #bool_est_prop_ispos = (est_prop_mask>0).float()
        bool_est_prop_select = (est_prop_mask>0.02).float()
        bool_fg_prop = torch.einsum("bc->b",est_prop_mask[:,1:])
        bool_fg_prop = (bool_fg_prop>0.02).float()
        bool_est_prop_select[:,0] = bool_fg_prop
        #print(bool_est_prop_ispos)
        # if the network estimates that the structure is absent, put the gt_estimation to zero.
        #gt_prop_pos = torch.einsum("bc,bc->bc", [gt_prop,bool_est_prop_ispos])
        est_prop_select = torch.einsum("bc,bc->bc", [est_prop,bool_est_prop_select])
        #gt_fg_prop = torch.einsum("bc->b",gt_prop_pos[:,1:])
        # correct the background size to get correct sum
        #gt_prop_pos[:,0] = 1 -gt_fg_prop
        #print(gt_prop,bool_est_prop_ispos,gt_prop_pos)
        log_est_prop: Tensor = (est_prop + 1e-10).log()
        #log_gt_prop: Tensor = (gt_prop_pos + 1e-10).log()
        log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        log_est_prop_mask: Tensor = (est_prop_mask + 1e-10).log()
        #print(est_prop,log_gt_prop,log_est_prop)
        #print(est_prop,est_prop_select)
        loss_cons_prior = - torch.einsum("bc,bc->", [est_prop_select, log_gt_prop])  + torch.einsum("bc,bc->", [est_prop_select, log_est_prop])
        # Adding division by batch_size to normalise 
        loss_cons_prior /= b
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        #print(bool_est_prop)
        #mask_weighted_pos = torch.einsum("bcwh,bc->bcwh", [mask,bool_est_prop])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10

        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se*loss_se, self.lamb_consprior*loss_cons_prior,est_prop 

class EntKLPropSelect():
    """
    Entropy minimization with KL proportion regularisation
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]

    def __call__(self, probs: Tensor, target: Tensor, bounds) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop_mask = self.__fn__(predicted_mask,self.power).squeeze(2)
        est_prop: Tensor = self.__fn__(probs,self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:,:,0]
                bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            gt_prop = gt_prop[:,:,0]
        else:
            gt_prop: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
        if not self.curi:
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        #print(est_prop_mask)
        #bool_est_prop_ispos = (est_prop_mask>0).float()
        bool_gt_prop_select = torch.einsum("bc->b",gt_prop)
        bool_gt_prop_select = (bool_gt_prop_select>0).float()

        est_prop_select = torch.einsum("bc,b->bc", [est_prop,bool_gt_prop_select])
        #log_est_prop: Tensor = (est_prop + 1e-10).log()
        log_est_prop: Tensor = (est_prop_select + 1e-10).log()
        #log_gt_prop: Tensor = (gt_prop_pos + 1e-10).log()
        log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        log_est_prop_mask: Tensor = (est_prop_mask + 1e-10).log()
        #print(est_prop,log_gt_prop,log_est_prop)
        #print(est_prop,est_prop_select)
        loss_cons_prior = - torch.einsum("bc,bc->", [est_prop_select, log_gt_prop])  + torch.einsum("bc,bc->", [est_prop_select, log_est_prop])
        # Adding division by batch_size to normalise
        loss_cons_prior /= b
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        #print(bool_est_prop)
        #mask_weighted_pos = torch.einsum("bcwh,bc->bcwh", [mask,bool_est_prop])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10

        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se*loss_se, self.lamb_consprior*loss_cons_prior,est_prop



class EntKLProp():
    """
    Entropy minimization with KL proportion regularisation
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]

    def __call__(self, probs: Tensor, target: Tensor, bounds) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        predicted_mask = probs2one_hot(probs).detach()
        est_prop_mask = self.__fn__(predicted_mask,self.power).squeeze(2)
        est_prop: Tensor = self.__fn__(probs,self.power)
        if self.curi:
            if self.ivd:
                bounds = bounds[:,:,0] 
                bounds= bounds.unsqueeze(2)
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            gt_prop = gt_prop[:,:,0]
        else:
            gt_prop: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
        if not self.curi:
            gt_prop = gt_prop.squeeze(2)
        est_prop = est_prop.squeeze(2)
        log_est_prop: Tensor = (est_prop + 1e-10).log()

        log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        log_est_prop_mask: Tensor = (est_prop_mask + 1e-10).log()
        #print(est_prop,log_gt_prop,log_est_prop)
        loss_cons_prior = - torch.einsum("bc,bc->", [est_prop, log_gt_prop])  + torch.einsum("bc,bc->", [est_prop, log_est_prop])
        # Adding division by batch_size to normalise 
        loss_cons_prior /= b
        log_p: Tensor = (probs + 1e-10).log()
        mask: Tensor = probs.type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss_se = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_se /= mask.sum() + 1e-10

        assert loss_se.requires_grad == probs.requires_grad  # Handle the case for validation

        return self.lamb_se*loss_se, self.lamb_consprior*loss_cons_prior,est_prop 


class SelfEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.weights: List[float] = kwargs["weights"]
        self.dtype = kwargs["dtype"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = probs[:, self.idc, ...].type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss /= mask.sum() + 1e-10

        return loss


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.weights: List[float] = kwargs["weights"]
        self.dtype = kwargs["dtype"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = target[:, self.idc, ...].type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss /= mask.sum() + 1e-10
        return loss


class BCELoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.dtype = kwargs["dtype"]
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, d_out: Tensor, label: float):
        bce_loss = torch.nn.BCEWithLogitsLoss()
        loss = bce_loss(d_out,Tensor(d_out.data.size()).fill_(label).to(d_out.device))
        return loss


class BCEGDice():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.lamb: List[int] = kwargs["lamb"]
        self.weights: List[float] = kwargs["weights"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")
    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss_gde = divided.mean()

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask_weighted = torch.einsum("bcwh,c->bcwh", [tc, Tensor(self.weights).to(tc.device)])
        loss_ce = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss_ce /= tc.sum() + 1e-10
        loss = loss_ce + self.lamb*loss_gde

        return loss



class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss
