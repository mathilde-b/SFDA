#!/usr/env/bin python3.6

from typing import List, Tuple
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



class EntKLProp():
    """
    CE between proportions
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_se: float = kwargs["lamb_se"]
        self.lamb_conspred: float = kwargs["lamb_conspred"]
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
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        #assert simplex(probs) and simplex(target)

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
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        #assert simplex(probs) and simplex(target)

        log_p: Tensor = (probs[:, self.idc, ...] + 1e-10).log()
        mask: Tensor = target[:, self.idc, ...].type((torch.float32))
        mask_weighted = torch.einsum("bcwh,c->bcwh", [mask, Tensor(self.weights).to(mask.device)])
        loss = - torch.einsum("bcwh,bcwh->", [mask_weighted, log_p])
        loss /= mask.sum() + 1e-10

        #import matplotlib.pyplot as plt
        #plt.imshow(probs[:, self.idc, ...].squeeze(0).squeeze(0).detach().cpu().numpy())
        #plt.imshow(target[:, self.idc, ...].squeeze(0).squeeze(0).detach().cpu().numpy())
        return loss


class NegCrossEntropy(CrossEntropy):
    """
    Apply the cross-entropy ONLY if the whole image is the selected class.
    This is useful to supervise the background class when we have weak labels.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        _, _, w, h = probs.shape
        trimmed: Tensor = target[:, self.idc, ...]
        full_img: Tensor = torch.einsum("bcwh->b", trimmed) == (w * h)  # List of images that are fully covered

        if full_img.any():
            where = torch.nonzero(full_img).flatten()
            return super().__call__(probs[where], target[where], bounds[where])

        return torch.zeros(1).to(probs.device)

class NaivePenalty():
    """
    Implementation in the same fashion as the log-barrier
    """
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.C = len(self.idc)
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.power: int = kwargs["power"]
        self.curi: int = kwargs["curi"]
        #print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, bounds: Tensor) -> Tensor:
        def penalty(z: Tensor) -> Tensor:
            assert z.shape == ()

            return torch.max(torch.zeros_like(z), z)**2

        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part
        #assert probs.shape == target.shape
        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        if len(bounds.shape)==3:
            bounds = torch.unsqueeze(bounds, 2) 
        _, _, k, two = bounds.shape  # scalar or vector
        assert two == 2
        # assert k == 1  # Keep it simple for now
        value: Tensor = self.__fn__(probs[:, self.idc, ...],self.power)
        #print(value.shape,"value shape")
        lower_b = bounds[:, self.idc, :, 0]
        upper_b = bounds[:, self.idc, :, 1]


        if len(value.shape)==2: #then its norm soft size ... to ameleiorate
            value = value.unsqueeze(2)
            lower_b = lower_b/(w*h)
            upper_b = upper_b/(w*h)
       
        assert value.shape == (b, self.C, k), value.shape
        assert lower_b.shape == upper_b.shape == (b, self.C, k), lower_b.shape

        upper_z: Tensor = (value - upper_b).type(torch.float32).flatten()
        lower_z: Tensor = (lower_b - value).type(torch.float32).flatten()

        upper_penalty: Tensor = reduce(add, (penalty(e) for e in upper_z))
        lower_penalty: Tensor = reduce(add, (penalty(e) for e in lower_z))
        #count_up: Tensor = reduce(add, (penalty(e)>0 for e in lower_z))
        #count_low: Tensor = reduce(add, (penalty(e)>0 for e in upper_z))

        res: Tensor = upper_penalty + lower_penalty
        #count = count_up + count_low
        loss: Tensor = res.sum() / (w * h)**2

        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation

        return loss


class KLPropInv():
    """
    CE between proportions
    """
    def __init__(self, **kwargs):
        self.power: int = kwargs["power"]
        self.__fn__ = getattr(__import__('utils'), kwargs['fn'])
        self.curi: bool = kwargs["curi"]
        self.idc: bool = kwargs["idc_c"]
        self.ivd: bool = kwargs["ivd"]
        self.weights: List[float] = kwargs["weights_se"]
        self.lamb_ce: float = kwargs["lamb_se"]
        self.lamb_conspred: float = kwargs["lamb_conspred"]
        self.lamb_consprior: float = kwargs["lamb_consprior"]
        self.inv_consloss: float = kwargs["inv_consloss"]

    def __call__(self, probs: Tensor, target: Tensor, bounds) -> Tensor:
        assert simplex(probs)  # and simplex(target)  # Actually, does not care about second part

        b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
        # est_prop is the proportion estimated by the network
        est_prop: Tensor = self.__fn__(probs,self.power)
        # gt_prop is the proportion in the ground truth
        if self.curi:
            bounds = bounds[:,0,0] 
            gt_prop = torch.ones_like(est_prop)*bounds/(w*h)
            gt_prop = gt_prop[:,:,0]
        else:
            gt_prop: Tensor = self.__fn__(target,self.power) # the power here is actually useless if we have 0/1 gt labels
        est_prop = est_prop.squeeze(2)
        log_gt_prop: Tensor = (gt_prop + 1e-10).log()
        log_est_prop: Tensor = (est_prop + 1e-10).log()
        loss = -torch.einsum("bc,bc->", [est_prop, log_gt_prop]) + torch.einsum("bc,bc->", [est_prop, log_est_prop])
        assert loss.requires_grad == probs.requires_grad  # Handle the case for validation
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



def d_loss_calc(pred, label):
    loss_params = {'idc' : [0, 1]}
    criterion = BCELoss(**loss_params, dtype="torch.float32")
    return criterion(pred, label)
