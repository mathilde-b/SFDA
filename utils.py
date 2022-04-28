#!/usr/bin/env python3.6

from random import random
from pathlib import Path
from multiprocessing.pool import Pool
import os
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union
import scipy as sp
import torch
import numpy as np
from tqdm import tqdm
from torch import einsum
from torch import Tensor
from functools import partial, reduce
from skimage.io import imsave,imread
from PIL import Image, ImageOps
from scipy.spatial.distance import directed_hausdorff
import torch.nn as nn
#import pydensecrf.densecrf as dcrf
#from pydensecrf.utils import unary_from_labels
#from pydensecrf.utils import unary_from_softmax
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from viewer import display_item
from typing import Dict
from PIL import Image, ImageOps
import nibabel as nib
import warnings

# functions redefinitions
tqdm_ = partial(tqdm, ncols=125,
                leave=False,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [' '{rate_fmt}{postfix}]')

A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)


def get_weights(list1):
    if len(list1)==5:
        m0,m1,m2,m3,m4 = list1
    elif len(list1)==4:
        m1,m2,m3,m4 = list1
        m0 = 256*256 -m1 -m2 -m3 -m4
        list1 = [m0]+list1
    elif len(list1)==1:
        m1 = list1[0]
        m0 = 256*36 -m1
        list1 = [m0]+list1
    inv = [1/m for m in list1]
    N = np.sum(inv)
    w_vec = [i/N for i in inv]
    print(np.sum(w_vec))
    return np.round(w_vec,2)


def Nmaxelements(list1, N):
    final_list = []

    for i in range(0, N):
        max1 = 0

        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j];

        list1.remove(max1);
        final_list.append(max1)

    return(final_list)

def get_optimal_crop(size_in_mr,size_in_ct):
    a=256*256
    area = (size_in_mr * a) / size_in_ct
    length = np.sqrt(area)
    crop = np.round((256-length)/2,0)
    return crop


def read_anyformat_image(filename):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        #print(filename)
        #acc = imread(filename)

        if Path(filename).suffix == ".png":
            acc: np.ndarray = imread(filename)
        elif Path(filename).suffix == ".npy":
            acc: np.ndarray = np.load(filename)
        elif Path(filename).suffix == ".nii":
            acc: np.ndarray = read_nii_image(filename)
            acc = np.squeeze(acc)
    return(acc)


def read_unknownformat_image(filename):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        #acc = imread(filename)
        print(filename)
        try:
            if Path(filename).suffix == ".png":
                acc: np.ndarray = imread(filename)
            elif Path(filename).suffix == ".npy":
                acc: np.ndarray = np.load(filename)
            elif Path(filename).suffix == ".nii":
                acc: np.ndarray = read_nii_image(filename)
                #acc = np.squeeze(acc)
        except:
            #print('changing extension')
            filename = os.path.splitext(filename)[0]+".png"
            acc: np.ndarray = imread(filename)
            acc = np.expand_dims(acc,0)
        return(acc)




def read_nii_image(input_fid):
    """read the nii image data into numpy array"""
    img = nib.load(str(input_fid))
    return img.get_data()

def exp_lr_scheduler(optimizer, epoch, lr_decay=0.1, lr_decay_epoch=20):
    """Decay learning rate by a factor of lr_decay every lr_decay_epoch epochs"""
    if (epoch % lr_decay_epoch) or epoch==0:
        return optimizer
                        
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
        return optimizer


def depth(e: List) -> int:
    """
    Compute the depth of nested lists
    """
    if type(e) == list and e:
        return 1 + depth(e[0])

    return 0

def iIoU(pred: Tensor, target: Tensor) -> Tensor:
    IoUs = inter_sum(pred, target) / (union_sum(pred, target) + 1e-10)
    assert IoUs.shape == pred.shape[:2]

    return IoUs

def inter_sum(a: Tensor, b: Tensor) -> Tensor:
    return einsum("bcwh->bc", intersection(a, b).type(torch.float32))


def union_sum(a: Tensor, b: Tensor) -> Tensor:
    return einsum("bcwh->bc", union(a, b).type(torch.float32))

def compose(fns, init):
    return reduce(lambda acc, f: f(acc), fns, init)


def compose_acc(fns, init):
    return reduce(lambda acc, f: acc + [f(acc[-1])], fns, [init])


def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))


def mmap_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return Pool().map(fn, iter)


def uc_(fn: Callable) -> Callable:
    return partial(uncurry, fn)


def uncurry(fn: Callable, args: List[Any]) -> Any:
    return fn(*args)


def id_(x):
    return x


# fns
def soft_size(a: Tensor,power:int) -> Tensor:
    return torch.einsum("bcwh->bc", [a])[..., None]

def norm_soft_size(a: Tensor, power:int) -> Tensor:
    b, c, w, h = a.shape
    sl_sz = w*h
    amax = a.max(dim=1, keepdim=True)[0]+1e-10
    #amax = torch.cat(c*[amax], dim=1)
    resp = (torch.div(a,amax))**power
    ress = einsum("bcwh->bc", [resp]).type(torch.float32)
    ress_norm = ress/(torch.sum(ress,dim=1,keepdim=True)+1e-10)
    #print(torch.sum(ress,dim=1))
    return ress_norm.unsqueeze(2)


def batch_soft_size(a: Tensor) -> Tensor:
    return torch.einsum("bcwh->c", [a])[..., None]


def soft_centroid(a: Tensor) -> Tensor:
    b, c, w, h = a.shape

    ws, hs = map_(lambda e: Tensor(e).to(a.device).type(torch.float32), np.mgrid[0:w, 0:h])
    assert ws.shape == hs.shape == (w, h)

    flotted = a.type(torch.float32)
    tot = einsum("bcwh->bc", [a]).type(torch.float32) + 1e-10

    cw = einsum("bcwh,wh->bc", [flotted, ws]) / tot
    ch = einsum("bcwh,wh->bc", [flotted, hs]) / tot

    res = torch.stack([cw, ch], dim=2)
    assert res.shape == (b, c, 2)

    return res


def soft_centroid2(a: Tensor) -> Tensor:
    b, c, w, h = a.shape

    ws, hs = map_(lambda e: Tensor(e).to(a.device).type(torch.float32), np.mgrid[0:w, 0:h])
    ws, hs = index2cartesian(a)
    assert ws.shape == hs.shape == (w, h)

    flotted = a.type(torch.float32)
    tot = einsum("bcwh->bc", [a]).type(torch.float32) + 1e-10

    cw = einsum("bcwh,wh->bc", [flotted, ws]) / tot
    ch = einsum("bcwh,wh->bc", [flotted, hs]) / tot

    res = torch.stack([cw, ch], dim=2)
    assert res.shape == (b, c, 2)
    return res


def index2cartesian(t: Tensor) -> Tensor:
    b, c, h, w = t.shape
    grid = np.indices((h, w))
    cart = torch.Tensor(2, h, w).to(t.device).type(torch.float32)
    cart[0, :] = torch.from_numpy(grid[1] - np.floor(w / 2))  # x coord
    cart[1, :] = torch.from_numpy(-grid[0] + np.floor(h / 2))  # y coord
    return cart


def soft_intensity(a: Tensor, im: Tensor) -> Tensor:
    #b, c, w, h = a.shape

    flotted = a.type(torch.float32)
    tot = einsum("bcwh->bc", [a]).type(torch.float32) + 1e-10

    si = einsum("bcwh,wh->bc", [flotted, im]) / tot

    return si


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def simplex(t: Tensor, axis=1, dtype=torch.float32) -> bool:
    _sum = t.sum(axis).type(dtype)
    _ones = torch.ones_like(_sum, dtype=_sum.dtype)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1, dtype=torch.float32) -> bool:
    return simplex(t, axis, dtype) and sset(t, [0, 1])


# # Metrics and shitz
def meta_dice(sum_str: str, label: Tensor, pred: Tensor, smooth: float = 1e-8, dtype=torch.float32):
    assert label.shape == pred.shape, print(label.shape, pred.shape)
    assert one_hot(label)
    assert one_hot(pred)

    inter_size: Tensor = einsum(sum_str, [intersection(label, pred)]).type(dtype)
    sum_sizes_label: Tensor = einsum(sum_str, [label]).type(dtype)
    sum_sizes_pred: Tensor = einsum(sum_str, [pred]).type(dtype)

    dices: Tensor = (2 * inter_size + smooth) / ((sum_sizes_label + sum_sizes_pred).type(dtype)+ smooth)

    return dices, inter_size, sum_sizes_label,sum_sizes_pred


dice_coef = partial(meta_dice, "bcwh->bc")
dice_batch = partial(meta_dice, "bcwh->c")  # used for 3d dice


def intersection(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a & b


def union(a: Tensor, b: Tensor) -> Tensor:
    assert a.shape == b.shape
    assert sset(a, [0, 1])
    assert sset(b, [0, 1])
    return a | b


# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)

    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)

    return res


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    #print('range classes:',list(range(C)))
    #print('unique seg:',torch.unique(seg))
    #print("setdiff:",set(torch.unique(seg)).difference(list(range(C))))
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)

    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)

    return res


# Misc utils
def save_images_p(segs: Tensor, names: Iterable[str], root: str, mode: str, iter: int, remap: True) -> None:
    b, w, h = segs.shape  # Since we have the class numbers, we do not need a C axis

    for seg, name in zip(segs, names):
        seg = seg.cpu().numpy()
        if remap:
            #assert sset(seg, list(range(2)))
            seg[seg == 1] = 255
        #save_path = Path(root, f"iter{iter:03d}", mode, name).with_suffix(".png")
        save_path = Path(root,mode,"WatonInn_pjce",name).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        imsave(str(save_path), seg.astype('uint8'))

# Misc utils
def save_be_images(segs: Tensor, names: Iterable[str], root: str, mode: str, iter: int, remap: True) -> None:
    b, w, h = segs.shape  # Since we have the class numbers, we do not need a C axis

    for seg, name in zip(segs, names):
        seg = seg.cpu().numpy()
        if remap:
            #assert sset(seg, list(range(2)))
            seg[seg == 1] = 255
        save_path = Path(root, "best", name).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        imsave(str(save_path), seg.astype('uint8'))


# Misc utils
def save_images(segs: Tensor, names: Iterable[str], root: str, mode: str, iter: int, remap: True) -> None:
    b, w, h = segs.shape  # Since we have the class numbers, we do not need a C axis
    for seg, name in zip(segs, names):
        #print(root,iter,mode,name)
        seg = seg.cpu().numpy()
        if remap:
            #assert sset(seg, list(range(2)))
            seg[seg == 1] = 255
        save_path = Path(root, f"iter{iter:03d}", mode, name).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        imsave(str(save_path), seg.astype('uint8'))


def save_images_ent(segs: Tensor, names: Iterable[str], root: str, mode: str, iter: int) -> None:
    b, w, h = segs.shape  # Since we have the class numbers, we do not need a C axis
    for seg, name in zip(segs, names):
        #print(root,iter,mode,name)
        seg = seg.cpu().numpy()*255 #entropy is smaller than 1
        save_path = Path(root, f"iter{iter:03d}", mode, name).with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        imsave(str(save_path), seg.astype('uint8'))


def augment(*arrs: Union[np.ndarray, Image.Image], rotate_angle: float = 45,
            flip: bool = True, mirror: bool = True,
            rotate: bool = True, scale: bool = False) -> List[Image.Image]:
    imgs: List[Image.Image] = map_(Image.fromarray, arrs) if isinstance(arrs[0], np.ndarray) else list(arrs)

    if flip and random() > 0.5:
        imgs = map_(ImageOps.flip, imgs)
    if mirror and random() > 0.5:
        imgs = map_(ImageOps.mirror, imgs)
    if rotate and random() > 0.5:
        angle: float = uniform(-rotate_angle, rotate_angle)
        imgs = map_(lambda e: e.rotate(angle), imgs)
    if scale and random() > 0.5:
        scale_factor: float = uniform(1, 1.2)
        w, h = imgs[0].size  # Tuple[int, int]
        nw, nh = int(w * scale_factor), int(h * scale_factor)  # Tuple[int, int]

        # Resize
        imgs = map_(lambda i: i.resize((nw, nh)), imgs)

        # Now need to crop to original size
        bw, bh = randint(0, nw - w), randint(0, nh - h)  # Tuple[int, int]

        imgs = map_(lambda i: i.crop((bw, bh, bw + w, bh + h)), imgs)
        assert all(i.size == (w, h) for i in imgs)

    return imgs


def augment_arr(*arrs_a: np.ndarray) -> List[np.ndarray]:
    arrs = list(arrs_a)  # manoucherie type check

    if random() > 0.5:
        arrs = map_(np.flip, arrs)
    if random() > 0.5:
        arrs = map_(np.fliplr, arrs)
    # if random() > 0.5:
    #     orig_shape = arrs[0].shape

    #     angle = random() * 90 - 45
    #     arrs = map_(lambda e: sp.ndimage.rotate(e, angle, order=1), arrs)

    #     arrs = get_center(orig_shape, *arrs)

    return arrs



def get_center(shape: Tuple, *arrs: np.ndarray) -> List[np.ndarray]:
    def g_center(arr):
        if arr.shape == shape:
            return arr

        dx = (arr.shape[0] - shape[0]) // 2
        dy = (arr.shape[1] - shape[1]) // 2

        if dx == 0 or dy == 0:
            return arr[:shape[0], :shape[1]]

        res = arr[dx:-dx, dy:-dy][:shape[0], :shape[1]]  # Deal with off-by-one errors
        assert res.shape == shape, (res.shape, shape, dx, dy)

        return res

    return [g_center(arr) for arr in arrs]


def pad_to(img: np.ndarray, new_h, new_w) -> np.ndarray:
    if len(img.shape) == 2:
        h, w = img.shape
    else:
        b, h, w = img.shape
    padd_lr = int((new_w - w) / 2)
    if (new_w - w) / 2 == padd_lr:
        padd_l = padd_lr
        padd_r = padd_lr
    else: #no divisible by 2
        padd_l = padd_lr
        padd_r = padd_lr+1
    padd_ud = int((new_h - h) / 2)
    if (new_h - h) / 2 == padd_ud:
        padd_u = padd_ud
        padd_d = padd_ud
    else: #no divisible by 2
        padd_u = padd_ud
        padd_d = padd_ud+1

    if len(img.shape) == 2:
        new_im = np.pad(img, [(padd_u, padd_d), (padd_l, padd_r)], 'constant')
    else:
        new_im = np.pad(img, [(0, 0), (padd_u, padd_d), (padd_l, padd_r)], 'constant')

    #print(img.shape, '-->', new_im.shape)
    return new_im


def mask_resize(t, new_w):
    b, c, h, w = t.shape
    new_t = t
    if w != new_w:
        device = t.device()
        dtype = t.dtype()
        padd_lr = int((w - int(new_w)) / 2)
        m = torch.nn.ZeroPad2d((0, 0, padd_lr, padd_lr))
        mask_resize = torch.ones([new_w, h], dtype=dtype)
        mask_resize_fg = m(mask_resize)
        mask_resize_bg = 1 - mask_resize_fg
        new_t = torch.einsum('wh,bcwh->bcwh', [mask_resize, t]).to(device)
    return new_t

def resize(t, new_w):
    b, c, h, w = t.shape
    new_t = t
    if w != new_w:
        padd_lr = int((w - int(new_w)) / 2)
        new_t = t[:,: , :, padd_lr-1:padd_lr+new_w-1]
    return new_t


def resize_im(t, new_w):
    w, h = t.shape
    padd_lr = int((w - int(new_w)) / 2)
    new_t = t[:,padd_lr-1:padd_lr+new_w-1]
    return new_t

def haus_p(haus_s,all_p):
    _,C = haus_s.shape
    unique_p = torch.unique(all_p)
    haus_all_p = []
    for p in unique_p:
        haus_p = torch.masked_select(haus_s, all_p == p).reshape((-1, C))
        haus_all_p.append(haus_p.mean().cpu().numpy())
    return(np.mean(haus_all_p),np.std(haus_all_p))


def haussdorf(preds: Tensor, target: Tensor, dtype=torch.float32) -> Tensor:
    assert preds.shape == target.shape
    assert one_hot(preds)
    assert one_hot(target)

    B, C, _, _ = preds.shape

    res = torch.zeros((B, C), dtype=dtype, device=preds.device)
    n_pred = preds.cpu().numpy()
    n_target = target.cpu().numpy()

    for b in range(B):
        for c in range(C):
            res[b, c] = numpy_haussdorf(n_pred[b, c], n_target[b, c])

    return res

def haussdorf_asd(preds: Tensor, target: Tensor, dtype=torch.float32) -> Tensor:
    assert preds.shape == target.shape
    assert one_hot(preds)
    assert one_hot(target)

    B, C, _, _ = preds.shape

    res = torch.zeros((B, C), dtype=dtype, device=preds.device)
    res2 = torch.zeros((B, C), dtype=dtype, device=preds.device)
    n_pred = preds.cpu().numpy()
    n_target = target.cpu().numpy()

    for b in range(B):
        for c in range(C):
            res[b, c] = numpy_haussdorf(n_pred[b, c], n_target[b, c])
            res2[b, c] = numpy_asd(n_pred[b, c], n_target[b, c])

    return res

def numpy_haussdorf(pred: np.ndarray, target: np.ndarray) -> float:
    assert len(pred.shape) == 2
    assert pred.shape == target.shape

    return max(directed_hausdorff(pred, target)[0], directed_hausdorff(target, pred)[0])


def numpy_asd(pred: np.ndarray, target: np.ndarray):
    assert len(pred.shape) == 2
    assert pred.shape == target.shape
    res = directed_hausdorff(pred, target)
    res = res[0]
    return res


def run_CRF(preds: Tensor, nit: int):
    # preds is either probability or hard labeling
    #here done in two classes case only
    b, c, w, h = preds.shape
    dtype = torch.float32
    output = torch.zeros((b, c, w, h), dtype=dtype, device=preds.device)
    for i in range(0,b):
        im = preds[i,:,:,:]
        if sset(im, [0, 1]):  # hard labels in one hot form or class form
            hard = True
            if c > 1:
                im = class2one_hot(im,c)
            im = im.cpu().numpy()
            u = unary_from_labels(im, c, 0.5, zero_unsure=False)
        else:  # labels in a softmax
            im = im.cpu().numpy()
            u = unary_from_softmax(im)
        d = dcrf.DenseCRF2D(w, h, c)  # width, height, nlabels
        print(u.shape)  # -> (5, 480, 640)
        d.setUnaryEnergy(u)
        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(nit)
        new_im = np.array(Q)
        if hard == True:
            new_im = np.argmax(Q, axis=0).reshape((w, h))
        new_im = new_im.from_numpy(new_im)
        output[i,:,:,:] = new_im

    return output


def run_CRF_im(im: np.ndarray, nit: int, conf: 0.5):
    # preds is either probability or hard labeling
    #here done in two classes case only
    w, h = im.shape
    output = im
    if set(np.unique(im)) > {0}:
        if set(np.unique(im)) <= {0, 255} or set(np.unique(im)) <= {0, 1} :  # hard labels in one hot form or class form
            hard = True
            im[im == 255] = 1
            u = unary_from_labels(im, 2, conf, zero_unsure=False)
        else:  # labels in a softmax
            u = unary_from_softmax(im)
        d = dcrf.DenseCRF2D(w, h, 2)  # width, height, nlabels
        #print(u.shape)  # -> (5, 480, 640)
        d.setUnaryEnergy(u)
        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(nit)
        new_im = np.array(Q)
        if hard == True:
            new_im = np.argmax(Q, axis=0).reshape((w, h))
            new_im[new_im == 1] = 255
        output = new_im

    return output


def interp(input):
    _, _, w, h = input.shape
    return nn.Upsample(input, size=(h, w), mode='bilinear')


def interp_target(input):
    _, _, w, h = input.shape
    return nn.Upsample(size=(h, w), mode='bilinear')


def plot_t(input):
    _, c, w, h = input.shape
    axis_to_plot = 1
    if c ==1:
        axis_to_plot = 0
    if input.requires_grad:
        im = input[0, axis_to_plot, :, :].detach().cpu().numpy()
    else:
        im = input[0, axis_to_plot, :, :].cpu().numpy()
    plt.close("all")
    plt.imshow(im, cmap='gray')
    plt.title('plotting on channel:'+ str(axis_to_plot))
    plt.colorbar()


def plot_all(gt_seg, s_seg, t_seg, disc_t):
    _, c, w, h = s_seg.shape
    axis_to_plot = 1
    if c ==1:
        axis_to_plot = 0
    s_seg = s_seg[0, axis_to_plot, :, :].detach().cpu().numpy()
    t_seg = t_seg[0, axis_to_plot, :, :].detach().cpu().numpy()
    gt_seg = gt_seg[0, axis_to_plot, :, :].cpu().numpy()
    disc_t = disc_t[0, 0, :, :].detach().cpu().numpy()
    plt.close("all")
    plt.subplot(141)
    plt.imshow(gt_seg, cmap='gray')
    plt.subplot(142)
    plt.imshow(s_seg, cmap='gray')
    plt.subplot(143)
    plt.imshow(t_seg, cmap='gray')
    plt.subplot(144)
    plt.imshow(disc_t, cmap='gray')
    plt.suptitle('gt, source seg, target seg, disc_t', fontsize=12)
    plt.colorbar()


def plot_as_viewer(gt_seg, s_seg, t_seg, s_im, t_im):
    _, c, w, h = s_seg.shape
    axis_to_plot = 1
    if c ==1:
        axis_to_plot = 0

    s_seg = s_seg[0, axis_to_plot, :, :].detach().cpu().numpy()
    t_seg = t_seg[0, axis_to_plot, :, :].detach().cpu().numpy()
    s_im = s_im[0, 0, :, :].detach().cpu().numpy()
    s_im = resize_im(s_im, s_seg.shape[1])
    t_im = t_im[0, 0, :, :].detach().cpu().numpy()
    t_im = resize_im(t_im, t_seg.shape[1])
    gt_seg = gt_seg[0, axis_to_plot, :, :].cpu().numpy()

    plt.close("all")
    fig = plt.figure()
    gs = gridspec.GridSpec(1, 3)

    axe = fig.add_subplot(gs[0, 0])
    axe.imshow(gt_seg, cmap='gray')
    axe = fig.add_subplot(gs[0, 1])
    display_item(axe, s_im, s_seg, True)
    axe = fig.add_subplot(gs[0, 2])
    display_item(axe, t_im, t_seg, True)
    #fig.show()

    fig.suptitle('gt, source seg, target seg', fontsize=12)


def save_dict_to_file(dic, workdir):
    save_path = Path(workdir, 'params.txt')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(save_path,'w')
    f.write(str(dic))
    f.close()


def load_dict_from_file(workdir):
    f = open(workdir+'/params.txt','r')
    data = f.read()
    f.close()
    return eval(data)


def remap(changes: Dict[int, int], im):
    assert set(np.unique(im)).issubset(changes), (set(changes), np.unique(im))
    for a, b in changes.items():
        im[im == a] = b
    return im


def get_remaining_time(done, total, its):
    time_s= (total-done)/its
    time_m = round(time_s/60,0)
    return time_m

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, i_iter, learning_rate, num_steps, power):
    lr = lr_poly(learning_rate, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def loss_calc(pred, label, device):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).to(device=device)
    #criterion = CrossEntropy2d().to(device=device)
    loss_params={'idc' : [0, 1], 'weights':[9216/9099,9216/117]}
    bounds = torch.randn(1)
    criterion = CrossEntropy(**loss_params, dtype="torch.float32")
    return criterion(pred, label, bounds)

def d_loss_calc(pred, label):
    #label = Variable(label.long()).to(device=device)
    loss_params={'idc' : [0, 1]}
    criterion = BCELoss(**loss_params, dtype="torch.float32")
    return criterion(pred, label)


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



def adjust_learning_rate_D(optimizer, i_iter, learning_rate_D, num_steps, power):
    lr = lr_poly(learning_rate_D, i_iter, num_steps, power)
    optimizer.param_groups[0]['lr'] = lr
    print(f'> New learning Rate: {lr}')
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


#### Helpers for file IOs
def _read_lists(fid):
    """
    Read all kinds of lists from text file to python lists
    """
    if not os.path.isfile(fid):
        return None
    with open(fid,'r') as fd:
        _list = fd.readlines()

    my_list = []
    for _item in _list:
        if len(_item) < 3:
            _list.remove(_item)
        my_list.append(_item.split('\n')[0])
    return my_list

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
