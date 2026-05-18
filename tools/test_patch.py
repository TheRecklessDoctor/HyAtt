import h5py
import math
from typing import Tuple, Any


import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
from skimage.measure import label
from skimage.morphology import remove_small_holes
import numpy as np


from .metrics import compute_absolute_lesion_difference, compute_lesion_f1_score



def getLargestCC(
        segmentation
    ):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


def get_absolute_volume_difference(
        im1,
        im2,
        voxel_size:float
) -> float:

    im1 = np.asarray(im1).astype(bool)
    im2 = np.asarray(im2).astype(bool)
    voxel_size = voxel_size.astype(bool)

    if im1.shape != im2.shape:
        print("Shape mismatch")

    ground_truth_volume = np.sum(im1) * voxel_size
    prediction_volume = np.sum(im2) * voxel_size

    return np.abs(ground_truth_volume - prediction_volume)
    


def var_all_case(
        model:torch.nn.Module, 
        num_classes: int, 
        logger:callable,
        iter: int,
        dataset_name: str = "isles-24",
        patch_size: Tuple[int, int, int] = (80, 80, 80), 
        stride_xy: int = 18, 
        stride_z: int = 4
    ) -> tuple[float, np.floating[Any]]:

    
    validation_file = "validation.list" if dataset_name == "isles-24" else "validation_22.list"
    total_dice = 0.0
    with open(f"src/tools/{validation_file}", 'r') as f:
        image_list = f.readlines()
    
    image_list = [image.replace('\n', '') for image in image_list]
    loader = tqdm(image_list)
    dice_scores = []

    for img_path in loader:

        h5f = h5py.File(img_path, 'r')
        dwi, adc, label = h5f["dwi"][:], h5f["adc"][:], h5f["label"][:]
        # print("Label unique:", torch.unique(torch.from_numpy(label.astype(np.int32))))
        prediction_sub, _ = test_single_case_all(model, dwi, adc, stride_xy, stride_z, patch_size, num_classes=num_classes)
        print("prediction_sub unique:", torch.unique(torch.from_numpy(prediction_sub)))
        if np.sum(prediction_sub)==0:
            dice = 0
        else:
            dice = metric.binary.dc(prediction_sub, label)
        total_dice += dice
        dice_scores.append(dice)

    avg_dice = total_dice / len(loader)
    median_dice = np.median(dice_scores)

    logger.add_vals(
        ["average dice", "median dice"],
        [avg_dice, median_dice]
    )
    print(str(logger))

    return avg_dice, median_dice

def test_all_case(
        model, 
        num_classes: int, 
        logger,
        dataset_name:str = "isles-24",
        patch_size: Tuple[int, int, int] = (80, 80, 80), 
        stride_xy: int = 18, 
        stride_z: int = 4, 
        save_result: bool = True, 
        test_save_path = None, 
        preproc_fn = None, 
        nms: int = 0,
        experiment_num: int = 0,
        missing_holes:int = 0
    ) -> list:

    validation_file = "test.list" if dataset_name == "isles-24" else "test_22.list"
    # validation_file = "train_22.list"
    total_metric1 = 0.0

    with open(f"src/tools/{validation_file}", 'r') as f:
        image_list = f.readlines()
    image_list = [image.replace('\n', '') for image in image_list]
    loader = tqdm(image_list)
    ith = 0
    metrics = []
    individual_metrics = []
    for img_path in loader:

        print(f"path: {img_path}")
        h5f = h5py.File(img_path, 'r')
        dwi, adc, label = h5f["dwi"][:], h5f["adc"][:], h5f["label"][:]
        
        if preproc_fn is not None:
            image = preproc_fn(image)

        prediction, _= test_single_case_all(model, dwi, adc, stride_xy, stride_z, patch_size, num_classes=num_classes)
       
        if nms:
            prediction = getLargestCC(prediction)
        if missing_holes:
            prediction = remove_small_holes(prediction, area_threshold=20)
        if np.sum(prediction)==0:
            single_metric = (0,0,0,0,0,0,0,0,0)
        else:
            single_metric = calculate_metric_percase(prediction, label[:], logger=logger)
    
        individual_metrics.append(single_metric)
        single_metric = np.asarray(single_metric)
        total_metric1 += single_metric
        metrics.append(single_metric)

        logger.add_vals(
            ["metrics"],
            np.asarray(single_metric)
        )

        print(str(logger))
        save_image_path = test_save_path + f"/images/experiment_{experiment_num}/"

        if save_result:
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), save_image_path +  "%02d_pred.nii.gz" % ith)
            nib.save(nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)), save_image_path +  "%02d_gt.nii.gz" % ith)

        ith += 1

    avg_metric1 = total_metric1 / len(loader)
    median_metrics = np.median(metrics, axis=0)

    logger.add_vals(
        ["dice", "jc", "hd", "asd", "assd", "precision", "se", "sp", "F1"], 
        avg_metric1
    )
    print(str(logger))

    return avg_metric1, median_metrics, individual_metrics

def test_single_case_all(
        model, 
        image_1, 
        image_2, 
        stride_xy: int, 
        stride_z: int, 
        patch_size: Tuple[int, int, int], 
        num_classes: int = 1
    ) -> list:
    w, h, d = image_1.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image_1 = np.pad(image_1, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
        image_2 = np.pad(image_2, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)

    ww,hh,dd = image_1.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map_1 = np.zeros((num_classes, ) + image_1.shape).astype(np.float32)
    cnt = np.zeros(image_1.shape).astype(np.float16)

    model.eval()
    
    #patch extraction
    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):

                zs = min(stride_z * z, dd-patch_size[2])
                test_patch_1 = image_1[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch_1 = np.expand_dims(np.expand_dims(test_patch_1,axis=0),axis=0).astype(np.float32)
                # test_patch_1 = torch.from_numpy(test_patch_1).cuda()
                test_patch_1 = torch.from_numpy(test_patch_1)
                test_patch_2 = image_2[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch_2 = np.expand_dims(np.expand_dims(test_patch_2,axis=0),axis=0).astype(np.float32)
                # test_patch_2 = torch.from_numpy(test_patch_2).cuda()
                test_patch_2 = torch.from_numpy(test_patch_2)
                
                input = torch.cat([test_patch_1, test_patch_2], dim=1)
                input = input.cuda()
                # input = test_patch_1.cuda()

                #inference
                with torch.amp.autocast(device_type="cuda"), torch.no_grad():
                    # y1, _, _ = model(test_patch_1, test_patch_2)
                    y1,_,_ = model(input, input)
                    y1 = F.softmax(y1, dim=1)
                    # print("logits: ", y1.min(), y1.max(), y1.mean())
                y1 = y1.cpu().detach().numpy()
                y1 = y1[0,1,:,:,:]
                score_map_1[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map_1[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y1
                
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1

    score_map_1 = score_map_1/np.expand_dims(cnt,axis=0)

    label_map_1 = (score_map_1[0]>0.5).astype(np.int32)
    if add_pad:
        label_map_1 = label_map_1[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map_1 = score_map_1[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]

    # print("here")
    return label_map_1, score_map_1






def calculate_metric_percase(
        pred, 
        gt,
        logger
    ):
    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    assd = metric.binary.assd(pred, gt)
    # f1 = compute_lesion_f1_score(gt, pred)
    # ald = compute_absolute_lesion_difference(gt, pred)
    precision = metric.binary.precision(pred, gt)
    se = metric.binary.sensitivity(pred, gt)
    sp = metric.binary.specificity(pred, gt)

    label_gt = label(gt)
    label_gts = np.bincount(label_gt.flat)
    label_pred = label(pred)
    label_preds = np.bincount(label_pred.flat)
    M, N = label_gts.shape[0], label_preds.shape[0]
    index = np.where(label_gts<11)
    idx_offset = 0
    if index[0].size !=0:
        for idx in range(index[0].shape[0]):
            mask = label_gt==index[0][idx]-idx_offset
            label_gt[mask]=0
            # we need to close the gap after removing the label
            label_gt[label_gt>index[0][idx]-idx_offset] -=1
            idx_offset += 1
            M=M-1
    index = np.where(label_preds<11)
    idx_offset = 0
    if index[0].size !=0:
        for idx in range(index[0].shape[0]):
            mask = label_pred==index[0][idx]-idx_offset
            label_pred[mask]=0
            # we need to close the gap after removing the label
            label_pred[label_pred>index[0][idx]-idx_offset] -=1
            idx_offset += 1
            N=N-1
    H_ij = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            H_ij[i, j] = ((label_gt==i) * (label_pred==j)).sum()
    TPg=0
    for i in range(1, M):
        alpha = H_ij[i, 1:].sum() / (H_ij[i, :].sum() + 1e-18)
        if alpha > 0.1:
            wsum, k, vaccept=0, 0, True
            while wsum < 0.65:
                pk = np.argsort(-H_ij[i, 1:])[k]+1#np.argwhere(np.argsort(H_ij[i])==k)[0][0]
                tk = H_ij[0, pk] / H_ij[:, pk].sum()
                if tk >0.7:
                    vaccept = False
                    break
                wsum += H_ij[i, pk] / H_ij[i, 1:].sum()
                k +=1
            if vaccept == True:
                TPg +=1
    TPa=0
    H_ji = H_ij.T
    for j in range(1, N):
        alpha = H_ji[j, 1:].sum() / (H_ji[j, :].sum()+ 1e-18)
        if alpha > 0.1:
            wsum, k, vaccept=0, 0, True
            while wsum < 0.65:
                pk = np.argsort(-H_ji[j, 1:])[k]+1#np.argwhere(np.argsort(H_ji[j])==k)[0][0]
                tk = H_ji[0, pk] / H_ji[:, pk].sum()
                if tk >0.7:
                    vaccept = False
                    break
                wsum += H_ji[j, pk] / H_ji[j, 1:].sum()
                k +=1
            if vaccept == True:
                TPa +=1
    if N-1 == 0:
        sel, pl = 0, 0
    else:
        sel, pl = TPg/(M-1),TPa/(N-1)
    if sel == 0 or pl == 0:
        F1 = 0
    else:
        F1 = (2 * sel * pl) / (sel+pl)

    logger.add_vals(
        ["TPg", "M", "TPa", "N"],
        [TPg, M-1, TPa, N-1]
    )
    print(str(logger))

    logger.add_vals(
        ["sel", "pl", "f1"], 
        [sel, pl, F1]
    )
    print(str(logger))

    logger.add_vals(
        ["dice", "jc", "95hd", "asd", "assd", "pr", "se", "sp"],
        [dice, jc, hd, asd, assd, precision, se, sp]
    )
    print(str(logger))

    return dice, jc, hd, asd, assd, precision, se, sp, F1
