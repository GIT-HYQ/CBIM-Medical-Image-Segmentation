import numpy as np
import SimpleITK as sitk
from utils import ResampleXYZAxis, ResampleLabelToRef, CropForeground, reorient_image
import os
import random
import yaml
import copy
import numpy as np
import pdb

def ResampleImage(imImage, imLabel, save_path, name, target_spacing=(1., 1., 1.)):

    assert imImage.GetSpacing() == imLabel.GetSpacing()
    assert imImage.GetSize() == imLabel.GetSize()

    imImage = reorient_image(imImage, 'RAI')
    imLabel = reorient_image(imLabel, 'RAI')

    spacing = imImage.GetSpacing()
    origin = imImage.GetOrigin()


    npimg = sitk.GetArrayFromImage(imImage).astype(np.int32)
    nplab = sitk.GetArrayFromImage(imLabel).astype(np.uint8)
    z, y, x = npimg.shape

    if not os.path.exists('%s'%(save_path)):
        os.mkdir('%s'%(save_path))


    re_img_xy = ResampleXYZAxis(imImage, space=(target_spacing[0], target_spacing[1], spacing[2]), interp=sitk.sitkBSpline)
    re_lab_xy = ResampleLabelToRef(imLabel, re_img_xy, interp=sitk.sitkNearestNeighbor)

    re_img_xyz = ResampleXYZAxis(re_img_xy, space=(target_spacing[0], target_spacing[1], target_spacing[2]), interp=sitk.sitkNearestNeighbor)
    re_lab_xyz = ResampleLabelToRef(re_lab_xy, re_img_xyz, interp=sitk.sitkNearestNeighbor)
    
    if np.random.uniform() < 0.25:
        pass
    else:
        if nplab.max() == 0:
            pass
        else:
            re_img_xyz, re_lab_xyz = CropForeground(re_img_xyz, re_lab_xyz, context_size=[20, 30, 30])

    sitk.WriteImage(re_img_xyz, '%s/%s.nii.gz'%(save_path, name))
    sitk.WriteImage(re_lab_xyz, '%s/%s_gt.nii.gz'%(save_path, name))


if __name__ == '__main__':


    src_path = '/data/local/yg397/abdomenatlasmini/AbdomenAtlasMini1.0/'
    tgt_path = '/data/local/yg397/dataset/abdomenatlas/'


    name_list = os.listdir(src_path)
    lab_name_list = ['aorta', 'kidney_left', 'kidney_right', 'liver', 'postcava', 'stomach', 'gall_bladder', 'pancreas', 'spleen']


    if not os.path.exists(tgt_path+'list'):
        os.mkdir('%slist'%(tgt_path))
    with open("%slist/dataset.yaml"%tgt_path, "w",encoding="utf-8") as f:
        yaml.dump(name_list, f)

    os.chdir(src_path)
    
    for name in range(4892, 4894):
        img_name = os.path.join(f"BDMAP_{name:0>8}", 'ct.nii.gz')
        itk_img = sitk.ReadImage(img_name)

        lab_list = []
        for lab_name in lab_name_list:
            lab_list.append(sitk.ReadImage(os.path.join(src_path, f"BDMAP_{name:0>8}", 'segmentations', f"{lab_name}.nii.gz")))
        np_lab = sitk.GetArrayFromImage(lab_list[0])

        for idx in range(1, len(lab_name_list)):
            tmp_np_lab = sitk.GetArrayFromImage(lab_list[idx])
            np_lab[tmp_np_lab == 1] = idx + 1

        itk_lab = sitk.GetImageFromArray(np_lab)
        itk_lab.CopyInformation(lab_list[0])
        
        ResampleImage(itk_img, itk_lab, tgt_path, f"BDMAP_{name:0>8}", (0.814453125, 0.814453125, 1.0))
        print(name, 'done')


