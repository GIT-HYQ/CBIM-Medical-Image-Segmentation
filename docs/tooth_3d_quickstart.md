# Tooth 3D quickstart

This quickstart describes the first runnable integration for tooth multi-class 3D segmentation.

## 1) Analyze spacing (recommended for target spacing)

Run spacing analysis on `imagesTr` to get median/p10/p90 and a recommended target spacing:

```powershell
python dataset_conversion/analyze_tooth_spacing.py --images_dir RAW/images --image_suffix .mha --save_report .\spacing_stats.yaml
```

Or use the merged analyzer:

```powershell
python dataset_conversion/analyze_tooth_labels.py --report_mode spacing --images_dir RAW/images --image_suffix .mha --save_spacing_report .\spacing_stats.yaml
```

Use `recommended_target_spacing` (or nearby rounded values like `0.4,0.4,0.4`) for `tooth_3d.py --target_spacing`.

## 2) Analyze labels (optional but recommended)

Run a dataset-wide label audit to get unique IDs and a suggested contiguous mapping:

```powershell
python dataset_conversion/analyze_tooth_labels.py --labels_dir RAW/labels --label_suffix .mha --save_report .\label_stats.yaml --save_map .\label_map.yaml
```

## 3) Prepare label map

Create a YAML map from raw tooth IDs to contiguous training IDs.

Example (`label_map.yaml`):

```yaml
0: 0
11: 1
12: 2
13: 3
14: 4
15: 5
16: 6
17: 7
18: 8
21: 9
22: 10
23: 11
24: 12
25: 13
26: 14
27: 15
28: 16
31: 17
32: 18
33: 19
34: 20
35: 21
36: 22
37: 23
38: 24
41: 25
42: 26
43: 27
44: 28
45: 29
46: 30
47: 31
48: 32
```

## 4) Convert `.mha` to CBIM format

Expected source layout:

- `RAW/images/*.mha`
- `RAW/labels/*.mha`

Run conversion:

```powershell
python dataset_conversion/tooth_3d.py --src_images RAW/images --src_labels RAW/labels --dst_root DATA/tooth_3d --label_map .\label_map.yaml --strict_unmapped --target_spacing 0.4,0.4,0.4
```

Output layout:

- `DATA/tooth_3d/case_xxx.nii.gz`
- `DATA/tooth_3d/case_xxx_gt.nii.gz`
- `DATA/tooth_3d/list/dataset.yaml`

## 5) Update config

Edit `config/tooth/medformer_3d.yaml`:

- `data_root` -> your converted folder
- `classes` -> number of classes including background
- `weight` length must equal `classes`

## 6) Train and test

```powershell
python train.py --dataset tooth --model medformer --dimension 3d
python test.py --dataset tooth --model medformer --dimension 3d --load /path/to/fold_0_best.pth
```

## 7) Predict new images

```powershell
python prediction.py --dataset tooth --model medformer --dimension 3d --load /path/to/fold_0_best.pth --img_path /path/to/images --save_path /path/to/preds --target_spacing 0.4,0.4,0.4
```

## 8) Smoke test conversion

```powershell
python dataset_conversion/tooth_3d_smoke_test.py
```
