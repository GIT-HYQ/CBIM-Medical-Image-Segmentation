# Tooth 3D quickstart

This quickstart describes the first runnable integration for tooth multi-class 3D segmentation.

## 1) Analyze spacing (recommended for target spacing)

Run spacing analysis on `imagesTr` to get median/p10/p90 and a recommended target spacing:

```powershell
python dataset_conversion/analyze_tooth_spacing.py --images_dir RAW/images --image_suffix .mha --save_report .\spacing_stats.yaml
```


Use `recommended_target_spacing` (or nearby rounded values like `0.4,0.4,0.4`) for `tooth_3d.py --target_spacing`.

## 2) Analyze labels (optional but recommended)

Run a dataset-wide label audit to get unique IDs and a suggested contiguous mapping:

```powershell
python dataset_conversion/analyze_tooth_labels.py --labels_dir RAW/labels --label_suffix .mha --save_report .\label_stats.yaml --save_map .\label_map.yaml
```

## 3) Prepare label map

Create a YAML map from raw tooth IDs to contiguous training IDs.

Important policy: raw IDs `1..10` are treated as non-tooth/background and should map to `0`.

Example (`label_map.yaml`):

```yaml
0: 0
1: 0
2: 0
3: 0
4: 0
5: 0
6: 0
7: 0
8: 0
9: 0
10: 0
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

Name matching rule (important):

- By default, `tooth_3d.py` uses `--image_stem_suffix _0000`.
- Example: image `ToothFairy2F_001_0000.mha` is matched to label `ToothFairy2F_001.mha`.
- If your image and label stems are already the same, disable trimming with `--image_stem_suffix ""`.
- Output `case_name` follows the matched label stem, and files are written as `case_name.nii.gz` and `case_name_gt.nii.gz` in the same folder.

Run conversion:

```powershell
python dataset_conversion/tooth_3d.py --src_images RAW/images --src_labels RAW/labels --dst_root DATA/tooth_3d --label_map .\label_map.yaml --strict_unmapped --target_spacing 0.4,0.4,0.4
```

Optional (recommended for faster loading): foreground crop after remapping labels (`label > 0`), with context in `z,y,x`:

```powershell
python dataset_conversion/tooth_3d.py --src_images /home/share/clr/share/data/CBCT/Dataset112_ToothFairy2/imagesTr/ --src_labels /home/share/clr/share/data/CBCT/Dataset112_ToothFairy2/labelsTr/ -dst_root /home/share/clr/share/data/CBCT/tooth_fairy2 --label_map .\label_map.yaml --strict_unmapped --target_spacing 0.3,0.3,0.3 --crop_foreground --crop_context 10,30,30
```

Crop behavior notes:

- The crop bbox is computed from mapped labels (`> 0`) for each case.
- `--crop_context 10,30,30` is an empirical default in voxel units (`z,y,x`), not a fixed best value.
- Increase context (for example `15,40,40`) if validation shows boundary miss/under-segmentation.
- Decrease context to improve speed/memory only when validation quality stays stable.

Equivalent explicit command (same default behavior):

```powershell
python dataset_conversion/tooth_3d.py --src_images RAW/images --src_labels RAW/labels --dst_root DATA/tooth_3d --label_map .\label_map.yaml --strict_unmapped --target_spacing 0.4,0.4,0.4 --image_stem_suffix _0000
```

Output layout:

- `DATA/tooth_3d/case_xxx.nii.gz`
- `DATA/tooth_3d/case_xxx_gt.nii.gz`
- `DATA/tooth_3d/list/dataset.yaml`

## 5) Optional: pre-generate patch indices for training

If 3D training startup is slow or full-case preload uses too much memory, generate a cached patch index first.

This keeps validation / test in full-case mode, but training reads only indexed patches.

```powershell
python dataset_conversion/tooth_patch_index.py --data_root DATA/tooth_3d --patch_size 64,160,160 --patches_per_case 8 --foreground_fraction 0.875 --foreground_threshold 0.01 --background_max_fg_ratio 0.0
```

Default outputs:

- `DATA/tooth_3d/list/tooth_patch_index.npz`
- `DATA/tooth_3d/list/tooth_patch_index_meta.yaml`

Recommended meaning of the main arguments:

- `--patch_size`: use the same `z,y,x` as `training_size`
- `--patches_per_case`: how many candidate training patches to cache per converted case
- `--foreground_fraction`: foreground-preferred patch ratio (for example `0.875` means 7 foreground-preferred + 1 background-preferred when `patches_per_case=8`)
- `--foreground_threshold`: minimum foreground voxel ratio inside a patch to accept it as foreground-preferred

Patch-count sizing tip:

- make sure `train_cases * patches_per_case` is comfortably larger than `iter_per_epoch * batch_size`
- with the default config (`iter_per_epoch: 300`), `patches_per_case=8` is usually enough for a few hundred training cases per fold

## 6) Update config

Edit `config/tooth/medformer_3d.yaml`:

- `data_root` -> your converted folder
- `classes` -> number of classes including background
- `weight` length must equal `classes`
- enable indexed patch training when needed:

```yaml
patch_index_enabled: true
patch_index_file: null   # or an explicit .npz path
patch_index_cache_cases: 2
```

Notes:

- `patch_index_file: null` means use `data_root/list/tooth_patch_index.npz`
- `patch_index_cache_cases` is a small per-worker cache of full cases before patch extraction
- if `patch_index_enabled: false`, the dataset falls back to the previous random online crop behavior

## 7) Train and test

```powershell
python train.py --dataset tooth --model medformer --dimension 3d
python test.py --dataset tooth --model medformer --dimension 3d --load /path/to/fold_0_best.pth
```

## 8) Predict new images

```powershell
python prediction.py --dataset tooth --model medformer --dimension 3d --load /path/to/fold_0_best.pth --img_path /path/to/images --save_path /path/to/preds --target_spacing 0.4,0.4,0.4
```

## 9) Smoke tests

```powershell
python dataset_conversion/tooth_3d_smoke_test.py
python dataset_conversion/tooth_patch_index_smoke_test.py
```
