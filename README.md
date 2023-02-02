### Usage

Date preparation scripts for cropping images and updating respective annotation json file in coco format. 

### Directory Structure
```
.
├── images # img source 
├── scripts
│   └── aug_panel_ann.sh
├── test # output dir
│   ├── ann
│   └── images
├── train_coco.json # annotated json source
└── crop_img.py
```

### Instruction

1. Install augmentation library 
```
# conda 
conda config --add channels conda-forge
conda install imgaug

# pip
pip install imgaug
```

2. Adjust path directories in scripts/aug_panel_ann.sh
```
python ../crop_img.py \
	--dst <output dir> \
    --img_source <image dir source> \
    --json_source <coco annotation json>
```

3. Run script to get annotation after cropping panel level image
```
sh ../scripts/aug_panel_ann.sh
```