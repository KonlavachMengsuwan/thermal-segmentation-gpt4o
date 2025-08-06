# meta_sam

how to use

1,
$pip install -r requirements.txt
$pip install git+https://github.com/facebookresearch/segment-anything.git

2,
$mkdir input
then, put *_Visual_IR.jpeg and  *.xlsx or *.csv to input directory

3,
download https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
and put next to test2.py

4,
$python test2.py --path=./input/ --mask_type=sam --use_temp --filename_ext=_Visual_IR.jpeg
If you run at no gpu machine, please modify as below:
    src/_common.py
    before GEN_MASK_SAM_DEVICE : str = "cuda"
    after  GEN_MASK_SAM_DEVICE : str = "cpu"

5,
$python test2.py --path=./input/ --mask_type=xlsx --filename_ext=_Visual_IR.jpeg

generate compare.xlsx next to test2.py