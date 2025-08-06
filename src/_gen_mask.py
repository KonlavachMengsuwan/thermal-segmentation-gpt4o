# 1, import from installed library
import numpy as np

# 2, import from source code
import src._common as _com

# 3, import from python library
import pickle
import os
import cv2
import random
from abc import ABCMeta, abstractmethod

# TODO: deviceがAutoでいけるかどうか
# TODO: pickleを圧縮する
"""

Pythonオブジェクトを圧縮して保存したい, import bz2
https://qiita.com/_akisato/items/c4e6c94f2b69aa4fb742

自動で圧縮してくれるPickle, import gzip
https://gist.github.com/alfredplpl/328c6038c17dd063d296

Pythonのオブジェクト圧縮・展開 , import zlib = gzip
https://gist.github.com/yubessy/0cabb1f348a7a251d1ca

gzipの方が速い

joblib
https://www.robotech-note.com/entry/2016/10/22/010638
https://www.salesanalytics.co.jp/datascience/datascience044/
こっちの方が楽か?
"""

#########################################################


class Gen_Mask(metaclass = ABCMeta):
    __slots__ = ('model')

    def __init__(self):
        self.model = None

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def exec_model(self) -> list:
        pass

# -------------------------------------------------------


class Gen_Mask_SAM(Gen_Mask):
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    __slots__ = ('model')

    def __init__(self):
        super().__init__()

    def load_model(self):
        """_summary_
            SAMのモデルをロードする
        """
        if not _com.var.use_pickle:
            sam = self.sam_model_registry[_com.const.GEN_MASK_SAM_MODEL_TYPE](checkpoint = _com.const.GEN_MASK_SAM_CHECKPOINT)
            sam.to(device = _com.const.GEN_MASK_SAM_DEVICE)
            self.model = self.SamAutomaticMaskGenerator(model = sam,
                                                        points_per_side=32,
                                                        pred_iou_thresh=0.86,
                                                        stability_score_thresh=0.92,
                                                        crop_n_layers=1,
                                                        crop_n_points_downscale_factor=2,
                                                        )

    def exec_model(self) -> list:
        """_summary_
            1, SAMのモデルを実行、もしくは実行結果のpickleをロード
        Returns:
            list: _description_
        """

        rtn_array = []

        # Step 1: exec sam
        if _com.var.use_pickle:
            pickle_filename = _com.var.common_filename + '_sam.pickle'

            if (os.path.isfile(pickle_filename)):
                with open(_com.var.common_filename + "_sam.pickle", 'rb') as p:
                    masks = pickle.load(p)

        else:

            image = cv2.imread(_com.var.common_filename + _com.var.filename_ext)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            masks = self.model.generate(image)

            if _com.var.use_pickle:
                with open(_com.var.common_filename + "_sam.pickle", 'wb') as p:
                    pickle.dump(masks, p)

        # Step 2: maskについて、_com.var.min_area_percent 以上の面積なら採用する
        mask_cnt = 0

        min_area = masks[0]['segmentation'].size * _com.var.min_area_percent / 100
        # min_area = 0

        min_area_mask = np.zeros((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1]), dtype=bool)
        total_area_mask = np.zeros((masks[0]['segmentation'].shape[0], masks[0]['segmentation'].shape[1], 3), dtype=np.uint8)

        for i, mask in enumerate(masks):
            min_area_flag = False
            if (mask['area'] < min_area):
                min_area_flag = True
                # continue

            mask_bool = mask['segmentation']

            # astype(np.uint8) ... convert False to 0, True to 1
            # multiple 255 ... # convert 1 to 255
            mask_tmp = (mask_bool.astype(np.uint8)) * 255

            # generate total mask
            mask = np.stack([mask_tmp, mask_tmp, mask_tmp], axis=2)  # convert from 2 dimension array to 3 dimension array
            # cv2.imwrite(common_filename + "_Org_Mask_Segment" + str(i) + ".png", mask)
            before_color = [255, 255, 255]
            after_color = [random.randint(64, 255), random.randint(64, 255), random.randint(64, 255)]
            mask[np.where((mask == before_color).all(axis=2))] = after_color
            total_area_mask = cv2.bitwise_or(total_area_mask, mask)

            if (not min_area_flag):
                array_tmp = []
                array_tmp.append(mask_tmp)
                # array_tmp.append("Num" + str(mask_cnt))  # SAM has no classification information
                array_tmp.append("")  # SAM has no classification information
                mask_cnt += 1
                rtn_array.append(array_tmp)
            else:
                min_area_mask  = np.add(min_area_mask, mask_bool)

        # debug output total mask
        total_area_mask = cv2.cvtColor(total_area_mask, cv2.COLOR_RGB2BGR)
        cv2.imwrite(_com.var.common_filename + "_Total_Segment.png", total_area_mask)

        min_area_mask_uint = (min_area_mask.astype(np.uint8)) * 255

        min_area_ratio = cv2.countNonZero(min_area_mask_uint) / masks[0]['segmentation'].size
        print(f"min_area_ratio...{str(min_area_ratio)}")

        # if (cv2.countNonZero(min_area_mask_uint) > 0) :
        if (min_area_ratio > _com.const.MIN_AREA_RATIO_TH):

            mask = np.stack([min_area_mask_uint, min_area_mask_uint, min_area_mask_uint], axis=2)  # convert from 2 dimension array to 3 dimension array
            cv2.imwrite(_com.var.common_filename + "_Min_Area_Segment.png", mask)

            array_tmp = []
            array_tmp.append(min_area_mask_uint)
            array_tmp.append("")
            rtn_array.append(array_tmp)

        return rtn_array
