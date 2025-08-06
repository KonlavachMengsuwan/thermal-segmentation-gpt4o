# 1, import from installed library
import cv2

# 2, import from source code
import src._common as _com

# 3, import from python library
from abc import ABCMeta, abstractmethod


#########################################################

class Modify_Mask(metaclass = ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def exec_model(self, mask_array: list) -> list:
        pass

# -------------------------------------------------------


class Modify_Mask_Default(Modify_Mask):
    """_summary_
        mask_arrayについて重なりがある場合は、それらをマージする

    Args:
        Modify_Mask (_type_): _description_
    """
    def __init__(self):
        super().__init__()

    def exec_model(self, mask_array : list) -> list:
        mask_array_tmp = mask_array.copy()

        img_size = mask_array_tmp[0][0].size

        while_flag = True
        for_flag = False

        while while_flag:
            for_flag = False
            for i in range(len(mask_array_tmp)):
                for j in range(len(mask_array_tmp)):
                    if (i != j):
                        mask1 = mask_array_tmp[i][0]
                        mask2 = mask_array_tmp[j][0]

                        mask_and = cv2.bitwise_and(mask1, mask2)

                        mask_and_count = cv2.countNonZero(mask_and)
                        mask1_cnt = cv2.countNonZero(mask1)
                        mask2_cnt = cv2.countNonZero(mask2)

                        mask_cnt = 0
                        target = 0

                        if (mask1_cnt > mask2_cnt):
                            mask_cnt = mask2_cnt
                            target = i    # remove large
                        else:
                            mask_cnt = mask1_cnt
                            target = j    # remove large

                        mask_ratio = mask_and_count / mask_cnt
                        # print('mask_ratio...' + str(mask_ratio))

                        if (mask_ratio > 0.9):
                            mask_and_not = cv2.bitwise_not(mask_and)
                            mask_array_tmp[target][0] = cv2.bitwise_and(mask_array_tmp[target][0], mask_and_not)
                            # cv2.imwrite(_com.var.common_filename + "_and_Mask_Segment" + str(target) + ".png", mask_and)
                            # cv2.imwrite(_com.var.common_filename + "_and_not_Mask_Segment" + str(target) + ".png", mask_and_not)
                            for_flag = True

                    if for_flag:
                        break

                if for_flag:
                    break

            if not for_flag:
                while_flag = False

        min_area = img_size * _com.var.min_area_percent / 100
        mask_array = filter(lambda x: cv2.countNonZero(x[0]) > min_area, mask_array_tmp)
        mask_array = sorted(mask_array, key=(lambda x: cv2.countNonZero(x[0])), reverse=True)

        """
        # combine under min area mask
        min_mask_array = filter(lambda x: cv2.countNonZero(x[0]) <= min_area, mask_array_tmp)
        min_mask_array = sorted(min_mask_array, key=(lambda x: cv2.countNonZero(x[0])), reverse=True)

        if ((_com.var.mask_type == 'sam') & (len(min_mask_array) > 0)):
            min_combine_mask = zeros((mask_array[0][0].shape[0], mask_array[0][0].shape[1]), dtype=np.uint8)

            for i, min_mask in enumerate(min_mask_array):
                cv2.imwrite(_com.var.common_filename + "_min_combine_Segment" + str(i) + ".png", min_mask[0])
                min_combine_mask.z = cv2.bitwise_or(min_combine_mask, min_mask[0])

            # mask_array.append(min_combine_mask)
        """

        max_size = len(_com.const.GEN_TEMP_MASK_COLOR_ARRAY)

        if (len(mask_array) > max_size):
            del mask_array[max_size: len(mask_array)]

        return mask_array
