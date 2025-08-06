# 1, import from installed library
import csv
import numpy as np
import cv2
import openpyxl
import matplotlib.pyplot as plt

# 2, import from source code
import src._common as _com

# 3, import from python library
import os
from abc import ABCMeta, abstractmethod

#########################################################

class Gen_Temp(metaclass=ABCMeta):

    def __init__(self):
        self.model = None

    @abstractmethod
    def exec_model(self, mask_array) -> list:
        pass

# -------------------------------------------------------

class Gen_Temp_CSV(Gen_Temp):

    def __init__(self):
        super().__init__()

    def exec_model(self, mask_array) -> list:
        # generate  #000000 image
        color_mask_marge = np.zeros((mask_array[0][0].shape[0], mask_array[0][0].shape[1], 3), dtype=np.uint8)
        color_mask_marge_bin = np.zeros((mask_array[0][0].shape[0], mask_array[0][0].shape[1], 3), dtype=np.uint8)

        image_rgb = cv2.imread(_com.var.common_filename + _com.var.filename_ext)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        # get temp max, min from csv or xlsx
        if _com.var.use_temp:
            csv_filename = _com.var.common_filename + '.csv'
            xlsx_filename = _com.var.common_filename + '.xlsx'
            if os.path.isfile(csv_filename):
                with open(csv_filename) as f:
                    reader = csv.reader(f)
                    values = [row for row in reader]
                    temp_matrix = []
                    for i in range(16, 208):
                        temp_matrix.append(values[i][0:256])
                    temp_matrix = [[s.replace(',', '.') for s in text] for text in temp_matrix]
                    temp_matrix = np.asarray(temp_matrix, dtype=np.float32)
            else:
                wb = openpyxl.load_workbook(xlsx_filename)
                sheet = wb[wb.sheetnames[0]]
                temp_matrix = sheet['A17:IV208']
                temp_matrix = [[cell.value for cell in row] for row in temp_matrix]
                temp_matrix = np.asarray(temp_matrix, dtype=np.float32)

        rtn_array = []
        centroid_array = []

        for i, mask_array_temp in enumerate(mask_array):
            mask_array_uint = mask_array_temp[0]

            mask_array_bool = (mask_array_uint.astype(np.bool_))
            centroid_array.append(np.mean(np.argwhere(mask_array_bool), axis=0))

            # get mask image
            mask = np.stack([mask_array_uint, mask_array_uint, mask_array_uint], axis=2)  # convert from 2 dimension array to 3 dimension array

            # output mask image (optional)
            # cv2.imwrite(common_filename + "_Mask_Segment" + str(i) + ".png", mask)

            # get marged color mask image
            color_mask = mask.copy()
            before_color = [255, 255, 255]
            after_color = _com.const.GEN_TEMP_MASK_COLOR_ARRAY[i]
            color_mask[np.where((color_mask == before_color).all(axis=2))] = after_color
            color_mask_marge = cv2.bitwise_or(color_mask_marge, color_mask)

            # get marged mask image
            color_mask_marge_bin = cv2.bitwise_or(color_mask_marge_bin, mask)

            if _com.var.use_temp:
                temp_array = self._generate_temperature_array_csv(mask_array_uint, temp_matrix)

                array_tmp = []
                array_tmp.append(temp_array)
                array_tmp.append(mask_array_temp[1])
                rtn_array.append(array_tmp)
            else:
                array_tmp = []
                array_tmp.append(0)
                array_tmp.append("")
                rtn_array.append(array_tmp)

        # Save segmented RGB and temperature matrix images
        self.save_segmented_rgb(image_rgb, mask_array, _com.var.common_filename)
        self.save_segmented_temp_matrix(temp_matrix, mask_array, _com.var.common_filename)

        rgb_with_mask = cv2.addWeighted(image_rgb, 0.6, color_mask_marge, 0.4, 0)  # exec alpha blending

        rgb_with_mask = cv2.multiply(rgb_with_mask.astype(float), color_mask_marge_bin.astype(float) / 255)  # only alpha blend & masked
        non_color_mask_image = (cv2.bitwise_and(cv2.bitwise_not(color_mask_marge_bin), image_rgb)).astype(float)  # only not masked images
        rgb_with_mask = (cv2.add(rgb_with_mask, non_color_mask_image)).astype(np.uint8)  # add alpha blend & masked and not masked images

        # Add index to image
        for i, pos in enumerate(centroid_array):
            pos_x = int(pos[1])
            pos_y = int(pos[0]) + 10
            disp_value = str(i + 1)

            cv2.putText(rgb_with_mask,
                        text=disp_value,
                        org=(pos_x, pos_y),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=0.7,
                        color=(224, 224, 224),
                        thickness=3,
                        lineType=cv2.LINE_AA)

            cv2.putText(rgb_with_mask,
                        text=disp_value,
                        org=(pos_x, pos_y),
                        fontFace=cv2.FONT_HERSHEY_DUPLEX,
                        fontScale=0.7,
                        color=(_com.const.GEN_TEMP_MASK_COLOR_ARRAY[i][0],
                               _com.const.GEN_TEMP_MASK_COLOR_ARRAY[i][1],
                               _com.const.GEN_TEMP_MASK_COLOR_ARRAY[i][2]),
                        thickness=1,
                        lineType=cv2.LINE_AA)

        rgb_with_mask = cv2.cvtColor(rgb_with_mask, cv2.COLOR_RGB2BGR)
        cv2.imwrite(_com.var.common_filename + '_' + _com.var.mask_type + "_withMask.png", rgb_with_mask)

        color_mask_marge = cv2.cvtColor(color_mask_marge, cv2.COLOR_RGB2BGR)
        cv2.imwrite(_com.var.common_filename + '_' + _com.var.mask_type + "_CMask.png", color_mask_marge)

        return rtn_array

    @staticmethod
    def save_segmented_rgb(image_rgb, mask_array, common_filename):
        """Saves each masked RGB image segment separately while preserving the original color grading"""
        for i, mask_array_temp in enumerate(mask_array):
            mask_array_uint = mask_array_temp[0].astype(bool)
            
            # Create a copy of the original image
            masked_rgb = image_rgb.copy()
            
            # Set areas outside the mask to black (optional)
            masked_rgb[~mask_array_uint] = [0, 0, 0]
            
            # Save the segmented image using Matplotlib, ensuring original color is retained
            plt.imshow(masked_rgb)
            plt.axis('off')  # Turn off axis
            plt.savefig(f"{common_filename}_Segment_{i + 1}_RGB.png", bbox_inches='tight', pad_inches=0)
            plt.close()

    @staticmethod
    def export_original_image(image_rgb, common_filename):
        """Exports the original RGB image to check color correctness"""
        cv2.imwrite(f"{common_filename}_Original.png", image_rgb)

    @staticmethod
    def save_segmented_temp_matrix(temp_matrix, mask_array, common_filename):
        """Saves each segmented temperature matrix as an image with inverted Spectral colormap and black background outside the segment"""
        # Resize the mask to match the temperature matrix dimensions (256x192)
        temp_matrix_shape = temp_matrix.shape

        for i, mask_array_temp in enumerate(mask_array):
            mask_array_uint = mask_array_temp[0].astype(bool)

            # Resize the mask to match the dimensions of the temperature matrix
            resized_mask = cv2.resize(mask_array_uint.astype(np.uint8), (temp_matrix_shape[1], temp_matrix_shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

            # Apply the resized mask to the temperature matrix
            temp_segment = np.zeros_like(temp_matrix)
            temp_segment[resized_mask] = temp_matrix[resized_mask]

            # Set threshold values for the temperature range
            min_temp = 15
            max_temp = 40

            # Clip temperature values outside the range of 15 to 40
            temp_segment = np.clip(temp_segment, min_temp, max_temp)

            # Set all areas outside the segment to black
            background = np.zeros_like(temp_segment)
            background[resized_mask] = temp_segment[resized_mask]

            # Save the temperature matrix segment as an image with inverted Spectral colormap
            plt.imshow(background, cmap='Spectral_r', vmin=min_temp, vmax=max_temp)  # Use 'Spectral_r' to invert the colormap
            plt.colorbar(label='Temperature (C)')
            plt.savefig(f"{common_filename}_Segment_{i + 1}_Temperature.png")
            plt.close()

    @staticmethod
    def _generate_temperature_array_csv(mask_array_uint, temp_matrix):
        # Shrink mask image from 640x480 to 256x192
        mask_shrink = cv2.resize(mask_array_uint.astype(np.uint8), dsize=(256, 192), interpolation=cv2.INTER_NEAREST)

        mask_bool = mask_shrink.astype(bool)
        temp_numpy = temp_matrix[mask_bool]

        temp_list = [float(format(temp_value, '.3f')) for temp_value in temp_numpy]
        temp_list_filtered = list(filter(lambda x: x > 0.0, temp_list))

        if len(temp_list_filtered) == 0:
            temp_list_filtered = [0.0]

        return temp_list_filtered
