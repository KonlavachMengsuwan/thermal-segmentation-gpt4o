# -*- coding: utf-8 -*-
# 1, import from installed library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# 2, import from source code
import src._util as _util
import src._common as _com

# 3, import from python library
import argparse

##############################################################

def mask_process(mask_type, use_temp, filename_ext):
    """Main function to process mask and temperature data and generate images"""
    _util.resize_image(1280)  # Resize image if necessary

    # Generate mask array
    mask_array = _com.var.gen_mask.exec_model()

    if len(mask_array) == 0:
        return

    # Modify mask array
    mask_array = _com.var.modify_mask.exec_model(mask_array)

    # Load the original RGB image
    image_rgb = cv2.imread(_com.var.common_filename + filename_ext)

    # Export the original image to check correctness
    _com.var.gen_temp.export_original_image(image_rgb, _com.var.common_filename)

    # Save the segmented RGB images
    _com.var.gen_temp.save_segmented_rgb(image_rgb, mask_array, _com.var.common_filename)

    # Calculate temperature and export boxplot and CSV
    temperature_array = _com.var.gen_temp.exec_model(mask_array)

    if use_temp:
        print("Generating boxplot and CSV...")
        _com.var.gen_graph.exec_model(temperature_array)
        print("Boxplot and CSV have been generated successfully!")

    # Generate additional mask images with borders and labels
    generate_additional_images(mask_array, filename_ext)

    # Generate new image with median value labels
    generate_median_label_image(mask_array, temperature_array, filename_ext)


def generate_additional_images(mask_array, filename_ext):
    """Generate images with mask borders of thickness 5 and 10, and match number labels to `sam_withMask.png`."""
    # Load the image in BGR format (default for OpenCV)
    image_bgr = cv2.imread(_com.var.common_filename + filename_ext)

    # Label properties to match `sam_withMask.png`
    label_font = cv2.FONT_HERSHEY_DUPLEX
    label_font_scale = 0.7
    label_thickness_buffer = 3  # White buffer around text
    label_thickness_text = 1    # Main text thickness

    for thickness in [5, 10]:
        # Make a copy of the original BGR image for each thickness
        bordered_image = image_bgr.copy()
        for i, mask_temp in enumerate(mask_array):
            mask = mask_temp[0].astype(bool)
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            color = tuple(_com.const.GEN_TEMP_MASK_COLOR_ARRAY[i])

            # Draw the contour with the specified thickness
            cv2.drawContours(bordered_image, contours, -1, color, thickness=thickness)

            # Calculate the centroid for the number label
            moments = cv2.moments(mask.astype(np.uint8))
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"]) + 10  # Offset for better positioning
            else:
                cx, cy = 10, 10  # Fallback position

            # Add the buffer and the main label with the same style as `sam_withMask.png`
            label_text = str(i + 1)
            # White buffer around the text
            cv2.putText(bordered_image,
                        text=label_text,
                        org=(cx, cy),
                        fontFace=label_font,
                        fontScale=label_font_scale,
                        color=(224, 224, 224),
                        thickness=label_thickness_buffer,
                        lineType=cv2.LINE_AA)
            # Main text in the mask color
            cv2.putText(bordered_image,
                        text=label_text,
                        org=(cx, cy),
                        fontFace=label_font,
                        fontScale=label_font_scale,
                        color=color,
                        thickness=label_thickness_text,
                        lineType=cv2.LINE_AA)

        # Save the image in its original BGR format without converting to RGB
        cv2.imwrite(_com.var.common_filename + f'_withMaskBorder_thickness_{thickness}.png', bordered_image)



def generate_median_label_image(mask_array, temperature_array, filename_ext):
    """Generate an image with mask borders and label each segment with the median temperature."""
    # Load the image in BGR format (default for OpenCV)
    image_bgr = cv2.imread(_com.var.common_filename + filename_ext)

    # Label properties
    label_font = cv2.FONT_HERSHEY_DUPLEX
    label_font_scale = 0.7
    label_thickness_buffer = 3  # White buffer around text
    label_thickness_text = 1    # Main text thickness
    label_color = (0, 0, 0)  # Black for all labels

    # Make a copy of the original BGR image for the median value labeling
    median_labeled_image = image_bgr.copy()

    for i, mask_temp in enumerate(mask_array):
        mask = mask_temp[0].astype(bool)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color = tuple(_com.const.GEN_TEMP_MASK_COLOR_ARRAY[i])

        # Draw the contour with thickness 5
        cv2.drawContours(median_labeled_image, contours, -1, color, thickness=5)

        # Calculate the centroid for the label
        moments = cv2.moments(mask.astype(np.uint8))
        if moments["m00"] != 0:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"]) + 10  # Offset for better positioning
        else:
            cx, cy = 10, 10  # Fallback position

        # Get the median temperature value for this segment
        median_value = str(round(np.median(temperature_array[i][0]), 1))

        # Add the buffer and the main label with the median temperature value
        cv2.putText(median_labeled_image,
                    text=median_value,
                    org=(cx, cy),
                    fontFace=label_font,
                    fontScale=label_font_scale,
                    color=(224, 224, 224),  # White buffer around the text
                    thickness=label_thickness_buffer,
                    lineType=cv2.LINE_AA)
        cv2.putText(median_labeled_image,
                    text=median_value,
                    org=(cx, cy),
                    fontFace=label_font,
                    fontScale=label_font_scale,
                    color=label_color,  # Black for the main text
                    thickness=label_thickness_text,
                    lineType=cv2.LINE_AA)

    # Save the new image with median labels
    cv2.imwrite(_com.var.common_filename + '_withMaskBorder_medianValue.png', median_labeled_image)





def main():
    """Main function to initiate the process"""
    _com.var.gen_mask.load_model()  # Initialize process by loading the model
    filelist = _util.generate_filelist()

    for _, filename in enumerate(filelist):
        _com.var.common_filename = filename

        if not _util.check_require_files():
            continue

        print(" ")
        print("----------------------------------------------------------------")
        print(f"Filename: {_com.var.common_filename}{_com.var.filename_ext}     ProcessType: {_com.var.mask_type}")
        mask_process(_com.var.mask_type, _com.var.use_temp, _com.var.filename_ext)
        print(" ")


def parse_opt():
    """Parse command-line options"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="input5", help='image directory')
    parser.add_argument('--mask_type', type=str, default="sam", help='sam, cityscapes, ade20k, coco, all, xlsx')
    parser.add_argument('--use_pickle', action='store_true', help='save segments to pickle')
    parser.add_argument('--min_area_percent', type=float, default=2, help='ignore areas that are smaller than min_area_percent')
    parser.add_argument('--use_temp', action='store_true', help='calc temperature boxplot')
    parser.add_argument('--filename_ext', type=str, default='_Visual_IR.jpeg', help='extension of image file')
    opt = parser.parse_args()
    return opt

######################################################################

if __name__ == "__main__":
    opt = parse_opt()

    _com.const = _com.ConstVariable.generate_instance()
    _com.var = _com.GlobalVariable.generate_instance(opt=opt)

    main()
