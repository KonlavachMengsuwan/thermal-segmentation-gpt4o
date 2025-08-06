# 1, import from installed library
import numpy as np
import matplotlib.pyplot as plt

# 2, import from source code
import src._common as _com

# 3, import from python library
import statistics
import csv
from abc import ABCMeta, abstractmethod

#########################################################

class Gen_Graph(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def exec_model(self, temp_array):
        pass

# ----------------------------------------------------------------

class Gen_Graph_BoxPlot(Gen_Graph):

    def __init__(self):
        super().__init__()

    def exec_model(self, temp_array):
        temp_data = []
        temp_data_text = []
        temp_labels = []
        csv_list = []
        csv_segment_id = ["Segment ID"]
        csv_median_temp = ["Median Temperature"]

        flat = []
        for row in temp_array:
            for x in row[0]:
                flat.append(x)
        temp_min = min(flat)

        csv_list.append(["Filename", _com.var.common_filename + _com.var.filename_ext])
        csv_list.append([""])

        # Loop through the temperature arrays and segments
        for i, temp_list in enumerate(temp_array):
            boxplot = self._calc_boxplot_value(temp_list[0])
            temp_data_text.append(boxplot)
            temp_data.append(temp_list[0])

            if temp_list[1] != '':
                temp_labels.append(temp_list[1])
                csv_segment_id.append(temp_list[1])
            else:
                temp_labels.append(str(i + 1))
                csv_segment_id.append(i + 1)

            csv_median_temp.append(boxplot[2])

        csv_list.append(csv_segment_id)
        csv_list.append(csv_median_temp)

        # Create the boxplot
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        bp = ax.boxplot(temp_data,
                        sym="",
                        vert=True,
                        patch_artist=True,
                        widths=0.8,
                        labels=temp_labels)

        # Assign colors based on the segment order in the mask_color_array2
        for i, box in enumerate(bp['boxes']):
            color = _com.var.mask_color_array2[i]  # Get the color from mask_color_array2
            box.set_facecolor(color)  # Set the box color

        for median in bp['medians']:
            median.set_color('black')

        # Add median values as text below each boxplot
        x_width = 1.0 / len(temp_data_text)
        x_start = x_width * 0.5
        for pos, temp_text in enumerate(temp_data_text):
            ax.text(x=x_start + x_width * pos, y=0.01, s=str(temp_text[2]), size='x-small', horizontalalignment="center", transform=ax.transAxes)

        ax.set_xlabel('segment')
        ax.set_ylabel('temperature')
        ax.set_title(_com.var.common_filename + _com.var.filename_ext)
        
        # Save the boxplot as a PNG file
        plt.savefig(_com.var.common_filename + '_' + _com.var.mask_type + '_Temperature_boxplot.png')

        # Save the temperature data as a CSV file
        with open(_com.var.common_filename + '_' + _com.var.mask_type + '_Temperature_boxplot.csv', 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerows(csv_list)

    @staticmethod
    def _calc_boxplot_value(array: list) -> list:
        """Calculate quartiles and other boxplot statistics."""
        q25, q50, q75 = np.percentile(array, [25, 50, 75])
        iqr = q75 - q25

        q25 = float(format(q25, '.1f'))
        q50 = float(format(q50, '.1f'))
        q75 = float(format(q75, '.1f'))

        new_array = [i for i in array if (q25 - 1.5 * iqr) <= i <= (q75 + 1.5 * iqr)]

        median_mean = round(statistics.mean([min(new_array), max(new_array)]), 2)

        return [min(new_array), q25, q50, q75, max(new_array), median_mean]
