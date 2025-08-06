# Thermal Segmentation and Land Surface Classification with SAM × GPT-4o

![Segmented_Thermal_Temperature_V6](https://github.com/user-attachments/assets/6dd82968-6563-4c18-9057-556f2810bb3d)

This repository contains a Python pipeline for segmenting RGB images using the Segment Anything Model (SAM) and classifying each segment into land surface categories using GPT-4o. The project then links each segment with thermal data from a handheld infrared camera to generate fine-scale temperature profiles across different land surface types.

## Directory Structure

thermal-segmentation-gpt4o/
├── input/
│   ├── *.jpg / *.png               # RGB images
│   └── *.csv                       # Corresponding thermal matrix (.csv)
├── src/
│   ├── _common.py
│   ├── _gen_graph.py
│   ├── _gen_mask.py
│   ├── _gen_temp.py
│   ├── _modify_mask.py
│   └── _util.py
├── temp_rgb_array.txt             # Path pairs of RGB image and thermal .csv
├── requirements.txt               # Python dependencies
├── test2.py                       # Main script
├── README.md                      # Project overview

## Getting Started

1. Clone the Repository

    git clone https://github.com/KonlavachMengsuwan/thermal-segmentation-gpt4o.git
    cd thermal-segmentation-gpt4o

2. Install Dependencies

It is recommended to use a virtual environment (e.g., venv or conda):

    pip install -r requirements.txt

3. Prepare Input Data

- Place RGB images and corresponding .csv thermal matrices inside the input/ folder.
- Fill temp_rgb_array.txt with tab-separated file paths for each image–matrix pair, one per line:

    input/img001.jpg    input/img001.csv  
    input/img002.jpg    input/img002.csv

4. Run the Main Script

    python test2.py

## What This Code Does

- Segments RGB images using the Segment Anything Model (SAM).
- Prompts GPT-4o to classify each segment into a hierarchical land surface category.
- Integrates segment-level classification with median temperature from the corresponding thermal matrix.
- Outputs include temperature-labeled segments, statistics per land surface class, and visualizations.

## Example Outputs

(You may add visual examples here in future)
- Segmented RGB image
- Temperature overlay by segment
- Classification table (Level-1 and Level-2)
- Temperature distribution plots per land surface class

## License

This project is licensed under the MIT License – see the LICENSE file for details.

## Citation and Funding Acknowledgement

If you use this code, please cite the associated publication (link and DOI coming soon via Zenodo).

This work was supported by the Federal Ministry of Education and Research (BMBF – Bundesministerium für Bildung und Forschung) project “Multi-modal data integration, domain-specific methods, and AI to strengthen data literacy in agricultural research” (16DKWN089), WIR! - Land - Innovation - Lausitz (LIL) project “Landscape Innovations in Lausitz (Lusatia) for Climate-adapted Bioeconomy and nature-based Bioeconomy-Tourism” (03WIR3017A), and the Brandenburg University of Technology Cottbus-Senftenberg (BTU).
