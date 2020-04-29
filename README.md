# Histology
Naive classification of histology slides.

Keras/Tensor Flow naive classification of cancer outcomes using untrimmed or cleaned histology slides.
Data sourced from National Cancer Institute - Genomic Data Commons (https://portal.gdc.cancer.gov/)

## Notes/To Do
find the tool **nconvert** to possibly convert the raw .svs to .tiff or .jpg to allow system to run with large raw files

nconvert -out tiff -multi -dpi 100 -c 4 -keepdocsize -keepfiledate mysource.tif

otherwise work with openslide and a pipeline for segmenting/tiling the slides.


## Random thoughts

~~try back here at the SCNN
https://github.com/CancerDataScience/SCNN~~

the SCNN container is not useful, scripts are binary and not readable or not available

Downloading the data
GDC does not currently enable direct querying of the TCGA diagnostic images for a specific project. To generate a list of the files to download, you have to first generate a manifest of all whole-slide images in TCGA (both frozen and diagnostic), filter the frozen section images in this list, and then match the identifiers against the sample identifiers (TCGA-##-####) for the project(s) of interest.

The manifest for all TCGA whole-slide images can be generated using the GDC Legacy Archive query.

Rows containing diagnostic image files can be identified using the Linux command line

cut -d$'\t' -f 2 gdc_manifest.txt | grep -E '\.*-DX[^-]\w*.'
After matching the slide filenames against the sample IDs from the clinical data for the project(s) of interest, the relevant filenames can be used with the GDC Data Transfer Tool or the GDC API.

go to portable disk and download

cd /media/matt/0D830EB30D830EB3/stomach 
./gdc-client -m download ~/Git_repos/Histology/data/stomach_march/gdc_manifest.2020-01-26.txt

Extracting regions of interest  

~~Regions of interest can be extracted using the python script generate_rois.py. This script consumes a tab-delimited text file describing the whole-slide image files, ROI coordinates, desired size and magnification for extracted ROIs, and then generates a collection of ROI .png images. These images are transformed into a binary for model training and testing by the software described below.

Note: region extraction depends on the OpenSlide library.

# svs images
https://github.com/debuggingtissue/deepslide-svs-wsi-to-jpeg-patch-generator
https://github.com/BMIRDS/deepslide

pip install -r scripts/requirements.txt

python wsi_svs_to_jpeg_tiles.py  -i /path/to/svs_image_directory -o /path/to/jpeg_tiles_folder

# move files
find . -name '*.svs' -exec mv {} /path/to/single/target/directory/ \;
mv **/*.svs /path/to/single/target/directory/

highlight is the term in the clinical file to match 
*TCGA-HF-7134*-01Z-00-DX1.A84E2A6E-05F4-4A4B-8F64-9B9D1B33838E.svs



