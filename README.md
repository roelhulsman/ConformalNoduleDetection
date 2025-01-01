# Conformal Risk Control for Pulmonary Nodule Detection
Author: Roel Hulsman, Valentin Comte, Lorenzo Bertolini, Tobias Wiesenthal, Antonio Puertas Gallardo and Mario Ceresa. 

European Commission, Joint Research Centre (JRC), Ispra, Italy.


## Description
Quantitative tools are increasingly appealing for decision support in healthcare, driven by the growing capabilities of advanced AI systems. However, understanding the predictive uncertainties surrounding a tool's output is crucial for decision-makers to ensure reliable and transparent decisions. In this paper, we present a case study on pulmonary nodule detection for lung cancer screening, enhancing an advanced detection model with an uncertainty quantification technique called conformal risk control (CRC). We demonstrate that prediction sets with conformal guarantees are attractive measures of predictive uncertainty in the safety-critical healthcare domain, allowing end-users to achieve arbitrary validity by trading off false positives and providing formal statistical guarantees on model performance. Among ground-truth nodules annotated by at least three radiologists, our model achieves a sensitivity that is competitive with that generally achieved by individual radiologists, with a slight increase in false positives. Furthermore, we illustrate the risks of using off-the-shelve prediction models when faced with ontological uncertainty, such as when radiologists disagree on what constitutes the ground truth on pulmonary nodules. 


## How To Run
Follow the steps below to fully reproduce the results of this project. We recommend using Conda as a package manager. 

Steps:
1. Clone this repository and set it as working directory. 
2. Create and activate the virtual environment using the following command lines. Note that the environment progressed from a previous project and is thus not a minimal working product. 
```
conda env create -f environment.yml
conda activate conformal_nodule_detection
```
3. Download the LIDC-IDRI images to the folder `data/` from https://www.cancerimagingarchive.net/collection/lidc-idri/. Note that you need the NBIA Data Retriever. 
4. Extract annotation data from the LIDC-IDRI images, then transform the images with nodules not in LUNA from .dcm to .nii format. Do so by running the following command lines. This populates the folder `data/images/`. We owe credit to Project-MONAI for the transformation script, since it is a small adaptation from the example code provided in the MONAI Tutorials detection repository (Apache 2.0 license). 
```
python scripts/preprocess_lidc.py
python scripts/transform_images_dicom_nii.py
gunzip -r data/images
```
5. [Optional] If you have limited disk space, feel free to now delete `data/manifest-1600709154662/` to save around 121GB. You won't need the raw LIDC-IDRI images anymore. The folder `data/images/` will roughly triple in size because of output from the prediction pipeline in the next steps, so make sure you have enough disk space available. 
6. Download the detection and segmentation models from MONAI by running the following command lines. This creates a folder `apps/`. We owe credit again for the do_infer.py script, adapted from MONAILabel (Apache 2.0 license).
```
monailabel apps --name monaibundle --download --output apps
python scripts/do_infer.py -t download --det_nd --app_dir apps/monaibundle/
python scripts/do_infer.py -t download --seg_ct --app_dir apps/monaibundle/
```
7. In the file `apps/monaibundle/model/lung_nodule_ct_detection/configs/inference.json`, change the following lines. This is to make sure to resample input images when running the detection model and to extract filtered anchor boxes. Change line 2 from ``"whether_raw_luna16": false,`` to ``"whether_raw_luna16": true,``. Change line 77 from ``"$@detector.set_box_selector_parameters(score_thresh=0.02,topk_candidates_per_level=1000,nms_thresh=0.22,detections_per_img=300)",`` to ``"$@detector.set_box_selector_parameters(score_thresh=0,topk_candidates_per_level=100000,nms_thresh=0.22,detections_per_img=100000)",``. 
8. Run the prediction pipeline by running the following command lines. Note this is a Bash script, tested on a Linux machine, and might cause problems on different systems. Running the detection model takes roughly 17 minutes per image on one A100 GPU. Running the segmentation model takes roughly 2 mintes per image on one A100 GPU. 
```
bash scripts/run_prediction_model.sh data/images/
```
9. [Optional] If an error occurs during the prediction pipeline and you would like to re-run the prediction model on all images, you can use the command ``python scripts/clear_model_output.py data/images/`` to clear prediction output.
10. Run the matching and conformal pipeline by running the following command lines. This populates the `results/` folder. Running the matching should take a couple minutes on one A100 GPU. Computing the risk takes a couple hours, since for each image (represented by +-30k+ anchor boxes) several metrics (e.g. sensitivity) are computed for each of 1000 values of the confidence threshold. This computation is done a-priori to speed up the later conformal script. All figures can be replicated using the conformal script. It takes a couple hours to run due to repeating the conformal procedure over R=10000 seeds over 4 datasets. 
```
python scripts/do_matching.py data/images/
python scripts/compute_risk.py 
python scripts/do_conformal.py
```


## How To Use
To visualize CT scans and predicted nodules, we recommend using 3D slicer (download at https://download.slicer.org/). 


## Citing Our Work
```
@article{hulsman2024conformal,
      title={Conformal Risk Control for Pulmonary Nodule Detection}, 
      author={Roel Hulsman and Valentin Comte and Lorenzo Bertolini and Tobias Wiesenthal and Antonio Puertas Gallardo and Mario Ceresa},
      year={2024},
      journal={arXiv preprint arXiv:2412.20167},
}
```


## Credits
We acknowledge the National Cancer Institute and the Foundation for the National Institutes of Health for their critical role in the creation of the free publicly available LIDC-IDRI dataset used in this study. We further acknowledge Project-MONAI for their efforts in providing a set of open-source frameworks for AI research in medical imaging, such as MONAILabel.


## License
Apache 2.0 License.


