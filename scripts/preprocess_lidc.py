import os
import json
import pylidc as pl
import pandas as pd
import numpy as np
import SimpleITK as sitk


def load_image(uid):

	print(uid)	
	scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == uid).first()
	scan_id = scan.patient_id
	uid_last_digits = uid[-5:]
	uid_dir = os.popen('find ./data/manifest-1600709154662/LIDC-IDRI/%s/*/*%s -type d' %(scan_id, uid_last_digits)).read().split('\n')[0]
	print(uid_dir)
	
	series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(uid_dir)
	series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(uid_dir, series_IDs[0])
	series_reader = sitk.ImageSeriesReader()
	series_reader.SetFileNames(series_file_names)
	series_reader.MetaDataDictionaryArrayUpdateOn()
	series_reader.LoadPrivateTagsOn()
	img = series_reader.Execute()
	
	origin = img.GetOrigin()
	spacing = img.GetSpacing()
	direction = np.array(img.GetDirection())[[0,4,8]]
	size = img.GetSize()	
	
	print('Origin: ', origin)
	print('Spacing: ', spacing)
	print('Direction: ', direction)
	print('Size: ', size)
	
	return origin, spacing, direction, size


def process_pylidc_data(output_path):

	scans = pl.query(pl.Scan).filter()
	print('Amount of scans: ', scans.count())

	results_img = pd.DataFrame(columns=['scan_id', 'seriesuid', 'nr_slices', 'slice_thickness', 'slice_spacing', 'pixel_spacing', 'img_spacing', 'origin', 'direction', 'size', 'nr_annotations'])
	results_ann_clustered = pd.DataFrame(columns=['scan_id', 'seriesuid', 'nr_slices', 'slice_thickness', 'slice_spacing', 'pixel_spacing', 'img_spacing', 'origin', 'direction', 'size', 'nod_id', 'nr_annotations', 'coordX', 'coordY', 'coordZ', 'diameter_mm'])
	results_ann_full = pd.DataFrame(columns=['scan_id', 'seriesuid', 'nr_slices', 'slice_thickness', 'slice_spacing', 'pixel_spacing', 'img_spacing', 'origin', 'direction', 'size', 'nod_id', 'annotation_id', 'coordX', 'coordY', 'coordZ', 'diameter_mm'])

	i=0
	j=0
	k=0
	for scan in scans:
		
		scan_id = scan.patient_id
		scan_seriesuid = scan.series_instance_uid
		scan_nr_slices = scan.sorted_dicom_file_names.count('dcm')
		scan_slice_thickness = scan.slice_thickness
		scan_slice_spacing = scan.slice_spacing
		scan_pixel_spacing = scan.pixel_spacing
		scan_nr_anns = len(scan.annotations)
		
		scan_origin, scan_spacing, scan_direction, scan_size = load_image(scan_seriesuid)
		
		nods = scan.cluster_annotations()
		results_scan = [scan_id] + [scan_seriesuid] + [scan_nr_slices] + [scan_slice_thickness] + [scan_slice_spacing] + [scan_pixel_spacing] + [scan_spacing] + [scan_origin] + [scan_direction] + [scan_size] + [scan_nr_anns] 
		results_img.loc[i] = results_scan
		i += 1
		print(results_scan)
		
		l=0
		for nod in nods:
			nod_nr_anns = len(nod)
			nod_centroid = np.mean([ann.centroid for ann in nod], axis=0)
			nod_diam = np.mean([ann.diameter for ann in nod])
			results_nod = [scan_id] + [scan_seriesuid] + [scan_nr_slices] + [scan_slice_thickness] + [scan_slice_spacing] + [scan_pixel_spacing] + [scan_spacing] + [scan_origin] + [scan_direction] + [scan_size] + [l] + [nod_nr_anns] + nod_centroid.tolist() + [nod_diam]
			results_ann_clustered.loc[j] = results_nod
			j += 1
			l += 1
			print(results_nod)
			
			m=0
			for ann in nod:
				ann_centroid = ann.centroid
				ann_diam = ann.diameter
				results_ann = [scan_id] + [scan_seriesuid] + [scan_nr_slices] + [scan_slice_thickness] + [scan_slice_spacing] + [scan_pixel_spacing] + [scan_spacing] + [scan_origin] + [scan_direction] + [scan_size] + [l] + [m] + ann_centroid.tolist() + [ann_diam]
				results_ann_full.loc[k] = results_ann
				print(results_ann)
				k += 1
				m += 1

	results_img.to_csv(output_path + 'pylidc_images.csv', index=False)
	results_ann_clustered.to_csv(output_path + 'pylidc_annotations.csv', index=False)
	results_ann_full.to_csv(output_path + 'pylidc_annotations_full.csv', index=False)
	
	return results_img, results_ann_clustered, results_ann_full


def transform_coord(uid, nods):
	origin, spacing, direction, size = load_image(uid)
		
	for i in nods.index:
		nod_pos = nods.loc[i, ['coordY', 'coordX', 'coordZ']]
		new_coord = direction*(nod_pos*spacing+direction*origin)
		nods.loc[i, ['coordX', 'coordY', 'coordZ']] = new_coord.tolist()
	print(nods)
		
	return nods

	
def process_coord_transforms(annotations, save_to_path=''):

	annotations_transformed = pd.DataFrame()
	
	for uid in annotations['seriesuid'].unique():
		nods = annotations[annotations['seriesuid']==uid]
		nods = transform_coord(uid, nods)
		annotations_transformed = pd.concat([annotations_transformed, nods], ignore_index=True)
	
	if save_to_path != '':
		annotations_transformed.to_csv(save_to_path, index=False)
		print('Saved at: ', save_to_path)
	
	return annotations_transformed
	
	
def filter_annotations(annotations, seriesuid_luna, save_to_path=''):

	min_diameter = 3
	max_slice_thickness = 3
	min_nr_anns = [1, 2, 3, 4]
	
	# Note in transforming from .dcm to .nrrd we lose an image due to unknown errors. This is an image with 2 nods. 
	exceptions = [
		'LIDC-IDRI-0793',
	]
	
	annotations_filtered = {}
	
	for nr_anns in min_nr_anns:
		annotations_filtered[nr_anns] = annotations[(~annotations['seriesuid'].isin(seriesuid_luna)) & (annotations['diameter_mm']>=min_diameter) & (annotations['slice_thickness']<=max_slice_thickness) & (annotations['nr_annotations']>=nr_anns) & (~annotations['scan_id'].isin(exceptions))].reset_index()
	
		if save_to_path != '':
			annotations_filtered[nr_anns].to_csv(save_to_path + '/annotations_{}.csv'.format(nr_anns), index=False)

		print('Nr imgs with nods with >{} annotations: '.format(nr_anns), len(set(annotations_filtered[nr_anns]['seriesuid'])))
		print('Nr nods with >{} annotations: '.format(nr_anns), len(annotations_filtered[nr_anns]))
		
	return annotations_filtered
		
		
def save_calibration_images_to_json(uids, save_to_path):

	results = []
	for uid in uids:
		scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == uid).first()
		scan_id = scan.patient_id
		uid_last_digits = uid[-5:]
	
		dir_name = os.popen('find ./data/manifest-1600709154662/LIDC-IDRI/%s/*/*%s -type d' %(scan_id, uid_last_digits)).read().split('/LIDC-IDRI/')[-1].split('\n')[-2]
		results.append({'image': dir_name})
		
	calibration_images = {'calibration': results}
    	
	with open(save_to_path + 'calibration_images.json', 'w') as f:
    		json.dump(calibration_images, f, ensure_ascii=False, indent=4)
    		print('Datasplit saved')
	
	return calibration_images
	
	

if __name__ == '__main__':

    	
    	data_path = 'data/manifest-1600709154662/LIDC-IDRI/'
    	annotations_path = 'data/annotations/'
    	calibration_images_path = 'data/'
    	
    	# These two lines are used to extract annotation data from LIDC images using pylidc package, and subsequently transform image coordinates to real world coordinates. 
    	# You can use pre-run dataframes to save computation time, therefore these lines are commented out. 
    	#pylidc_images, pylidc_annotations, pylidc_annotations_full = process_pylidc_data(annotations_path)
    	#pylidc_annotations_transformed = process_coord_transforms(pylidc_annotations, annotations_path + 'pylidc_annotations_transformed.csv')
    	
    	pylidc_images = pd.read_csv(annotations_path + 'pylidc_images.csv')
    	pylidc_annotations = pd.read_csv(annotations_path + 'pylidc_annotations.csv')
    	annotations_transformed = pd.read_csv(annotations_path + 'pylidc_annotations_transformed.csv')
    	
    	print(pylidc_annotations)
    	print(annotations_transformed)
    	
    	annotations_luna = pd.read_csv(annotations_path + 'LUNA16_annotations.csv')
    	seriesuid_luna = list(set(annotations_luna['seriesuid']))
    	
    	annotations_filtered = filter_annotations(annotations_transformed, seriesuid_luna, annotations_path)
    	
    	seriesuid_not_luna = list(set(pylidc_images[(~pylidc_images['seriesuid'].isin(seriesuid_luna))]['seriesuid']))
    	seriesuid_not_luna_nod = list(set(annotations_filtered[1]['seriesuid']))
    	
    	print('Nr images total: ', len(list(set(pylidc_images['seriesuid']))))
    	print('Nr images LUNA: ', len(seriesuid_luna))
    	print('Nr images not LUNA: ', len(seriesuid_not_luna))
    	print('Nr images not LUNA with nodules: ', len(seriesuid_not_luna_nod))
    	
    	print('Max nodules per image (LUNA): ', annotations_luna['seriesuid'].value_counts().max())
    	for i in range(1, 5):
    		print('Max nodules per image (Set %s): ' %(i), annotations_filtered[i]['seriesuid'].value_counts().max())
    	
    	# We create a .json file with all images with nodules not in LUNA. This needs to adhere to a certain format due to the scripts we adapted from the MONAI Tutorials repository to transform images from .dcm to .nii format. 
    	calibration_images = save_calibration_images_to_json(seriesuid_not_luna_nod, calibration_images_path)

    	
    	
    	



	

