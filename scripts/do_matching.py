import sys, os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
	
	
def intersection_over_union(boxA, boxB):

	assert (len(boxA) == 6) & (len(boxB) == 4), 'Error - Length of boxes should be 6 and 4'

	# determine the (x, y, z)-coordinates of the intersection rectangle
	xA = max(boxA[0]-boxA[3]/2, boxB[0]-boxB[3]/2)
	yA = max(boxA[1]-boxA[4]/2, boxB[1]-boxB[3]/2)
	zA = max(boxA[2]-boxA[5]/2, boxB[2]-boxB[3]/2)
	xB = min(boxA[0]+boxA[3]/2, boxB[0]+boxB[3]/2)
	yB = min(boxA[1]+boxA[4]/2, boxB[1]+boxB[3]/2)
	zB = min(boxA[2]+boxA[5]/2, boxB[2]+boxB[3]/2)
    
	if xB < xA or yB < yA or zB < zA:
    		return 0
	else:
		# compute the area of intersection rectangle
    		interArea = (xB - xA) * (yB - yA) * (zB - zA)

    		# compute the area of both the prediction and ground-truth
    		# rectangles
    		boxAArea = pow(boxA[3], 3)
    		boxBArea = pow(boxB[3], 3)

    		# compute the intersection over union by taking the intersection
    		# area and dividing it by the sum of prediction + ground-truth
    		# areas - the interesection area
    		iou = interArea / float(boxAArea + boxBArea - interArea)

    		# return the intersection over union value
    		return iou
	
	
def process_img(model_output_path, img_id, annotations, use_seg=True, thresh=0):
	
	if use_seg:	
		f = open(model_output_path + img_id + '/test_labels/labels_det_nd.json')
	else:
		f = open(model_output_path + img_id + '/test_labels/labels_det_nd_old.json')
	model_output = json.load(f)
	
	true_boxes = [row.tolist() for ind, row in annotations.loc[annotations['seriesuid']==img_id, annotations.columns.isin(['coordX', 'coordY', 'coordZ', 'diameter_mm'])].iterrows()]
	nr_nodules = len(true_boxes)
	
	nr_boxes = len(model_output['box'])
	boxes_conf = model_output['label_score']
	boxes_coord = model_output['box']
	
	boxes_ground_truth = [None]*nr_boxes
	boxes_iou = [None]*nr_boxes
	boxes_label = [0]*nr_boxes
		
	if nr_nodules>0:
		D = np.zeros((nr_boxes, nr_nodules)) # The distance matrix.
		for i in range(nr_boxes):
			for j in range(nr_nodules):
        			D[i,j] = intersection_over_union(boxes_coord[i], true_boxes[j])
        			
		boxes_iou = [iou if iou>thresh else None for iou in np.max(D, axis=1)]
		boxes_ground_truth = [true_boxes[i] for i in np.argmax(D, axis=1)]
		boxes_ground_truth = [boxes_ground_truth[i] if boxes_iou[i] is not None else None for i in range(nr_boxes)]
		for i in range(nr_nodules):
			if true_boxes[i] in boxes_ground_truth:
				boxes_label[boxes_ground_truth.index(true_boxes[i])] = 1
		
				
	img_results = pd.DataFrame({
		'img_id': img_id, 
		'box_id': list(range(nr_boxes)),
		'box_conf': boxes_conf, 
		'box_coord': boxes_coord,
		'ground_truth': boxes_ground_truth,
		'ground_truth_iou': boxes_iou,
		'box_label': boxes_label,
	})

	return img_results


def plot_nod_conf(results, nr_anns, output_dir):

	sns.set_context("talk")
	plt.figure(figsize=(15,10))
	
	plot = sns.histplot(results[results['box_label']==1], x='box_conf', stat='probability')
	plot.set(xlabel='Box Confidence', ylabel='Density', title='Model Confidence in True Nodules')
	fig = plot.get_figure()
	fig.savefig(output_dir + 'plot_nod_confidence.png'.format(nr_anns))
	print('Figure - Nod Confidence Density - Saved')
	plt.clf()
	


def aggregate_results(model_output_path, annotations_path, output_dir, nr_anns, use_seg=True, thresh=0):
	
	annotations = pd.read_csv(annotations_path)
	annotations = annotations.loc[:,['seriesuid', 'coordX', 'coordY', 'coordZ', 'diameter_mm']]
	print(annotations)

	results = pd.DataFrame()
	nr_boxes_processed = 0
	nr_nodules_processed = 0
	nr_img_processed = 0
	nr_nodules_missed_total = 0
	nodules_missed = pd.DataFrame()
	
	for img_id in annotations['seriesuid'].unique():
		print('Processing ', img_id)
		
		img_annotations = annotations.loc[annotations['seriesuid']==img_id]
		img_results = process_img(model_output_path=model_output_path, img_id=img_id, annotations=img_annotations, use_seg=use_seg, thresh=thresh)
		results = pd.concat([results, img_results], axis=0, ignore_index=True)
		print(img_results.head(25))
		
		#print(img_results[img_results['ground_truth_iou'] > thresh])
		print(img_results[img_results['box_label'] == 1])
		print(img_results['ground_truth'].value_counts())
		
		nr_boxes = len(img_results)
		nr_nodules_assigned = sum(img_results['box_label'])
		nr_nodules_missed = len(img_annotations) - nr_nodules_assigned
		print('Number of boxes processed: ', nr_boxes)
		print('Number of nodules assigned: ', nr_nodules_assigned)
		print('Number of nodules missed: ', nr_nodules_missed, '\n')
		
		if nr_nodules_missed > 0: 
			print('Warning - Nodule(s) missed (%s)!' %(nr_nodules_missed))
			print(img_annotations)
			nodules_missed = pd.concat([nodules_missed, img_annotations], axis=0, ignore_index=True)
		
		nr_img_processed += 1
		nr_boxes_processed += nr_boxes
		nr_nodules_processed += nr_nodules_assigned
		nr_nodules_missed_total += nr_nodules_missed
		print('Number of images processed total: ', nr_img_processed)
		print('Number of nodules assigned total: ', nr_nodules_processed)
		print('Number of nodules missed total: ', nr_nodules_missed_total, '\n')
	
	print('Number of images processed: ', nr_img_processed)
	print('Number of boxes processed: ', nr_boxes_processed)
	print('Number of nodules assigned: ', nr_nodules_processed)
	print('Number of nodules missed: ', nr_nodules_missed_total)
	
	results = results.sort_values(by = ['img_id', 'box_id'], ascending = [True, True], ignore_index=True)
	results_nod = results[results['box_label']==1]
	
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
		os.makedirs(output_dir + 'figures/')
	
	if use_seg:
		results.to_csv(output_dir + 'results_matching.csv', index=False)
		results_nod.to_csv(output_dir + 'results_matching_nod.csv', index=False)
	else:
		results.to_csv(output_dir + 'results_matching_no_seg.csv', index=False)
		results_nod.to_csv(output_dir + 'results_matching_no_seg_nod.csv', index=False)	
		
	print('Results saved at: ', output_dir)
	
	plot_nod_conf(results, nr_anns, output_dir + 'figures/')
	
	return results
	
	
if __name__ == '__main__':
	
	model_output_path = sys.argv[1]
	
	use_seg = True
	thresh=0
	
	for nr_anns in [1,2,3,4]:
		results = aggregate_results(model_output_path=model_output_path, annotations_path='data/annotations/annotations_{}.csv'.format(nr_anns), output_dir='results/set_{}/'.format(nr_anns), nr_anns=nr_anns, use_seg=use_seg, thresh=thresh)
	
	

	
	
	
	
	
	
	
	
