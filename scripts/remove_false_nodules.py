# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import SimpleITK as sitk
import numpy as np
import json
import matplotlib.pyplot as plt
import skimage
import sys, os

ID = sys.argv[1]
img_id = os.path.basename(os.path.normpath(ID))
print('IMAGE ID: ', img_id)
	
print("Checking nodules in %s" %ID)

#load files
labels = sitk.ReadImage("%s/test_labels/%s_seg_ct.nii" %(ID, img_id))
labels_np = sitk.GetArrayFromImage(labels)

#select lungs seg
labels_np[labels_np<13]=0
labels_np[labels_np>17]=0
labels_np[labels_np!=0]=1

f = open("%s/test_labels/labels_det_nd.json" %(ID))

data = json.load(f)
data_thresh = data.copy()


if data["box"] != []:

    nodules_pos = []
    for box in data["box"]:
        nodules_pos.append(box[0:3])
        
    spacing = labels.GetSpacing()
    origin = labels.GetOrigin()
    direction = labels.GetDirection()
        
    nd = (np.array(direction)[[0,4,8]]*np.asarray(nodules_pos)-np.array(direction)[[0,4,8]]*origin)/spacing
    dX, dY, dZ = labels_np.shape[0]//20+1, labels_np.shape[1]//20+1, labels_np.shape[2]//20+1
    
    print('Spacing: ', spacing)
    print('Origin: ', origin)
    print('Direction: ', direction)
    print('dX: ', dX)
    print('dY: ', dY)
    print('dZ: ', dZ)
    print(nd)
    
    boxes = {'box': data["box"], 'label': data['label'], 'label_score': data['label_score']}
    boxes_new = {'box': [], 'label': [], 'label_score': []}
    boxes_thresh = {'box': [], 'label': [], 'label_score': []}
    boxes_removed = {'box': [], 'label': [], 'label_score': []}
    boxes_error = {'box': [], 'label': [], 'label_score': []}
    
    for i in range(len(nodules_pos)):
        ndX, ndY, ndZ = int(nd[i,2]),int(nd[i,1]),int(nd[i,0])
        try:
        	if labels_np[max(0,ndX-dX):min(ndX+dX,labels_np.shape[0]),max(0,ndY-dY):min(ndY+dY,labels_np.shape[1]),max(0,ndZ-dZ):min(ndZ+dZ,labels_np.shape[2])].max()==1:
        		boxes_new['box'].append(boxes['box'][i])
        		boxes_new['label'].append(boxes['label'][i])
        		boxes_new['label_score'].append(boxes['label_score'][i])
        		
        		if boxes['label_score'][i] >= 0.5:
            			boxes_thresh['box'].append(boxes['box'][i])
            			boxes_thresh['label'].append(boxes['label'][i])
            			boxes_thresh['label_score'].append(boxes['label_score'][i])
        	else:
        		boxes_removed['box'].append(boxes['box'][i])
        		boxes_removed['label'].append(boxes['label'][i])
        		boxes_removed['label_score'].append(boxes['label_score'][i])
        except:
        	boxes_error['box'].append(boxes['box'][i])
        	boxes_error['label'].append(boxes['label'][i])
        	boxes_error['label_score'].append(boxes['label_score'][i])
        
    print("Removed %i out of %i nodules" %(len(boxes_removed['box']) + len(boxes_error['box']),len(nodules_pos)))
    print("of which %i errors" %(len(boxes_error['box'])))
    data["box"] = boxes_new['box']
    data["boxes_removed"] = boxes_removed['box']
    data["boxes_error"] = boxes_error['box']
    data['label'] = boxes_new['label']
    data['boxes_removed_label'] = boxes_removed['label']
    data['boxes_error_label'] = boxes_error['label']
    data['label_score'] = boxes_new['label_score']
    data['boxes_removed_label_score'] = boxes_removed['label_score']
    data['boxes_error_label_score'] = boxes_error['label_score']
    data_thresh['box'] = boxes_thresh['box']
    data_thresh['label'] = boxes_thresh['label']
    data_thresh['label_score'] = boxes_thresh['label_score']
    
    os.system("mv %s/test_labels/labels_det_nd.json %s/test_labels/labels_det_nd_old.json" %(ID, ID))
    
    with open("%s/test_labels/labels_det_nd.json" %(ID), "w", encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    with open("%s/test_labels/labels_det_nd_thresh.json" %(ID), "w", encoding='utf-8') as f:
        json.dump(data_thresh, f, indent=4)
        
        
        
else:
    print("No nodules detected")
