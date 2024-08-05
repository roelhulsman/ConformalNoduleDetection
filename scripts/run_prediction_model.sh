#!/bin/bash

set -e

echo $SHELL

c_det=0
c_seg=0
c_fp=0
skip_det=0
skip_seg=0
skip_fp=0

for dir in "$1"/*
do
        input_image="$dir"/
        echo "Processing $input_image"
        
        set -x
        
        
        labels_det=$dir"/test_labels/labels_det_nd.json"

	# Run nodule detection
	if [[ ! -e "$labels_det" ]]; then
		echo -e "\tDetecting nodules..."
		let $((c_det=c_det+1))

		python scripts/do_infer.py -t infer --studies "$dir" --det_nd

	else
		echo -e "\tNodule detection already done: $labels_det"
		let $((skip_det=skip_det+1))
	fi
	
	
        labels_seg=$dir"/test_labels/labels_seg_ct.json"
        
        # Run body segmentation
	if [[ ! -e "$labels_seg" ]]; then
    		echo -e "\tSegmenting..."
    		let $((c_seg=c_seg+1))
    		
    		python scripts/do_infer.py -t infer --studies "$dir" --seg_ct
    		
	else
    		echo -e "\tSegmentation already done! $labels_seg"
    		let $((skip_seg=skip_seg+1))
	fi


        labels_removed=$dir"/test_labels/labels_det_nd_old.json"

	# Remove nodules outside lung
	if [[ ! -e "$labels_removed" ]]; then
		echo -e "\tRemoving false positives..."
		let $((c_fp=c_fp+1))

		python scripts/remove_false_nodules.py "$dir"
    	
	else
		echo -e "\tFalse positives already removed: $labels_removed"
		let $((skip_fp=skip_fp+1))
	fi


	#python remove_redundant_files.py "$dir"
        
        set +x

done

echo "Processed $c_det image detections"
echo "Processed $c_seg image segmentations"
echo "Processed $c_fp image false positives removed"
echo "Skipped $skip_det image detections"
echo "Skipped $skip_seg image segmentations"
echo "Skipped $skip_fp image false positives removed"


