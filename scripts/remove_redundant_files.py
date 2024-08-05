import sys, os

path = sys.argv[1]
img_id = os.path.basename(os.path.normpath(path))
	
print('Removing redundant files: ', img_id)
try:
	os.remove(path + '/test_labels/' + img_id + '_seg_ct.nii')
except FileNotFoundError:
	print('No _seg_ct file to remove')
	
try:
	os.remove(path + '/test_labels/' + img_id + '_det_nd.json')
except FileNotFoundError:
	print('No _det_nd file to remove')
	
try:
	os.remove(path + '/test_labels/' + img_id + '_det_nd_old.json')
except FileNotFoundError:
	print('No _det_nd_old file to remove')
