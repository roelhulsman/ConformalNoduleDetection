import sys, os
import shutil

path = sys.argv[1]
print('Removing model output in: ', path)
	
for img_id in os.listdir(path):
	print('Removing model output in: ', path + img_id)
	if os.path.isdir(path + img_id + '/test_labels/'):
		shutil.rmtree(path + img_id + '/test_labels/')
		print('Removed /test_labels/: ', img_id)
	if os.path.isdir(path + img_id + '/labels/'):
		shutil.rmtree(path + img_id + '/labels/')
		print('Removed /labels/: ', img_id)
	if os.path.exists(path + img_id + '/.lock'):
		os.remove(path + img_id + '/.lock')
		print('Removed /.lock: ', img_id)
	if os.path.exists(path + img_id + '/datastore_v2.json'):
		os.remove(path + img_id + '/datastore_v2.json')
		print('Removed /datastore_v2.json: ', img_id)
	
	
