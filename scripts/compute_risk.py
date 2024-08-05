import sys, os
import pandas as pd
import numpy as np


def risk_function(results_matching, lam):
	nod = results_matching[results_matching['box_label']==1]	
	nonod = results_matching[results_matching['box_label']==0]
	
	nr_nod = len(nod)
	nr_nonod = len(nonod)
	tp = sum(nod['box_conf']>=lam)
	fp = sum(nonod['box_conf']>=lam)
	tn = nr_nonod - fp
	fn = nr_nod - tp
	
	efficiency = tp + fp
	
	if nr_nod > 0:
		sensitivity = tp/nr_nod
	else:
		print('Warning - No nods matched: ', results_matching['img_id'])
		sensitivity = 1
	
	if efficiency>0:
		precision = tp/efficiency
	else:
		precision = 1
	
	return sensitivity, precision, efficiency, tp, fp, tn, fn


def gather_risk_results(results_matching, lambdas, save_to_path):
	
	output_file = save_to_path + 'results_risk.csv'
	
	results_risk = pd.DataFrame(columns=['img_id', 'lambda', 'sensitivity', 'precision', 'efficiency', 'true positives', 'false positives', 'true negatives', 'false negatives'])
	uids = results_matching['img_id'].unique()
	
	i=0
	print('Nr imgs: ', len(uids))
	
	for j in range(len(uids)):
		print('Processing img: ', j)
		print(uids[j])
		uid_results_matching = results_matching[results_matching['img_id']==uids[j]]
		
		for lam in lambdas:
			uid_sensitivity, uid_precision, uid_efficiency, uid_tp, uid_fp, uid_tn, uid_fn = risk_function(uid_results_matching, lam)
			uid_results_risk = [uids[j]] + [lam] + [uid_sensitivity] + [uid_precision] + [uid_efficiency] + [uid_tp] + [uid_fp] + [uid_tn] + [uid_fn]
			results_risk.loc[i] = uid_results_risk
			i += 1
			
	results_risk.to_csv(output_file, index=False)
	
	return results_risk
	

if __name__ == '__main__':

	consensus = [1, 2, 3, 4]
	lambdas = np.linspace(0, 1, 1000)
	use_seg = True
	
	for c in consensus:
		
		if use_seg:
			path_matching = 'results/set_{}/results_matching.csv'.format(c)
		else:
			path_matching = 'results/set_{}/results_matching_no_seg.csv'.format(c)
	
		print('Reading results...')
		results_matching = pd.read_csv(path_matching, index_col=False)
		print(results_matching.head(25))
	
		results_risk = gather_risk_results(results_matching, lambdas, 'results/set_{}/'.format(c))
		
		
