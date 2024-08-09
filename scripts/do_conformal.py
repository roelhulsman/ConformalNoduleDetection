import sys, os
import pandas as pd
import numpy as np
from scipy.optimize import brentq
from scipy.stats import binom
import seaborn as sns
import matplotlib.pyplot as plt
import random
import json


def split_data(risk_results, seed=42):
	
	n = len(risk_results)
	imgs = risk_results['img_id'].unique()
	n_imgs = len(imgs)
	
	# Split the data into calibration and validation sets (save the shuffling)
	idx = np.array([1] * int(n_imgs/2) + [0] * (n_imgs - int(n_imgs/2))) > 0
	random.Random(seed).shuffle(idx)
	cal_imgs, val_imgs = imgs[idx], imgs[~idx]
	
	cal_data, val_data = risk_results[risk_results['img_id'].isin(cal_imgs)], risk_results[risk_results['img_id'].isin(val_imgs)]
	
	#print('Cal imgs: ', len(cal_data['img_id'].unique()))
	#print('Val imgs: ', len(val_data['img_id'].unique()))
	
	return cal_data, val_data
	


def plot_single_figures_seed(strategies, metrics, results_risk, results_strat, seed, save_to_path):
	
	cal_data, val_data = split_data(results_risk, seed=seed)
	n = len(cal_data['img_id'].unique())
	data = {}
	data['calibration'] = cal_data.iloc[:,1:].groupby(['lambda']).mean().sort_values(by=['lambda'], ascending=True)
	data['validation'] = val_data.iloc[:,1:].groupby(['lambda']).mean().sort_values(by=['lambda'], ascending=True)
	
	sns.set_context("talk")
	plt.figure(figsize=(15,10))
	
	for ds_name in ['calibration', 'validation']:
		for metric in metrics:
			plot = sns.scatterplot(data[ds_name], x='lambda', y=metric)
			plot.set(xlabel='Lambda', ylabel=metric.title(), title=metric.title() + ' vs Lambda - ' + ds_name.title())
			for strategy in strategies.keys():
				plot.axhline(y = results_strat[strategy][ds_name][metric], color=strategies[strategy]['color'], label=strategy.title())
				plot.axvline(x = results_strat[strategy]['lhat'], color=strategies[strategy]['color'])
			if metric == 'efficiency' or metric == 'false positives':
				plt.yscale("log")
			plt.legend()
			fig = plot.get_figure()
			fig.savefig(save_to_path + 'plot_%s_%s_lambda_seed_%s.png' %(ds_name, metric, seed))
			print('Figure - %s - %s vs Lambda - Saved' %(ds_name.title(), metric.title()))
			plt.clf()
		
		plot = sns.scatterplot(data[ds_name], x='false positives', y='sensitivity')
		plot.set(xlabel='False Positives', ylabel='Sensitivity', title='FROC (Adapted) - ' + ds_name.title())
		plt.xlim([0, 15])
		plt.ylim([0.7, 1])
		for strategy in strategies.keys():
			plot.axhline(y = results_strat[strategy][ds_name]['sensitivity'], color=strategies[strategy]['color'], label=strategy.title())
			plot.axvline(x = results_strat[strategy][ds_name]['false positives'], color=strategies[strategy]['color'])
		plt.legend()
		fig = plot.get_figure()
		fig.savefig(save_to_path + 'plot_%s_sensitivity_fp_seed_%s.png' %(ds_name, seed))
		print('Figure - %s - Sensitivity vs False Positives - Saved' %(ds_name.title()))
		plt.clf()
	
		plot = sns.scatterplot(data[ds_name], x='precision', y='sensitivity')
		plot.set(xlabel='Precision', ylabel='Sensitivity', title='PRC - ' + ds_name.title())
		for strategy in strategies.keys():
			plot.axhline(y = results_strat[strategy][ds_name]['sensitivity'], color=strategies[strategy]['color'], label=strategy.title())
			plot.axvline(x = results_strat[strategy][ds_name]['precision'], color=strategies[strategy]['color'])
		plt.legend()
		fig = plot.get_figure()
		fig.savefig(save_to_path + 'plot_%s_sensitivity_precision_seed_%s.png' %(ds_name, seed))
		print('Figure - %s - Sensitivity vs Precision - Saved' %(ds_name.title()))
		plt.clf()
	
	
	
def plot_single_figures(strategies, metrics, results_strat, save_to_path, R=10000):
	
	sns.set_context("talk")
	plt.figure(figsize=(15,10))
	
	for strategy in strategies.keys():
		if strategy != 'Naive':
			plot = sns.histplot(pd.DataFrame(results_strat[strategy]['lhat'], columns=['lhat']), x='lhat', stat='probability', alpha=0.5, color=strategies[strategy]['color'], label=strategy.title())
	plot.set(xlabel='lhat', ylabel='Density', title='Density of Confidence Threshold - R=%s Splits' %(R))
	plt.legend()
	fig = plot.get_figure()
	if 'Risk-Controlling Prediction Sets' in strategies.keys():
		fig.savefig(save_to_path + 'plot_lhat_density_rcps.png')
	elif 'FROC Analysis' in strategies.keys():
		fig.savefig(save_to_path + 'plot_lhat_density_froc.png')
	else:
		fig.savefig(save_to_path + 'plot_lhat_density.png')
	print('Figure - lhat Density - Saved')
	plt.clf()
			
	for metric in metrics:
		for strategy in strategies.keys():
			plot = sns.histplot(pd.DataFrame(results_strat[strategy]['validation']), x=metric, stat='probability', alpha=0.5, color=strategies[strategy]['color'], label=strategy.title())
		plot.set(xlabel=metric.title(), ylabel='Density', title='Density of ' + metric.title() + ' - R=%s Splits' %(R))
		if metric == 'sensitivity':
			plt.xlim([0.6, 1.0])
			plt.ylim([0, 0.15])
			if 'Conformal Risk Control' in strategies.keys():
				plot.axvline(x = 1 - strategies['Conformal Risk Control']['alpha'], color='red', lw=2, label='Target Sensitivity')
			if 'Risk-Controlling Prediction Sets' in strategies.keys():
				plot.axvline(x = 1 - strategies['Risk-Controlling Prediction Sets']['alpha'], color='red', lw=2)
			elif 'FROC Analysis' in strategies.keys():
				plot.axvline(x = 1 - strategies['FROC Analysis']['alpha'], color='red', lw=2)
		plt.legend()
		fig = plot.get_figure()
		if 'Risk-Controlling Prediction Sets' in strategies.keys():
			fig.savefig(save_to_path + 'plot_%s_density_rcps.png' %(metric))
		elif 'FROC Analysis' in strategies.keys():
			fig.savefig(save_to_path + 'plot_%s_density_froc.png' %(metric))
		else:
			fig.savefig(save_to_path + 'plot_%s_density.png' %(metric))
		print('Figure - %s Density - Saved' %(metric))
		plt.clf()


def plot_froc_adapted(results_froc, strategies, lambdas, seed, save_to_path):

	for ds_name in ['calibration', 'validation']:
		sns.set_context("talk")
		fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
	
		for i in range(1, 5):
			results_risk = pd.read_csv(save_to_path + 'set_{}/results_risk.csv'.format(i), index_col=False)
		
			cal_data, val_data = split_data(results_risk, seed=seed)
			n = len(set(cal_data['img_id']))
			data = {}
			data['calibration'] = cal_data.iloc[:,1:].groupby(['lambda']).mean().sort_values(by=['lambda'], ascending=True)
			data['validation'] = val_data.iloc[:,1:].groupby(['lambda']).mean().sort_values(by=['lambda'], ascending=True)
		
			if i == 1: 
				k=0
				l=0
				ax[k, l].set_xticks([])
			elif i ==2: 
				k=0
				l=1
				ax[k, l].set_xticks([])
				ax[k, l].set_yticks([])
			elif i ==3: 
				k=1
				l=0
			elif i ==4: 
				k=1
				l=1
				ax[k, l].set_yticks([])
			plot = sns.scatterplot(data[ds_name], x='false positives', y='sensitivity', ax=ax[k, l], color='k')
			plot.set(xlabel='False Positives per Scan', ylabel='Sensitivity per Scan', title='Set %s' %(i))
			ax[k, l].set_xlim([0, 15])
			ax[k, l].set_ylim([0.65, 1])
			for strategy in strategies.keys():
				lhat = results_froc['Set ' + str(i)][strategy]['lhat'][0]
				ax[k, l].scatter(x = data[ds_name].iloc[np.where(lambdas==lhat)[0][0],:]['false positives'], y = data[ds_name].iloc[np.where(lambdas==lhat)[0][0],:]['sensitivity'], color=strategies[strategy]['color'], marker='o', s=100)
				plot.axvline(x = data[ds_name].iloc[np.where(lambdas==lhat)[0][0],:]['false positives'], color=strategies[strategy]['color'], linestyle='dashed', lw=3, label=strategy.title())
				plot.axhline(y = data[ds_name].iloc[np.where(lambdas==lhat)[0][0],:]['sensitivity'], color=strategies[strategy]['color'], linestyle='dashed', lw=3)
			plt.legend(fontsize='medium')
			
		
		fig.tight_layout()
		
		if 'Risk-Controlling Prediction Sets' in strategies.keys():
			fig.savefig(save_to_path + 'plot_froc_adapted_{}_seed_{}_rcps.png'.format(ds_name, seed))
		elif 'FROC Analysis' in strategies.keys():
			fig.savefig(save_to_path + 'plot_froc_adapted_{}_seed_{}_froc.png'.format(ds_name, seed))
		else:
			fig.savefig(save_to_path + 'plot_froc_adapted_{}_seed_{}.png'.format(ds_name, seed))
		print('Figure - FROC (Adapted) - {} - Saved'.format(ds_name.title()))
		plt.clf()
		
		
def plot_consensus_analysis(results_consensus, strategies, save_to_path, R):
	
	sns.set_context("talk")
	nr_datasets = len(set(results_consensus['dataset']))
	metrics = {'sensitivity': [[0.65, 1.0], [0, 0.11]], 'precision': [[0, 1.0], [0, 0.11]], 'false negatives': [[0, 0.8], [0, 0.11]], 'false positives': [[0, 20], [0, 0.11]]}
	fig, ax = plt.subplots(nrows=nr_datasets, ncols=len(metrics), figsize=(20, 10))
	

	for i in range(ax.shape[0]):
		for j in range(0, ax.shape[1]):
			for strategy in strategies.keys():
				plot = sns.histplot(results_consensus[(results_consensus['dataset'] == 'Set {}'.format(i+1)) & (results_consensus['strategy'] == strategy)], x=list(metrics.keys())[j], stat='probability', alpha=0.4, color=strategies[strategy]['color'], label=strategy.title(), ax=ax[i][j])
			if i==0:
				plot.set(title='Density of ' + list(metrics.keys())[j].title())
			else:
				plot.set(title=None)
			if j==0:
				plot.set(ylabel='Set %s\nProbability' %(i+1))
			else: 
				plot.set(ylabel=None)
				ax[i][j].set_yticks([])
			if i==ax.shape[0]-1:
				plot.set(xlabel=list(metrics.keys())[j].title() + ' per Scan')
				if j==ax.shape[1]-1:
					ax[i][j].legend(loc='upper right', fancybox=True, fontsize='small')
			else:
				plot.set(xlabel=None)
				ax[i][j].set_xticks([])
			if list(metrics.keys())[j] == 'sensitivity':
				plot.axvline(x = 1 - strategies['Conformal Risk Control']['alpha'], color='red', lw=3, linestyle='dashed', label='Target Sensitivity')
			ax[i][j].set_xlim(metrics[list(metrics.keys())[j]][0])
			ax[i][j].set_ylim(metrics[list(metrics.keys())[j]][1])
			 

	#plt.legend(loc='upper right', fancybox=True, fontsize=8)
	 
	plt.tight_layout()
	
	
	if 'Risk-Controlling Prediction Sets' in strategies.keys():
		fig.savefig(save_to_path + 'plot_consensus_analysis_rcps.png')
	elif 'FROC Analysis' in strategies.keys():
		fig.savefig(save_to_path + 'plot_consensus_analysis_froc.png')
	else:
		fig.savefig(save_to_path + 'plot_consensus_analysis.png')
	print('Figure - Saved')
	plt.clf()
	


# Function copied from Risk-Controlling Prediction Sets paper by S. Bates et al. 
def WSR_mu_plus(x, delta, maxiters): # this one is different.
    n = x.shape[0]
    muhat = (np.cumsum(x) + 0.5) / (1 + np.array(range(1,n+1)))
    sigma2hat = (np.cumsum((x - muhat)**2) + 0.25) / (1 + np.array(range(1,n+1))) 
    sigma2hat[1:] = sigma2hat[:-1]
    sigma2hat[0] = 0.25
    nu = np.minimum(np.sqrt(2 * np.log( 1 / delta ) / n / sigma2hat), 1)
    def _Kn(mu):
        return np.max(np.cumsum(np.log(1 - nu * (x - mu)))) + np.log(delta)
    if _Kn(1) < 0:
        return 1
    return brentq(_Kn, 1e-10, 1-1e-10, maxiter=maxiters)



# Function adapted from Conformal Risk Control paper by A. N. Angelopoulos et al. 
"""
    Gets the value of lambda hat that controls the marginal risk for a monotone risk function.
    The calib loss table should be ordered from small loss to large loss
"""
def get_lhat(strategy, data, n, thresh, lambdas, B=1):	
	
	if strategy == 'Naive':
		lhat_idx = np.argmax(lambdas >= 0.5)
	elif strategy == 'Empirical':
		rhat = 1 - data['sensitivity']
		lhat_idx = max(np.argmax(rhat >= thresh) - 1, 0) # Can't be -1.
	elif strategy == 'Conformal Risk Control':
		rhat = 1 - data['sensitivity']
		lhat_idx = max(np.argmax(((n/(n+1)) * rhat + B/(n+1) ) >= thresh) - 1, 0) # Can't be -1.
	elif strategy == 'Risk-Controlling Prediction Sets':
		rhat = data['wsr_bound']
		lhat_idx = max(np.argmax(rhat >= thresh) - 1, 0) # Can't be -1.
	elif strategy == 'FROC Analysis':
		rhat = 1 - data['froc_sensitivity']
		lhat_idx = max(np.argmax(rhat >= thresh) - 1, 0) # Can't be -1.
	
	return lambdas[lhat_idx]
	


def initiate_strat_dict(strategies, metrics):
	
	results_strat = {}
	for strategy in strategies.keys():
		results_strat[strategy] = {}
		results_strat[strategy]['seed'] = []
		results_strat[strategy]['lhat'] = []
		results_strat[strategy]['calibration'] = {}
		results_strat[strategy]['validation'] = {}
		for metric in metrics:
			results_strat[strategy]['calibration'][metric] = []
			results_strat[strategy]['validation'][metric] = []
	
	return results_strat
	
	
def run_strategies_seed(results_strat, strategies, metrics, results_risk, lambdas, seed):
	print('Seed: ', seed)
		
	cal_data, val_data = split_data(results_risk, seed=seed)
	n = len(cal_data['img_id'].unique())
	if 'Risk-Controlling Prediction Sets' in strategies.keys():
		wsr_bound_cal = []
		for i in range(len(lambdas)):
			loss = 1 - cal_data[cal_data['lambda'].round(5) == lambdas[i].round(5)]['sensitivity'].to_numpy()
			wsr_bound_cal.append(WSR_mu_plus(loss, strategies['Risk-Controlling Prediction Sets']['delta'], 1000))
	if 'FROC Analysis' in strategies.keys():
		froc_cal = []
		for i in range(len(lambdas)):
			tp_sum = cal_data[cal_data['lambda'].round(5) == lambdas[i].round(5)]['true positives'].sum()
			fn_sum = cal_data[cal_data['lambda'].round(5) == lambdas[i].round(5)]['false negatives'].sum()
			froc_cal.append(tp_sum/(tp_sum + fn_sum))
		
	cal_data = cal_data.iloc[:,1:].groupby(['lambda']).mean().sort_values(by=['lambda'], ascending=True)
	if 'Risk-Controlling Prediction Sets' in strategies.keys():
		cal_data['wsr_bound'] = wsr_bound_cal
	if 'FROC Analysis' in strategies.keys():
		cal_data['froc_sensitivity'] = froc_cal
	val_data = val_data.iloc[:,1:].groupby(['lambda']).mean().sort_values(by=['lambda'], ascending=True)

	for strategy in strategies.keys():
		results_strat[strategy]['seed'].append(seed)
		lhat = get_lhat(strategy, cal_data, n, strategies[strategy]['alpha'], lambdas, B=1)
		results_strat[strategy]['lhat'].append(lhat)
		for metric in metrics:
			results_strat[strategy]['calibration'][metric].append(cal_data[metric].iloc[np.argmax(lambdas==lhat)])
			results_strat[strategy]['validation'][metric].append(val_data[metric].iloc[np.argmax(lambdas==lhat)])
			
	return results_strat
				

def run_strategies(strategies, metrics, results_risk, lambdas, save_to_path, R=10000, load_old_file=False, seed=None):
	
	results_strat = initiate_strat_dict(strategies, metrics)
	
	if seed is not None:
		results_strat = run_strategies_seed(results_strat, strategies, metrics, results_risk, lambdas, seed)
		plot_single_figures_seed(strategies, metrics, results_risk, results_strat, seed, save_to_path + 'figures/')
			
		return results_strat
	else:
	
		if not load_old_file:
			for s in range(0, R):
				results_strat = run_strategies_seed(results_strat, strategies, metrics, results_risk, lambdas, s)
	
			if 'Risk-Controlling Prediction Sets' in strategies.keys():
				with open(save_to_path + 'results_strategies_{}_seeds_rcps.json'.format(R), "w", encoding='utf-8') as f:
        				json.dump(results_strat, f, indent=4)
			elif 'FROC Analysis' in strategies.keys():
        			with open(save_to_path + 'results_strategies_{}_seeds_froc.json'.format(R), "w", encoding='utf-8') as f:
        				json.dump(results_strat, f, indent=4)
			else:
				with open(save_to_path + 'results_strategies_{}_seeds.json'.format(R), "w", encoding='utf-8') as f:
        				json.dump(results_strat, f, indent=4)
		else:
			if 'Risk-Controlling Prediction Sets' in strategies.keys():
				with open(save_to_path + 'results_strategies_{}_seeds_rcps.json'.format(R)) as f:
        				results_strat = json.load(f)
			elif 'FROC Analysis' in strategies.keys():
        			with open(save_to_path + 'results_strategies_{}_seeds_froc.json'.format(R)) as f:
        				results_strat = json.load(f)
			else:
        			with open(save_to_path + 'results_strategies_{}_seeds.json'.format(R)) as f:
        				results_strat = json.load(f)
	
		plot_single_figures(strategies, metrics, results_strat, save_to_path + 'figures/', R)
	
		return results_strat	


def run_froc_adapted_analysis(strategies, metrics, lambdas, save_to_path, seed=7, load_old_file=False):
	
	if not load_old_file:
		results_froc = {}
		for i in range(1, 5):
			results_risk = pd.read_csv(save_to_path + 'set_{}/results_risk.csv'.format(i), index_col=False)
			
			results_froc['Set ' + str(i)] = run_strategies(strategies, metrics, results_risk, lambdas, save_to_path + 'set_{}/'.format(i), seed=seed)
		
		if 'Risk-Controlling Prediction Sets' in strategies.keys():
			with open(save_to_path + 'results_froc_adapted_analysis_seed_{}_rcps.json'.format(seed), "w", encoding='utf-8') as f:
        				json.dump(results_froc, f, indent=4)
		elif 'FROC Analysis' in strategies.keys():
        		with open(save_to_path + 'results_froc_adapted_analysis_seed_{}_froc.json'.format(seed), "w", encoding='utf-8') as f:
        				json.dump(results_froc, f, indent=4)
		else:
			with open(save_to_path + 'results_froc_adapted_analysis_seed_{}.json'.format(seed), "w", encoding='utf-8') as f:
        				json.dump(results_froc, f, indent=4)
			
	else:
		if 'Risk-Controlling Prediction Sets' in strategies.keys():
			with open(save_to_path + 'results_froc_adapted_analysis_seed_{}_rcps.json'.format(seed)) as f:
        			results_froc = json.load(f)
		elif 'FROC Analysis' in strategies.keys():
        		with open(save_to_path + 'results_froc_adapted_analysis_seed_{}_froc.json'.format(seed)) as f:
        			results_froc = json.load(f)
		else:
			with open(save_to_path + 'results_froc_adapted_analysis_seed_{}.json'.format(seed)) as f:
        			results_froc = json.load(f)
	
	plot_froc_adapted(results_froc, strategies, lambdas, seed, save_to_path)
	
	return results_froc



	
def run_consensus_analysis(strategies, metrics, lambdas, save_to_path, R=10000, load_old_file=False):

	if not load_old_file: 
		results_consensus = pd.DataFrame(columns=['dataset', 'strategy', 'seed', 'sensitivity', 'precision', 'efficiency', 'false positives', 'false negatives'])
	
		for i in range(1, 5):
			results_risk = pd.read_csv(save_to_path + 'set_{}/results_risk.csv'.format(i), index_col=False)
	
			results_strat = run_strategies(strategies, metrics, results_risk, lambdas, save_to_path + 'set_{}/'.format(i), R, load_old_file)
		
			for strategy in strategies.keys():
				results_set = pd.DataFrame({'dataset': 'Set {}'.format(i), 'strategy': strategy, 'seed': results_strat[strategy]['seed'], 'sensitivity': results_strat[strategy]['validation']['sensitivity'], 'precision': results_strat[strategy]['validation']['precision'], 'efficiency': results_strat[strategy]['validation']['efficiency'], 'false positives': results_strat[strategy]['validation']['false positives'], 'false negatives': results_strat[strategy]['validation']['false negatives']})
				results_consensus = pd.concat([results_consensus, results_set], ignore_index=True)
		
		if 'Risk-Controlling Prediction Sets' in strategies.keys():
			results_consensus.to_csv(save_to_path + 'results_consensus_analysis_rcps.csv', index=False)
		elif 'FROC Analysis' in strategies.keys():
			results_consensus.to_csv(save_to_path + 'results_consensus_analysis_froc.csv', index=False)
		else:
			results_consensus.to_csv(save_to_path + 'results_consensus_analysis.csv', index=False)
	
	else:
		if 'Risk-Controlling Prediction Sets' in strategies.keys():
			results_consensus = pd.read_csv(save_to_path + 'results_consensus_analysis_rcps.csv', index_col=False)
		elif 'FROC Analysis' in strategies.keys():
			results_consensus = pd.read_csv(save_to_path + 'results_consensus_analysis_froc.csv', index_col=False)
		else:
			results_consensus = pd.read_csv(save_to_path + 'results_consensus_analysis.csv', index_col=False)
	
	plot_consensus_analysis(results_consensus, strategies, save_to_path, R)

	print(results_consensus.drop('seed', axis=1).groupby(['dataset', 'strategy']).mean())
	
	return results_consensus
	


		
		
	
if __name__ == '__main__':
	
	save_to_path = 'results/'
	seed = 3
	lambdas = np.linspace(0, 1, 1000)
	metrics = ['sensitivity', 'precision', 'efficiency', 'true positives', 'false positives', 'true negatives', 'false negatives']
	
	
	# To replicate the main figures, run the following lines. 
	R = 10000
	strategies = {'Naive': {'color': 'brown', 'alpha': None}, 'Conformal Risk Control': {'color': 'green', 'alpha': 0.1}}
	run_froc_adapted_analysis(strategies, metrics, lambdas, save_to_path, seed=seed, load_old_file=True)
	run_consensus_analysis(strategies, metrics, lambdas, save_to_path, R, load_old_file=True)
	
	
	# To replicate the figures comparing CRC to FROC, run the following lines.
	R = 1000
	strategies = {'Naive': {'color': 'brown', 'alpha': None}, 'FROC Analysis': {'color': 'navy', 'alpha': 0.1}, 'Conformal Risk Control': {'color': 'green', 'alpha': 0.1}}
	run_froc_adapted_analysis(strategies, metrics, lambdas, save_to_path, seed=seed, load_old_file=True)
	run_consensus_analysis(strategies, metrics, lambdas, save_to_path, R, load_old_file=True)
	
	
	"""
	# To compare CRC/FROC to RCPS, run the following lines. Note that RPCS targets a guarantee in high probability instead of expectation, such that a fair comparison is difficult. 
	#strategies = {'Naive': {'color': 'brown', 'alpha': None}, 'FROC Analysis': {'color': 'navy', 'alpha': 0.1}, 'Conformal Risk Control': {'color': 'green', 'alpha': 0.1}, 'Risk-Controlling Prediction Sets': {'color': 'red', 'alpha': 0.2, 'delta': 0.2}}
	run_froc_adapted_analysis(strategies, metrics, lambdas, save_to_path, seed=seed, load_old_file=True)
	run_consensus_analysis(strategies, metrics, lambdas, save_to_path, R, load_old_file=True)
	"""
	
	

	

	
	
