import numpy as np
import os
import pandas as pd

def read_cluster_group_tsv(filename):
    info = pd.read_csv(filename, sep='\t')
    cluster_ids = info['cluster_id'].values.astype('int')
    
    try:
        cluster_quality = info['group'].values
    except KeyError:
        cluster_quality = info['KSLabel'].values

    return cluster_ids, cluster_quality


def load(folder, filename):
    return np.load(os.path.join(folder, filename))

def load_kilosort_data(folder, 
                       sample_rate = None, 
                       convert_to_seconds = False, 
                       use_master_clock = False, 
                       include_pcs = True,
                       template_zero_padding= 21):

    if use_master_clock:
        spike_times = load(folder,'spike_times_master_clock.npy')
    else:
        spike_times = load(folder,'spike_times.npy')
        
    spike_clusters = load(folder,'spike_clusters.npy')
    spike_templates = load(folder, 'spike_templates.npy')
    amplitudes = load(folder,'amplitudes.npy')
    templates = load(folder,'templates.npy')
    unwhitening_mat = load(folder,'whitening_mat_inv.npy')
    channel_map = np.squeeze(load(folder, 'channel_map.npy'))

    if include_pcs:
        pc_features = load(folder, 'pc_features.npy')
        pc_feature_ind = load(folder, 'pc_feature_ind.npy') 
        template_features = load(folder, 'template_features.npy') 

    templates = templates[:,template_zero_padding:,:] # remove zeros
    spike_clusters = np.squeeze(spike_clusters) # fix dimensions
    spike_templates = np.squeeze(spike_templates) # fix dimensions
    spike_times = np.squeeze(spike_times)# fix dimensions

    if convert_to_seconds and sample_rate is not None:
       spike_times = spike_times / sample_rate 
                    
    unwhitened_temps = np.zeros((templates.shape))
    
    for temp_idx in range(templates.shape[0]):
        unwhitened_temps[temp_idx,:,:] = np.dot(np.ascontiguousarray(templates[temp_idx,:,:]),np.ascontiguousarray(unwhitening_mat))
                    
    try:
        cluster_ids, cluster_quality = read_cluster_group_tsv(os.path.join(folder, 'cluster_group.tsv'))
    except OSError:
        cluster_ids = np.unique(spike_clusters)
        cluster_quality = ['unsorted'] * cluster_ids.size

    if not include_pcs:
        return spike_times, spike_clusters, spike_templates, amplitudes, unwhitened_temps, channel_map   
    else:
        return spike_times, spike_clusters, spike_templates, amplitudes, unwhitened_temps, channel_map, pc_features, pc_feature_ind, template_features

