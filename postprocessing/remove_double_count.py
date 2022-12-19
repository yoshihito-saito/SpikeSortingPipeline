import numpy as np


def remove_double_counted_spikes(spike_times, spike_clusters, spike_templates, amplitudes, channel_map, templates, pc_features, pc_feature_ind, template_features, sample_rate, params, epochs = None):

    unit_list = np.arange(np.max(spike_clusters)+1)
    peak_channels = np.squeeze(channel_map[np.argmax(np.max(templates,1) - np.min(templates,1),1)])
    order = np.argsort(peak_channels)
    overlap_matrix = np.zeros((peak_channels.size, peak_channels.size))

    within_unit_overlap_samples = int(params['within_unit_overlap_window'] * sample_rate)
    between_unit_overlap_samples = int(params['between_unit_overlap_window'] * sample_rate)

    print('Removing within-unit overlapping spikes...')

    spikes_to_remove = np.zeros((0,), dtype='i8')

    for idx1, unit_id1 in enumerate(unit_list[order]):
        for_unit1 = np.where(spike_clusters == unit_id1)[0]
        to_remove = find_within_unit_overlap(spike_times[for_unit1], within_unit_overlap_samples)
        overlap_matrix[idx1, idx1] = len(to_remove)
        spikes_to_remove = np.concatenate((spikes_to_remove, for_unit1[to_remove]))
    print(len(spikes_to_remove),"spikes removed")
    spike_times, spike_clusters, spike_templates, amplitudes, pc_features, template_features = remove_spikes(spike_times, 
                                                                        spike_clusters, 
                                                                        spike_templates, 
                                                                        amplitudes, 
                                                                        pc_features, 
                                                                        template_features,
                                                                        spikes_to_remove)

    print('Removing between-unit overlapping spikes...')

    spikes_to_remove = np.zeros((0,), dtype='i8')

    for idx1, unit_id1 in enumerate(unit_list[order]):
        for_unit1 = np.where(spike_clusters == unit_id1)[0]
        for idx2, unit_id2 in enumerate(unit_list[order]):
            if idx2 > idx1 and np.abs(peak_channels[unit_id1] - peak_channels[unit_id2]) < params['between_unit_channel_distance']:
                for_unit2 = np.where(spike_clusters == unit_id2)[0]
                to_remove1, to_remove2 = find_between_unit_overlap(spike_times[for_unit1], spike_times[for_unit2], between_unit_overlap_samples)
                overlap_matrix[idx1, idx2] = len(to_remove1) + len(to_remove2)
                spikes_to_remove = np.concatenate((spikes_to_remove, for_unit1[to_remove1], for_unit2[to_remove2]))
    print(len(spikes_to_remove), "spikes_removed")
    spike_times, spike_clusters, spike_templates, amplitudes, pc_features, template_features = remove_spikes(spike_times, 
                                                                         spike_clusters,
                                                                         spike_templates, 
                                                                         amplitudes, 
                                                                         pc_features, 
                                                                         template_features,
                                                                         np.unique(spikes_to_remove))

    return spike_times, spike_clusters, spike_templates, amplitudes, pc_features, template_features, overlap_matrix

                
def find_within_unit_overlap(spike_train, overlap_window = 5):

    spikes_to_remove = np.where(np.diff(spike_train) < overlap_window)[0]

    return spikes_to_remove


def find_between_unit_overlap(spike_train1, spike_train2, overlap_window = 5):

    spike_train = np.concatenate( (spike_train1, spike_train2) )
    original_inds = np.concatenate( (np.arange(len(spike_train1)), np.arange(len(spike_train2)) ) )
    cluster_ids = np.concatenate( (np.zeros((len(spike_train1),)), np.ones((len(spike_train2),))) )

    order = np.argsort(spike_train)
    sorted_train = spike_train[order]
    sorted_inds = original_inds[order][1:]
    sorted_cluster_ids = cluster_ids[order][1:]

    spikes_to_remove = np.diff(sorted_train) < overlap_window

    spikes_to_remove1 = sorted_inds[spikes_to_remove * (sorted_cluster_ids == 0)]
    spikes_to_remove2 = sorted_inds[spikes_to_remove * (sorted_cluster_ids == 1)]

    return spikes_to_remove1, spikes_to_remove2


def remove_spikes(spike_times, spike_clusters, spike_templates, amplitudes, pc_features, template_features, spikes_to_remove):

    spike_times = np.delete(spike_times, spikes_to_remove, 0)
    spike_clusters = np.delete(spike_clusters, spikes_to_remove, 0)
    spike_templates = np.delete(spike_templates, spikes_to_remove, 0)
    amplitudes = np.delete(amplitudes, spikes_to_remove, 0)
    if pc_features is not None:
        pc_features = np.delete(pc_features, spikes_to_remove, 0)
        template_features = np.delete(template_features, spikes_to_remove, 0)

    return spike_times, spike_clusters, spike_templates, amplitudes, pc_features, template_features


