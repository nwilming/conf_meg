'''
This module contains definitions of clusters of areas that
are of interest for the analysis.

This module also contains scripts to generate an image for
clusters of labels.
'''

import os
if 'DISPLAY' in list(os.environ.keys()):
    try:
        from surfer import Brain
    except:
        Brain = None
        print('No pysurfer support')

import numpy as np
import pandas as pd

from conf_analysis.behavior import metadata
from joblib import Memory

memory = Memory(cachedir=metadata.cachedir)

visual_field_clusters = {
    'vfcvisual': ('wang2015atlas.V1d', 'wang2015atlas.V1v',
                  'wang2015atlas.V2d', 'wang2015atlas.V2v',
                  'wang2015atlas.V3d', 'wang2015atlas.V3v',
                  'wang2015atlas.hV4'),
    'vfcVO': ('wang2015atlas.VO1', 'wang2015atlas.VO2',),
    'vfcPHC': ('wang2015atlas.PHC1', 'wang2015atlas.PHC2'),
    'vfcV3ab': ('wang2015atlas.V3A', 'wang2015atlas.V3B'),
    'vfcTO': ('wang2015atlas.TO1', 'wang2015atlas.TO2'),
    'vfcLO': ('wang2015atlas.LO1', 'wang2015atlas.LO2'),
    'vfcIPS_occ': ('wang2015atlas.IPS0', 'wang2015atlas.IPS1'),
    'vfcIPS_dorsal': ('wang2015atlas.IPS2', 'wang2015atlas.IPS3',
                      'wang2015atlas.IPS4', 'wang2015atlas.IPS5'),
    'vfcSPL': ('wang2015atlas.SPL1',),
    'vfcFEF': ('wang2015atlas.FEF')
}


jwrois = {'IPS_Pces': ('JWDG.lr_IPS_PCes',),
          'M1': ('JWDG.lr_M1',),
          'aIPS1': ('JWDG.lr_aIPS1',)}


frontal = {'ACC': ('G&S_cingul-Ant-lh', 'G&S_cingul-Mid-Ant'),
           'frontomargin': ('G&S_frontomargin-',),
           'frontopol': ('G&S_transv_frontopol'),
           'f_inf_opercular': ('G_front_inf-Opercular',),
           'f_inf_orbital': ('G_front_inf-Orbital',),
           'f_inf_Triangul': ('G_front_inf-Triangul',),
           'Gf_middle': ('G_front_middle',),
           'Gf_sup': ('G_front_sup',),
           'Sf_inf': ('S_front_inf',),
           'Sf_middle': ('S_front_middle',),
           'Sf_sup': ('S_front_sup',)}


rows = {'visual': visual_field_clusters,
        'choice': jwrois,
        'frontal': frontal}

all_clusters = {}
all_clusters.update(visual_field_clusters)
all_clusters.update(jwrois)
all_clusters.update(frontal)

layouts = {'visual': (2, 5), 'choice': {1, 3}, 'frontal': (3, 4)}


def rh(columns):
    return [x for x in columns if (x.startswith('rh') | x.endswith('rh'))]


def lh(columns):
    return [x for x in columns if (x.startswith('lh') | x.endswith('lh'))]


def filter_cols(columns, select):
    return [x for x in columns if any([y.lower() in x.lower() for y in select])]


def reduce(df, all_clusters=all_clusters):
    '''
    Reduce ROIs to visual field clusters

    df is a DataFrame that has areas as columns. This function
    will iterate through the all_clusters dictionary that defines
    clusters of labels that define a ROI.

    The resulting data frame averaged over labels within a ROI cluster.
    '''
    columns = df.columns.values
    clusters = []
    for hemi, hcolumns in zip(['-lh', '-rh'], [lh(columns), rh(columns)]):
        for name, cols in list(all_clusters.items()):
            cols = filter_cols(hcolumns, cols)
            # if name == "M1":
            #  import pdb; pdb.set_trace()
            cluster = df.loc[:, cols].mean(1)
            cluster.name = name + hemi
            clusters.append(cluster)
    clusters = pd.concat(clusters, 1)
    return clusters


def lateralize(data, ipsi, contra, suffix='_Lateralized'):
    '''
    Lateralize set of rois
    '''
    out = []
    for i, c in zip(ipsi, contra):
        out.append(data.loc[:, c] - data.loc[:, i])
        out[-1].name = i.replace('rh', 'lh').replace('PCeS',
                                                     'PCes') + suffix
    return pd.concat(out, 1)


'''
Following: Utility functions and plotting of ROIS on brain.
'''


def to_edges(vals):
    delta = np.diff(vals)[0]
    edges = vals - delta / 2.
    edges = np.array(list(edges) + [edges[-1] + delta])
    return edges


@memory.cache
def plot_roi(hemi, labels, colors, view='parietal', fs_dir='/home/nwilming/fs_subject_dir'):
    import os
    subject_id = "fsaverage"
    surf = "inflated"
    brain = Brain(subject_id, hemi, surf, offscreen=True)
    for label, color in zip(labels, colors):
        label_file = os.path.join(fs_dir, subject_id, 'label',
                                  (label.replace('-rh', '.label')
                                   .replace('-lh', '.label')
                                   .replace('&', '_and_')
                                   .replace('_Havg', '')
                                   .replace('_Lateralized', '')))
        brain.add_label(label_file, color=color)
    brain.show_view(view)
    return brain.screenshot()
