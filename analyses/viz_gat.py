from pylab import *
import seaborn as sns
import pandas as pd


#gat = pd.read_hdf('results/gat_all.hdf')
def fix_times(time, epoch):
    time = time/600.
    if epoch=='stimulus':
        return time-.2
    elif epoch=='response':
        return time-1.5
    elif epoch=='feedback':
        return time-0.5
    raise RuntimeError('Do not recogize epoch')


def heatmap(pt, epoch, center=None, vmin=None, vmax=None):
    xedges = pt.index.values
    dt = diff(xedges)[0]
    xedges = linspace(min(xedges)-dt/2., max(xedges)+dt/2., len(xedges))

    yedges = pt.columns.values
    dt = diff(yedges)[0]
    yedges = linspace(min(yedges)-dt/2., max(yedges)+dt/2., len(yedges))

    if vmin is None or vmax is None:
        print('Setting vmin, vmax')
        vmin, vmax = pt.values.min(), pt.values.max()
        if center is not None:
            pad = max([vmax-center, center-vmin])
            vmin = center-pad
            vmax = center+pad

    xedges = fix_times(xedges, epoch)
    yedges = fix_times(yedges, epoch)

    pcolor(xedges, yedges, pt.values, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    xlim([min(xedges), max(xedges)])
    ylim([min(yedges), max(yedges)])
    axhline(0, color='k')
    axvline(0, color='k')
    plt.plot([xedges[0], xedges[-1]], [yedges[0], yedges[-1]], 'k--')
    if epoch=='stimulus':
        axhline(1, color='k')
        axvline(1, color='k')


def plot(gat, sensors='all', label='response', epoch='response', vmin=0.3, vmax=0.7):
    label_data = (gat.xs(sensors, level='sensors')
                    .xs(label, level='label')
                    .xs(epoch, level='epoch'))

    m = pd.pivot_table(label_data.reset_index(), values='accuracy', index='predict_time', columns='train_time')
    heatmap(m, epoch, vmin=vmin, vmax=vmax)


def plot_sextet(gat, sensors='all', labels=['response', 'confidence']):
    #figure(figsize=array([ 12,   7]))
    gs = matplotlib.gridspec.GridSpec(len(labels),3)
    for ie, epoch in enumerate(['stimulus', 'response', 'feedback']):

        for il, label in enumerate(labels):
            subplot(gs[il, ie])
            plot(gat, epoch=epoch, label=label, sensors=sensors)
            if il == 0:
                title('Epoch: ' + epoch, fontweight='bold')
                xticks([])
            else:
                xlabel('Train time')
            if not ie:
                text(-0.25, 0.5, 'Label:' + label,
                     fontweight = 'bold',
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform = gca().transAxes,
                     rotation=90)
                ylabel('Predict time')
                #yticks([])

    sns.despine()
    tight_layout()
    gcf().set_size_inches([12, 7], forward=True)

def plot_all(gat):
    for sensors in ['all', 'occipital', 'posterior', 'frontal', 'temporal']:
        sensor_data = gat.xs(sensors, level='sensors')
        figure()
        gs = matplotlib.gridspec.GridSpec(2, 3)
        for ie, epoch in enumerate(['stimulus', 'response', 'feedback']):
            epoch_data = sensor_data.xs(epoch, level='epoch')
            for il, label in enumerate(['response', 'confidence']):
                label_data = epoch_data.xs(label, level='label')
                m = pd.pivot_table(label_data.reset_index(), values='accuracy', index='predict_time', columns='train_time')
                subplot(gs[il, ie])
                sns.heatmap(m, center=0.5, xticklabels=20, yticklabels=20)
                ylim(ylim()[::-1])
        suptitle(sensors)
