'''
Analyze behavior of single trials
'''


def trials_by_column(data, col):
    '''
    Takes data, df that contains all trials for one cluster, 
    and sorts by series col.
    '''
    # Make sure data has only trials as index
    names = data.index.names
    for name in names:
        if name == 'trial':
            continue
        data.index = data.index.droplevel(name)

    svals = col.index
    data = data.loc[svals, :]
    assert(all(data.index.values == svals.values))
    return data

def trials_by_index(data, col):
    # Make sure data has only trials as index
    names = data.index.names
    for name in names:
        if name == 'trial':
            continue
        data.index = data.index.droplevel(name)

    svals = col.index
    data = data.loc[svals, :]
    assert(all(data.index.values == svals.values))
    return data
