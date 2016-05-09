import patsy
from pylab import *
class empty_transform(object):
    '''
    Transforms events into
    '''
    def __init__(self):
        pass

    def memorize_chunk(self, *args, **kwargs):
        pass

    def memorize_finish(self, *args, **kwargs):
        pass

class Zscore(empty_transform):
    def transform(self, x):
        return (x-nanmean(x))/nanstd(x)


class DT(empty_transform):
    def transform(self, x):
        return r_[[0], diff(x)]


class Cc_t(empty_transform):
    def transform(self, x, levels=None):
        if levels is None:
            levels = unique(x)
        out = zeros((len(x), len(levels)))
        for i, level in enumerate(levels):
            idx = x==level
            out[:, i] = idx

        return out

class boxcar_t(empty_transform):
    def transform(self, x, pre=10, post=10, val='normalized'):
        if val == 'normalized':
            val = 1./(pre+post)
        idx = where(x)[0]
        for index in idx:
            x[idx-pre:idx+post] = val
        return x

class ramp_t(empty_transform):
    def transform(self, x, pre=10, post=10, ramp_type='upramp', start=0., end=1.):
        try:
            x = x.values
        except AttributeError:
            pass
        vals = start + arange(pre+post) * (end/(pre+post))
        if ramp_type == 'upramp':
            pass
        elif ramp_type == 'downramp':
            vals = vals[::-1]
        else:
            vals = ramp_type(arange(pre+post))
        if len(x.shape) == 1:
            idx = where(x[:])[0].ravel()
            for index in idx:
                x[index-pre:index+post] = vals
        else:
            for i in range(x.shape[1]):
                idx = where(x[:,i])[0].ravel()
                for index in idx:
                    x[index-pre:index+post, i] = vals
        return x

class event_ramp(empty_transform):
        def transform(self, x, start, end, pre=10, post=10, ramp='boxcar'):
            # This is a bit tricky because start and end can be different things
            # For now I assume that start and end are indices (maybe multi-indices)
            # Pre and post are in sample space
            x = x*0
            for s, e in zip(start.index.values, end.index.values):
                s = tuple(s[:-1] + ((s[-1]-pre),))
                e = tuple(e[:-1] + ((e[-1]+post),))
                v=1
                if ramp=='upramp':
                    v = linspace(0,1,len(x.loc[s:e]))
                elif ramp=='downramp':
                    v = linspace(1,0,len(x.loc[s:e]))
                x.loc[s:e] = v
            return x

class convolution(empty_transform):
    '''
    A stateful transform for patsy that convolves regressors with some function

    TODO: Documentation.
    '''
    def __init__(self):
        pass

    def transform(self, x, func=[1]):
        try:
            x = x.values
        except AttributeError:
            pass
        func = pad(func, [len(func), 0], mode='constant')
        if len(x.shape) > 1:
            out = array([convolve(t, func, mode='same') for t in x.T])
            return out.T
        return convolve(x, func, mode='same')

class multi_convolution(empty_transform):
    '''
    A stateful transform for patsy that convolves regressors with some function

    TODO: Documentation.
    '''
    def __init__(self):
        pass

    def transform(self, x, func=[1]):
        try:
            x = x.values
        except AttributeError:
            pass
        func = asarray(func)
        if len(func.shape)==1:
            out = self.conv(x, func)
            return out.T
        else:
            out = vstack([self.conv(x, f) for f in func])
            return out.T

    def conv(self, x, f):
        func = pad(f, [len(f), 0], mode='constant')
        if len(x.shape) > 1:
            out = vstack([convolve(t, f, mode='same') for t in x.T])
            return out
        out = convolve(x, func, mode='same')
        return out


class spline_convolution(object):
    '''
    A stateful transform for patsy that convolves regressors with a spline basis
    function.

    TODO: Documentation.
    '''
    def __init__(self):
        pass

    def memorize_chunk(self, x, **kwargs):
        pass

    def memorize_finish(self, **kwargs):
        pass

    def transform(self, x, degree=2, df=5, length=100):
        try:
            x = x.values
        except AttributeError:
            pass
        knots = [0, 0] + linspace(0,1,df-1).tolist() + [1,1]
        basis = patsy.splines._eval_bspline_basis(linspace(0,1,length), knots, degree)
        basis /= basis.sum(0)
        if len(x.shape) > 1 and not (x.shape[1] == 1):
            return r_[[self.conv_base(t, basis, length) for t in x]]
        else:
            return self.conv_base(x.ravel(), basis, length)

    def conv_base(self, x, basis, length):
        out = empty((len(x), basis.shape[1]))
        for i, base in enumerate(basis.T):
            out[:,i] = convolve(x, pad(base, [length, 0], mode='constant'), mode='same')
        return out

Cc = patsy.stateful_transform(Cc_t)
box = patsy.stateful_transform(boxcar_t)
ramp = patsy.stateful_transform(ramp_t)
BS = patsy.stateful_transform(spline_convolution)
F = patsy.stateful_transform(convolution)
MF = patsy.stateful_transform(multi_convolution)
Z = patsy.stateful_transform(Zscore)
evramp = patsy.stateful_transform(event_ramp)
dt = patsy.stateful_transform(DT)
