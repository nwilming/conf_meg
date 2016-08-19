import numpy as np
import cPickle

mapname = '/Users/nwilming/u/conf_analysis/meg/key_map.pickle'

try:
  cache = cPickle.load(open(mapname))
except IOError:
  cache = {'start':0}

def hash(x):
    x = tuple(x)
    try:
        return cache[x]
    except KeyError:
        cache[x] = max(cache.values())+1
        return cache[x]

def save():
    cPickle.dump(cache, open(mapname, 'w'), protocol=2)
