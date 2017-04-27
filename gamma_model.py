

c = vstack(meta.contrast_probe)
r = prctile_rank(c.ravel(), [25, 50, 75, 100]).reshape(c.shape)

matrix = []
for rank in unique(r):
    d = p*0
    trial, pos = where(r==rank)
    print(len(trial))
    for t, i in zip(trial, pos):
        tpos = argmin(abs(time - (i/10.)))
        d[t, tpos] = 1
    for k in range(10):
        d = d.T.ravel()
        print(d.shape)
        matrix.append(roll(d.copy(), k))
