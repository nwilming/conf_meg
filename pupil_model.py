IRFS = [yI/yI.sum(), ydnI/ydnI.sum(), ydtI/ydtI.sum()]
m, yh, y, X = pupil.eval_model('''
 pt.Z(pafilt) ~
                 pt.Z(left_gx) + pt.Z(left_gy) +
                 pt.MF(pt.Z(left_gx), IRFS) + pt.MF(pt.Z(left_gy), IRFS) + pt.MF(blink, IRFS) +
                 pt.MF(pt.Z(pt.dt(left_gx)), IRFS) + pt.MF(pt.Z(pt.dt(left_gy)), IRFS) +
                 pt.MF(conrast, func=IRFS) + pt.MF(dtcontrast, func=IRFS) +
                 pt.MF(pt.Cc(decision, levels=[21, 22, 23, 24]), IRFS) +
                 pt.MF(pt.Cc(feedback, levels=[-1,1]), IRFS) +
                 pt.MF(pt.evramp(decision, messages.decision_start_time, messages.decision_time), IRFS) +
                 pt.MF(pt.evramp(decision, messages.contrast_on-1000, messages.contrast_on-500), IRFS)
''', e)


mr, yhr, yr, Xr = pupil.eval_model('''
 pt.Z(pafilt) ~
                 pt.Z(left_gx) + pt.Z(left_gy) +
                 pt.MF(pt.Z(left_gx), IRFS) + pt.MF(pt.Z(left_gy), IRFS) + pt.MF(blink, IRFS) +
                 pt.MF(pt.Z(pt.dt(left_gx)), IRFS) + pt.MF(pt.Z(pt.dt(left_gy)), IRFS)
''', e)
