import rtde_control
import rtde_receive
import numpy as np
import time

ctrl = rtde_control.RTDEControlInterface('192.168.0.100')
ctrl.setCustomScriptFile('rtde_control-1.6.5-frictionfix.script')
recv = rtde_receive.RTDEReceiveInterface('192.168.0.100')

VIS = [.9, .9, .8, .9, .9, .9]
COU = [.9, .8, .8, .7, .8, 1.]

stamps = []
joints = []

try:
    while True:
        t0 = ctrl.initPeriod()
        ctrl.directTorque([0, 0, 0, 0, 0, 0], VIS, COU)
        q = recv.getActualQ()
        print(q, end='\r')
        stamps.append(time.perf_counter())
        joints.append(q)
        ctrl.waitPeriod(t0)
except KeyboardInterrupt:
    ctrl.stopScript()
    print('saving...')
    # np.savez('brr-log.npz', times=stamps, joints=joints)
