import rtde_control

ctrl = rtde_control.RTDEControlInterface('192.168.0.100')
ctrl.setCustomScriptFile('rtde_control-1.6.5-frictionfix.script')

while True:
    t0 = ctrl.initPeriod()
    ctrl.directTorque([0, 0, 0, 0, 0, 0])
    ctrl.waitPeriod(t0)

