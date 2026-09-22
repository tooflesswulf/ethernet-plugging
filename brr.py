import rtde_control

ctrl = rtde_control.RTDEControlInterface('192.168.0.100')
ctrl.speedL([0, .05, 0, 0, 0, 0], time=1.0)
# while True:
#     t = ctrl.initPeriod()
#     ctrl.speedL([.01, 0, 0, 0, 0, 0])
#     ctrl.waitPeriod(t)
