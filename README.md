# Readme

## Environment setup

```
conda create -n realrobot python=3.11
pip install -r requirements.txt
```

Realsense: https://github.com/realsenseai/librealsense/blob/master/doc/installation.md#install-dependencies
for hidapi:

`sudo apt install libhidapi-dev`

To read dualsense joystick via hidapi without sudo previlege, see: https://askubuntu.com/questions/978552/how-do-i-make-libusb-work-as-non-root

For deep learning, install latest version of torch and torchvision. 

```
pip3 install torch torchvision
```

Lastly, add this line to ~/.bashrc
```
export PYTHONPATH=${PYTHONPATH}:"/path/to/ethernet-plugging"
source ~/.bashrc 
# then reactivate your conda environment
```

## Internet Connection Check (Mac)

### Issue

When connecting a MacBook to the robot/gripper network via Ethernet cable while also connected to the internet via WiFi, two problems may occur:

1. **Internet access is lost** — macOS routes all traffic through the Ethernet interface, which has no internet gateway.
2. **Cannot reach the gripper web interface** (`http://192.168.0.20`) — the Mac may not have a valid IP on the `192.168.0.x` subnet, or routing conflicts prevent local traffic from reaching the device.

### Setup

| Device  | Interface | IP Address    |
| ------- | --------- | ------------- |
| MacBook | WiFi      | (internet)    |
| MacBook | Ethernet  | 192.168.0.x   |
| Robot   | Ethernet  | 192.168.0.101 |
| Gripper | Ethernet  | 192.168.0.20  |

Your computer, robot, gripper will be connected to the same router.

### Solution

Set **WiFi above Ethernet** in macOS network service priority:

1. Open **System Settings → Network**
2. Click the **three dots (⋯)** menu → **Set Service Order**
3. Drag **WiFi above Ethernet**
4. Click OK and apply
   This ensures macOS routes internet traffic through WiFi while still using Ethernet for local network communication with the robot and gripper.

### Verify

```bash
# Check both are reachable
ping 192.168.0.20    # Gripper
ping 8.8.8.8         # Internet
```

## Gripper Notes

To access the gripper web interface, navigate to:

```
http://192.168.0.20
```

> **Note:** Use `http://`, not `https://` — the gripper web interface does not use SSL by default.

| Location                        | Item                 | Notes                                                                    |
| ------------------------------- | -------------------- | ------------------------------------------------------------------------ |
| Settings > Command Interface    | Text Based Interface | Enable for programmatic control via Python                               |
| Settings > Network              | IP Address           | Verify or update the gripper's IP here                                   |
| Settings > Motion Configuration | Force Limit          | Max gripping force (e.g., 80N)                                           |
| Settings > Motion Configuration | Part Clamping        | Note part width tolerance and clamping travel — adjust per task          |
| Diagnostics > Fingers           | Finger0              | Only Finger0 (2-cable side) has force sensing                            |
| Diagnostics > Fingers           | Grasping Force Value | Press one side → positive; other side → negative; idle → ~1N fluctuating |
| Diagnostics > Gripper           | —                    | _Todo_                                                                   |
| Motion > Manual Control         | Manual Homing        | Opens the fingers                                                        |
| Motion > Manual Control         | Grip                 | Closes the fingers with current parameters                               |
| Help > Documentation            | —                    | Full WSG 50 docs available here                                          |

**Turn-off** the gripper by turn off the power strip (not the silver box, which is the power supply).

## Robot Notes

### Starting & Shutting Down

**To start:** On the teach pendant screen → **ON** → **START** (some noise is normal) → **Exit**

**To shut down:**

- **Emergency:** Hit the E-stop button
- **Normal:** Click the bottom-left corner → **OFF**

After exiting the start panel, you will land on the **Run** panel (indicated in the top-left corner).

### Local vs. Remote Control

By default the robot may be in **Remote** mode. To switch to local control:

- Top-right corner → toggle **Remote → Local**
- Once in Local mode, the **Program**, **Installation**, and **Move** panels (top-left) will brighten and become accessible.

### Moving the Robot

Navigate to the **Move** panel (top-left, Local mode only). Two options:

| Method                   | How to                                                                                                     |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| **Screen controls**      | Use the virtual buttons on the left side of the screen                                                     |
| **Freedrive (screen)**   | Click **Freedrive** at the bottom → panel appears on the right → press & hold, then move the robot by hand |
| **Freedrive (physical)** | Press & hold the physical button on the back-top of the teach pendant, then move the robot by hand         |
