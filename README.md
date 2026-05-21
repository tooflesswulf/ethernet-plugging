# Readme

## Internet Connection Check (Mac)
 
### Issue
 
When connecting a MacBook to the robot/gripper network via Ethernet cable while also connected to the internet via WiFi, two problems may occur:
 
1. **Internet access is lost** — macOS routes all traffic through the Ethernet interface, which has no internet gateway.
2. **Cannot reach the gripper web interface** (`http://192.168.0.20`) — the Mac may not have a valid IP on the `192.168.0.x` subnet, or routing conflicts prevent local traffic from reaching the device.
### Setup
 
| Device   | Interface | IP Address      |
|----------|-----------|-----------------|
| MacBook  | WiFi      | (internet) |
| MacBook  | Ethernet  | 192.168.0.x     |
| Robot    | Ethernet  | 192.168.0.101   |
| Gripper  | Ethernet  | 192.168.0.20    |
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
 
To access the gripper web interface, navigate to:
 
```
http://192.168.0.20
```
 
> **Note:** Use `http://`, not `https://` — the gripper web interface does not use SSL by default.
