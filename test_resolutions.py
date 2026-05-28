import pyrealsense2 as rs

# Create context
ctx = rs.context()

# Query connected devices
devices = ctx.query_devices()

if len(devices) == 0:
    print("No RealSense device found")
    exit()

for dev in devices:
    print(f"Device: {dev.get_info(rs.camera_info.name)} { dev.get_info(rs.camera_info.serial_number)}")

    print("-" * 50)

    sensors = dev.query_sensors()

    for sensor in sensors:
        print(f"\nSensor: {sensor.get_info(rs.camera_info.name)}")

        profiles = sensor.get_stream_profiles()

        for profile in profiles:
            vsp = profile.as_video_stream_profile()

            print(
                f"Stream: {vsp.stream_name():10s} "
                f"{vsp.width()}x{vsp.height()} "
                f"{vsp.fps():2d} FPS "
                f"Format: {profile.format()}"
            )
