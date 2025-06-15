Commands
```sh
python3 stream_zed.py 33738715
ros2 run apriltag_ros apriltag_node --ros-args \
    -r image_rect:=/zed/left/image_rect \
    -r camera_info:=/zed/left/camera_info \
    --params-file /home/ws/src/apriltag_ros/cfg/tags_36h11.yaml```