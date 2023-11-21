# Eyantra-eyrc2023-luminosity-drone

Our code for the Luminosity Drone theme of the e-Yantra Robotics Competition 2023-24 (team LD 3811).

In this theme a small drone (the Swift drone) flies over an arena in Gazebo, finds groups of glowing LEDs
("alien life forms") with its downward camera, works out what type each one is from the number of LEDs
and reports where it is. The drone position comes from a whycon marker seen by an overhead camera.

## Tasks

| Folder | Script | What it does |
| --- | --- | --- |
| Task 0 | `result-2023-LD-0-20230910.json` | System check output (Ubuntu 20.04, ROS Noetic, Gazebo 11, Python 3.8) |
| Task 1A | `LD_3811_position_hold.py` | PID control to hold the drone at whycon position (2, 2, 20) |
| Task 1B | `LD_3811_led_detection.py` | Finds LEDs in an image and writes their count, centroids and areas |
| Task 2A | `LD_3811_waypoint_navigation.py` | Flies through 10 fixed waypoints, each reached within 0.2 in x, y and z |
| Task 2B | `LD_3811_life_form_detector.py` | Searches the arena for one LED cluster, reports it and lands |
| Task 2C | `LD_3811_biota_detector.py` | Same as 2B but for several clusters in one flight |

The `.bag` files are rosbag recordings of our runs that were submitted for evaluation. All of them
have `/whycon/poses`. The 2B and 2C bags also have the `/astrobiolocation` messages (1 in 2B, 3 in 2C).

## How it works

**PID.** The drone takes RC style commands (1000 to 2000) on `/drone_command`. Error in x drives roll,
error in y drives pitch and error in z drives throttle (around a hover value of 1560). Each axis has
its own Kp, Ki and Kd and the loop runs at 30 Hz. The gains can also be changed live from the
`pid_tune` sliders. The errors are published on `/roll_error`, `/pitch_error` and
`/alt_error` so they can be plotted.

**LED detection (1B).** Grayscale, Gaussian blur, threshold at 200, erode and dilate to remove noise,
then connected component labelling with scikit-image. Blobs bigger than 300 pixels are kept and their
contours give the centroid and area of each LED.

![Task 1B result](Task%201B/LD_3811_led_detection_results.png)

**Search (2B, 2C).** The drone flies a grid over the arena with a loose waypoint tolerance. The same LED
detection runs on every frame from `/swift/camera_rgb/image_raw`. When a cluster shows up the drone keeps
nudging its setpoint towards the side of the image the cluster is on until the cluster centre is within
5 pixels of the image middle in both x and y. It then publishes a `Biolocation` message with the type
(2 LEDs = `alien_a`, 3 = `alien_b`, 4 = `alien_c`) and its whycon x, y and z. In 2C it then carries on
along the route until the cluster is out of view, so the same cluster is not counted again. After the
search it flies to the research station at (11, 11) and disarms.

The centring code assumes a 500x500 camera image and the scale factors were picked by trial in the
simulator, so they will need changing for a different camera or height.

## Running

You need Ubuntu 20.04 with ROS Noetic and Gazebo 11 and the Luminosity Drone workspace from e-Yantra,
which has the Gazebo arena and the `swift_msgs`, `pid_tune` and `luminosity_drone` message packages.
These scripts are only our nodes, the workspace itself is not in this repo.

```bash
pip3 install -r requirements.txt
```

Start the launch file for the task from the e-Yantra workspace. Then in another terminal source the
workspace (`source devel/setup.bash` inside it) and run the node:

```bash
python3 "Task 2B/LD_3811_life_form_detector.py"
```

Task 1B does not need ROS:

```bash
cd "Task 1B"
python3 LD_3811_led_detection.py led.jpg
```

It writes `LD_3811_led_detection_results.png` and `LD_3811_led_detection_results.txt` in the current
folder. The result files in the repo are from our run on the task image.

## Videos

### Task 2B
[![Watch the video](https://img.youtube.com/vi/TBKd3Wz9AXI/maxresdefault.jpg)](https://youtu.be/TBKd3Wz9AXI)

### Task 2C
[![Watch the video](https://img.youtube.com/vi/NhVHN_IS7Ug/maxresdefault.jpg)](https://youtu.be/NhVHN_IS7Ug)
