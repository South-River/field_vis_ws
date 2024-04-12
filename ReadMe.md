# Repository for RA-L response
**Requirements**: Ubuntu 20.04, ROS-Noetic

## Guidance

Compilation
```c++
cd ~/field_vis_ws
catkin_make
```

Running the code

field visualization
```c++
source ~/field_vis_ws/devel/setup.bash
roslaunch field_vis field_vis.launch
```
convergency test
```c++
source ~/field_vis_ws/devel/setup.bash
rosrun field_vis lbfgs_test
```

## Visualize different formation
Modify the file in src/field_vis/launch/field_vis.launch to visualize different formation.
0 for triangle, 1 for square, 2 for star, 3 for hexagon.