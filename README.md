# DanceFloorTracker

## Introduction
DanceFloorTracker is a fun party script to illuminate your dance floor that also allows for each person on the floor to move in their own spotlight and leave a glowing trace behind as they move around.

## Prerequisites
*Hardware:*
- Webcam
- Projector
<br />
<br /> 

*Python packages:*
- Numpy
- OpenCV
- Scipy
- Matplotlib

## Usage:
DanceFloorTracker has 2 main usage modes. One allowing for simple processing and displaying of the video from the camera and another for faithfully tracking and illuminating moving objects and persons.
### 1. Just processing:
To quickly get a feel about how the processed video stream looks like, use the JustFilter() method:
```
from DanceFloorTracker import DanceFloorTracker
DFT = DanceFloorTracker()
DFT.JustFilter()
```
To stop the stream, press 'q' and to release the camera, use the ShutDown method:
```
DFT.ShutDown()
```
### 2. Exact tracking:
To exactly track moving objects and display their traces in the correct position, calibration of the camera and identification of the projected area within the video are required before running the Track method.
The recommended procedure for this method is:
```commandline
from DanceFloorTracker import DanceFloorTracker
DFT = DanceFloorTracker()
# OPTIONAL: Stream the video from the mcamera to make sure it captures the full projected area:
DFT.PositionCamera() # press q to quit

# Measure the fisheye distortion of the camera and calculate remapping:
DFT.MeasureDistortion()
# Optional: Results can be seen using:
DFT.ConfirmLinear() # press q to quit

# Identify projected area in camera image:
DFT.FindScreen()
# OPTIONAL: Results can be confirmed using DFT.ConfirmSelection()

# These calibration steps can be run in advance and teh reuslts and the main tracking function can then be started an any time using:
DFT.Track()
```
To stop the stream, press 'q' and to release the camera, use the ShutDown method:
```
DFT.ShutDown()
```

## Input arguments:
**CamID** Index of camera to be initialised. Defaults to 0. **ImWidth** Width of projector resolution in px. Defaults to 1280 **ImHeight**Same for image height Defaults to 800 
**ColMaps** Specifies the set of colormaps to be used for displaying. Available list of colormaps: inferno,
hoy, bone, ocean, deepgreen, cividis, turbo, cool, parula.
Accepts an array of integers determining the position of selected colormaps, or 'all' to select all available colormaps. Defaults to 'all'. 
**ColTransition** Specifies if changes in the selected colormaps are smooth, by interpolating colormaps (True) or abrupt (False). Defaults to True.
**DurColMap** Duration in seconds each colormap is being used before transitioning to the next. Defaults to 10. 
**TransitionTime** Determines the duration [s] of transitioning between colormaps. Defaults to 5. Only used if ColTransition is True. 
**Mirror** Determines if the camera picture is flipped vertically. Defaults to True. Only used in JustFilter method. 

## Examples
See example video to get an idea of the working Track method and the best dog ever :)   


![Example.gif](Example.gif)  

![Example1.gif](Example1.gif)