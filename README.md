# DanceFloorTracker

## Introduction
DanceFloorTracker is a fun party script to illuminate your dance floor that also allows for each person on the floor to move in their own spotlight and leave a glowing trace behind as they move around.

## Prerequesits
*Hardware:*
- Webcamera
- Projector

*Python packages:*
- Numpy
- OpenCV
- Scipy
- Matplotlib

## Usage:
DanceFloorTracker has 2 main usage modes. One allowing for simple processing and displaying of the video from teh camera and another for faithfully tracking and illinating moving objects and people.
###1. Just processing:
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
