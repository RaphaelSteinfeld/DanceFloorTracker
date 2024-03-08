#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DanceFloorTracker is a fun party script to illuminate your dance floor that also allows for each person on the floor to move in their own spotlight and leave a glowing trace behind as they move around.
Created on Sat Feb 24 13:07:52 2024

@author: raphael steinfeld
"""
import numpy as np
import cv2 as cv
import platform
from scipy.signal import convolve2d 
import matplotlib.pyplot as plt


class DanceFloorTracker:
    
   def __init__(self, CamID=0,  ImWidth=1280, ImHeight=800, ColMaps='all',
                ColTransition=True, DurColMap=10, TransitionTime=5, Mirror=True):
        
        # Initialize Camera
        self.cap = cv.VideoCapture(CamID)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        
        # If there is an issue with the camer - try to fix it
        if not self.cap.isOpened():
            self.cap.open()    
            # if you cannot, give up
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()
        
        # Determine operating system
        self.os = platform.system()
                
        # Proportions of screen used (in px, not used for WINDOWS)
        self.ImWidth = ImWidth
        self.ImHeight = ImHeight    
         
        
        
        
        # Prepare the colormap parameters for the tracking image
        self.Thresh = 255*.125   # Cuttoff to binarize optic flow
        self.Decay = .965        # Exponential decay rate. Controls how mong past movement r
        if CamID == 0:
            # Adjust FilterSize 
            self.FilterSize = 37   # Filtersize for Gaussian blurr. Controls how wide the traces blurr
        else:
            self.FilterSize = 7   # If using internal webcam, use smaller filter.
        
        # List of available colormaps
        self.ColorMaps = np.array([cv.COLORMAP_INFERNO,
                         cv.COLORMAP_HOT, 
                         cv.COLORMAP_BONE, 
                         cv.COLORMAP_OCEAN, 
                         cv.COLORMAP_DEEPGREEN,
                         cv.COLORMAP_CIVIDIS,
                         cv.COLORMAP_TURBO,
                         cv.COLORMAP_COOL,
                         cv.COLORMAP_PARULA])
        
        # Define which colormaps to use
        if ColMaps == 'all':
            self.ColorMaps = self.ColorMaps
        else:
            self.ColorMaps = self.ColorMaps[ColMaps]
        
        
        # Time [s] the functions stays with each colourmap b5efore switching
        self.NumColMaps = len(self.ColorMaps) # Number of selected colormaps      
        self.ColTransition = ColTransition    #Are the ColorMaps changing Smoothly?
        self.TransitionTime = TransitionTime     #Time[s] for blending from one colormap to the next


        # Other camera settings
        self.FS = 30 # Desired Framerate
        self.DurColMap = DurColMap*self.FS # Number of frames each colormap will be used
       
        if self.ColTransition:   
            self.TransitionTime *= self.FS # Calculate number of frames the colormap blending will take
            self.BlendingFactors = np.arange(self.TransitionTime+1)/self.TransitionTime
            
        self.cap.set(cv.CAP_PROP_FPS, self.FS)
        
        # Create a fancy logo to show during calibration
        logo = cv.imread('Starter.png')
        self.logo = cv.resize(cv.cvtColor(logo, cv.COLOR_BGR2GRAY), (self.ImWidth, self.ImHeight))/255
        self.linear = np.nan
        self.screen = np.nan
        self.mirror = Mirror
      
        
   def ConfirmLinear(self):
       assert not np.isnan(self.linear), 'Distortion of camera is unknown. Run PartyTacker.MeasureDistorion() first.'
       ##############################################################################
       #   Confirm Removal of Distortion
       ##############################################################################
       if self.os == 'WINDOWS':
           cv.namedWindow('frame', cv.WND_PROP_FULLSCREEN)
           cv.setWindowProperty('frame',cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
           cv.waitKey(10)
       Ongoing = 1
       while Ongoing==1:   
           # Capture frame-by-frame
           ret, img = self.cap.read()
           #frame = cv.undistort(img, mtx, dist, None, newcameramtx)
           frame = cv.remap(img, self.mapx, self.mapy, cv.INTER_LINEAR)
           x, y, w, h = self.roi
           frame = frame[y:y+h, x:x+w]
           cv.imshow('frame',cv.resize(frame,(self.ImWidth, self.ImHeight)))
           if cv.waitKey(10) & 0xFF == ord('q'):
              cv.destroyAllWindows()
              cv.waitKey(150)
              Ongoing=0
                           
   def ConfirmSelection(self):
       assert not np.isnan(self.linear), 'Distortion of camera is unknown. Run PartyTacker.MeasureDistorion() first.'
       assert not np.isnan(self.screen), 'Position of window in camera image is unknown. Run PartyTacker.FindScreen() first.'  
       ##############################################################################
       #   Confirm correct detection of screen in camera image
       ##############################################################################
       if self.os == 'WINDOWS':
           cv.namedWindow('frame', cv.WND_PROP_FULLSCREEN)
           cv.setWindowProperty('frame',cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
           cv.waitKey(10)
       Ongoing = 1
       while Ongoing==1:   
             # Capture frame-by-frame
             ret, img = self.cap.read()
             while ret == False: 
                 ret, img = self.cap.read()
                 
             ##############################################################################
             #   REMOVE DISTORTIONS AND FIND PICTURE
             ##############################################################################       
             x, y, w, h = self.roi
             frame = cv.remap(img, self.mapx, self.mapy, cv.INTER_LINEAR)   
             frame = frame[self.Horz[0]+y:self.Horz[1]+y,self.Vert[0]+x:self.Vert[1]+x]   
            
             cv.imshow('frame',cv.resize(frame,(self.ImWidth, self.ImHeight)))
             if cv.waitKey(10) & 0xFF == ord('q'):
                 cv.destroyAllWindows()
                 cv.waitKey(150)
                 Ongoing=0    
           
   def Display(self,FinalSignal,counter):
        
        im_color = cv.applyColorMap(np.round(FinalSignal*255).astype('uint8'), self.ColorMaps[int(np.floor(counter/self.DurColMap))])
        
        if self.ColTransition and self.DurColMap - (counter % self.DurColMap) <= self.TransitionTime:
            
            Position = self.TransitionTime  - (self.DurColMap - (counter % self.DurColMap) )
            Factor = self.BlendingFactors[Position]
            if np.floor(counter/self.DurColMap) == self.NumColMaps-1:
                im_color2 = cv.applyColorMap(np.round(FinalSignal*255).astype('uint8'), self.ColorMaps[0])
            else:        
                im_color2 = cv.applyColorMap(np.round(FinalSignal*255).astype('uint8'), self.ColorMaps[int(np.floor(counter/self.DurColMap))+1])
                    
            im_color = (im_color*(1-Factor) + im_color2* Factor).astype('uint8')
            
        
        if counter >= (self.NumColMaps) * self.DurColMap - 1:
            counter = 0
        
        cv.imshow('frame', cv.resize(im_color,(self.ImWidth, self.ImHeight)))
        
        return counter
    
   def FindCorners(self):
       Nodes = [7,7]
         
       # Prepare everything for camera calibration
       # termination criteria
       criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)   
       # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
       objp = np.zeros((Nodes[0]*Nodes[1],3), np.float32)
       objp[:,:2] = np.mgrid[0:Nodes[0],0:Nodes[1]].T.reshape(-1,2)
       # Arrays to store object points and image points from all the images.
       objpoints = [] # 3d point in real world space
       imgpoints = [] # 2d points in image plane.
          
       happy = 0
       while happy == 0:
           # take image
           ret, frame = self.cap.read()
           # Convert camera image to grayscale
           gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
           
             # Find the chess board corners
           ret, corners = cv.findChessboardCorners(gray, (Nodes[0],Nodes[1]), None)
           if ret == True:
               objpoints.append(objp)
               corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
               imgpoints.append(corners2)
               happy+=1
               # Draw and display the corners
               
               
           else:
               print('Issue_left')
       return objpoints, imgpoints  

   def FindScreen(self):   
       assert not np.isnan(self.linear), 'Distortion of camera is unknown. Run PartyTacker.MeasureDistorion() first.'
       ##############################################################################
       #   FIND SCREEN IN CAMERA IMAGE
       ##############################################################################

       Vert = np.zeros((3,2))
       Horz= np.zeros((3,2))
       
       if self.os == 'WINDOWS':
           cv.namedWindow('frame', cv.WND_PROP_FULLSCREEN)
           cv.setWindowProperty('frame',cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
           cv.waitKey(10)
       
       cv.imshow('frame', self.logo)
       cv.waitKey(20) 
       x = 1
       while x==1:
          cv.imshow('frame', self.logo)
          if cv.waitKey(10) & 0xFF == ord('q'):
             x=0

       for c in range(3):    
           CLogo = np.expand_dims(255+0*self.logo, 2)
           CLogo = np.concatenate((CLogo,CLogo,CLogo),axis=2)
           cv.imshow('frame', CLogo)
           cv.waitKey(20) 
               
           ret, img = self.cap.read()  
           frame = cv.remap(img, self.mapx, self.mapy, cv.INTER_LINEAR)
          
           x, y, w, h = self.roi
           frame = frame[y:y+h, x:x+w]
           
           Vert[c,:], Horz[c,:]= self.ScreenPosition(frame,c)
          
       cv.destroyAllWindows()
       cv.waitKey(150)
       gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
           
       Horz2 = np.zeros((2,)).astype('int')
       Vert2 = np.zeros((2,)).astype('int')
       Horz2[0] = np.median(Horz[:,0])
       Horz2[1] = np.median(Horz[:,1])
            
       Vert2[0] = np.median(Vert[:,0])
       Vert2[1] = np.median(Vert[:,1])         
 
       self.Horz = Horz2
       self.Vert = Vert2
       
       plt.imshow(frame)
       plt.title('Full linear image')
       plt.show()       
       
       plt.imshow(gray[Horz2[0]:Horz2[1],Vert2[0]:Vert2[1]],cmap='gray')
       plt.title('Selected area')
       plt.show()
       
       self.screen = 1

       del Horz2, Vert2
       return frame

                
   def JustFilter(self):   
        
        # Initialise the window    if in windows
        if self.os == 'Windows':
            cv.namedWindow('frame', cv.WND_PROP_FULLSCREEN)
            cv.setWindowProperty('frame',cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
            cv.waitKey(10)
        
        counter = 0              
        while True:
            counter+=1
                
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            while ret == False:
                ret, frame = self.cap.read()
            if self.mirror:
                gray = cv.flip(cv.cvtColor(frame, cv.COLOR_BGR2GRAY),1)
            else:
                gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
                
            if not 'Frames' in locals():
                shapes = gray.shape
                Frames = np.zeros((shapes[0],shapes[1],2))
                FinalSignal=np.zeros(shapes)
               
            FinalSignal = self.Process(Frames,gray,FinalSignal)
            counter = self.Display(FinalSignal,counter)
                
            if cv.waitKey(1) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                cv.waitKey(150)
                break
    
   def MeasureDistortion(self,whiteframe = 'logo'):
       
        ##############################################################################
        #   MEASURE CAMERA DISTORTION
        ##############################################################################
        ImHeight = self.ImHeight
        ImWidth = self.ImWidth
        
        if self.os == 'WINDOWS':
            cv.namedWindow('frame', cv.WND_PROP_FULLSCREEN)
            cv.setWindowProperty('frame',cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
            cv.waitKey(10)
        if whiteframe == 'logo':
            whiteframe = self.logo    
        elif np.sum(whiteframe)== 0:
            whiteframe = np.ones((ImHeight,ImWidth))
        x = 1
        while x==1:
            cv.imshow('frame',whiteframe)
            if cv.waitKey(10) & 0xFF == ord('q'):
                x=0

        Border = 20
        NumSquares = 10
    
    
        #Create a chessboard pattern
        SquareSize = int((ImHeight-2*Border)/NumSquares)
        checkerboard = np.zeros((ImHeight-2*Border,ImHeight-2*Border))
        sequence = np.arange(int(ImWidth/SquareSize))
        for y in range(int(ImWidth/SquareSize)):
            for x in range(int(ImWidth/SquareSize)):
                checkerboard[x*SquareSize:(x+1)*SquareSize,y*SquareSize:(y+1)*SquareSize]=sequence[x]+y
        checkerboard= np.mod(checkerboard,2)*255
    
    
        # Make 2 pictures, one for the left, another for the right half of the screen
        checkerboard_test1 = np.ones((ImHeight,ImWidth))*127
        checkerboard_test1[Border:ImHeight-Border,Border:ImHeight-Border] = checkerboard
    
        checkerboard_test2 = np.ones((ImHeight,ImWidth))*127
        checkerboard_test2[Border:ImHeight-Border,ImWidth-ImHeight+Border:ImWidth-Border] = checkerboard
        
        # Show the first pattern and identify position of corners
        cv.imshow('frame', cv.applyColorMap(checkerboard_test1.astype('uint8'), cv.COLORMAP_INFERNO))
        cv.waitKey(10)        
        objpoints, imgpoints = self.FindCorners()  
        cv.waitKey(100)      
        
        # Now do the right side of the screen
        cv.imshow('frame', cv.applyColorMap(checkerboard_test2.astype('uint8'), cv.COLORMAP_INFERNO))
        cv.waitKey(10)   
        objpoints2, imgpoints2 = self.FindCorners()  
        cv.waitKey(100)    
        cv.destroyAllWindows()
        cv.waitKey(150)
        
        # Append the results:
        objpoints = np.append(objpoints,objpoints2,axis=0)
        imgpoints = np.append(imgpoints,imgpoints2,axis=0)
       
        
        # Take new image
        ret, frame = self.cap.read()
    
        # Convert camera image to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
     
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        img = frame
     
        h,  w = img.shape[:2]
     
        newcameramtx, self.roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        self.mapx, self.mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
        self.linear = 1     
    
   def PositionCamera(self):
        
        ##############################################################################
        #   Position Camera
        ##############################################################################
        Ongoing = 1
        if self.os == 'WINDOWS':
            cv.namedWindow('frame', cv.WND_PROP_FULLSCREEN)
            cv.setWindowProperty('frame',cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
            cv.waitKey(10)
        while Ongoing==1:   
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            cv.imshow('frame',cv.resize(frame,(self.ImWidth, self.ImHeight)))
            if cv.waitKey(10) & 0xFF == ord('q'):
                Ongoing=0
                cv.destroyAllWindows()
                cv.waitKey(150)
            
   def Process(self,Frames,gray,FinalSignal):
        Signal = []
        
        Frames[:,:,1:]=Frames[:,:,:-1]
        Frames[:,:,0]= gray
        
        # Look for pixels with high change
        Change =  np.diff(Frames) > self.Thresh * 1.0
        
        # exclude some pixels at the frame for stability
        Change[0:2,:]=0
        Change[-2:,:] = 0
        Change[:,0:2]=0
        Change[:,-2]=0
        
        
        Signal = np.concatenate((np.expand_dims(FinalSignal,2)*self.Decay,Change),2)
       
       
        FinalSignal = np.amax(Signal,-1)
        FinalSignal = cv.GaussianBlur(FinalSignal,(self.FilterSize,self.FilterSize),cv.BORDER_DEFAULT)
        
        return FinalSignal        
   
   def ScreenPosition(self,image,dim):
       
        ##############################################################################
        #   EDGE DETECTION
        ##############################################################################

        # image = np.squeeze(image[:,:,dim])
        # image = np.expand_dims(image, 2)
        # image = np.concatenate((image,image,image),axis=2)
             
        
        image = cv.Canny(image,100,200)
        
        V_Filter = np.ones((3,3))
        V_Filter[:,1]=0
        V_Filter[:,2]=-1
        
        H_Filter = np.ones((3,3))
        H_Filter[1,:] = 0
        H_Filter[2,:] = -1
        
        Vertical = convolve2d(image,V_Filter,mode='same')
        Horizontal = convolve2d(image,H_Filter,mode='same')
        
        Vert =  np.mean(Vertical,axis=0)
        Horz =  np.mean(Horizontal,axis=1)
        
        Vert = [np.argmin(Vert[20:int(len(Vert)/2)]), 
                np.argmax(Vert[int(len(Vert)/2):len(Vert)-20])+int(len(Vert)/2)]
       
        Horz = [np.argmin(Horz[20:int(len(Horz)/2)]), 
                np.argmax(Horz[int(len(Horz)/2):len(Horz)-20])+int(len(Horz)/2)]
        
        return Vert,Horz   
   
   def ShutDown(self):
         self.cap.release()
         cv.destroyAllWindows()
         cv.waitKey(500)

   def Track(self):                
       assert not np.isnan(self.linear), 'Distortion of camera is unknown. Run PartyTacker.MeasureDistorion() first.'
       assert not np.isnan(self.screen), 'Position of window in camera image is unknown. Run PartyTacker.FindScreen() first.'  
       
       
       # Initialise the window    if in windows
       if self.os == 'Windows':
           cv.namedWindow('frame', cv.WND_PROP_FULLSCREEN)
           cv.setWindowProperty('frame',cv.WND_PROP_FULLSCREEN,cv.WINDOW_FULLSCREEN)
           cv.waitKey(10)
       
       cv.imshow('frame', self.logo)
       cv.waitKey(20) 
       x = 1
       while x==1:
           cv.imshow('frame', self.logo)
           if cv.waitKey(10) & 0xFF == ord('q'):
              x=0
              
       counter = 0   
       x, y, w, h = self.roi        #Prepare variables for   removal of distortions  
       while True:
           counter+=1
               
           # Capture frame-by-frame
           ret, frame = self.cap.read()
           while ret == False:
               ret, frame = self.cap.read()

               
           #REMOVE DISTORTIONS AND FIND PICTURE    
           frame = cv.remap(frame, self.mapx, self.mapy, cv.INTER_LINEAR)   
           frame = frame[self.Horz[0]+y:self.Horz[1]+y,self.Vert[0]+x:self.Vert[1]+x]   
           gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
               
           if not 'Frames' in locals():
               shapes = gray.shape
               Frames = np.zeros((shapes[0],shapes[1],2))
               FinalSignal=np.zeros(shapes)
              
           FinalSignal = self.Process(Frames,gray,FinalSignal)
           counter = self.Display(FinalSignal,counter)
               
           if cv.waitKey(1) & 0xFF == ord('q'):
               cv.destroyAllWindows()
               cv.waitKey(150)
               break
   