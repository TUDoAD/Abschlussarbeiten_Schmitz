Code of Roberto Joe's master thesis

## Automation of a DN15 Kühni Extraction Column and Reactor using Image-Based Optical Measurement 

The objective of this master’s thesis was to develop a level measurement and control of a DN15 Kühni extraction column and reactor using an image-based optical measurement. 
The DN15 Kühni extraction column was part of a modular plant and the objective was also to automate the modular plant such that the optical level measurement and control 
also works for different kinds of unit operations. The DN15 extraction column was not able to measure and control level at the disengagement zone by using conventional level 
measurement techniques due to physical limitations of the setup, and this paved the way for an image-based optical level measurement and control. The idea of optical level 
measurement started with a clear study into the specifications of the camera, lens and the single board computer which is capable to target a certain region (disengagement 
zone) of the Kühni extraction column and which can also cater to the increased processing capacity that the optical level control demands. 



The algorithm is designed to extract the level measurements of the top layer and the interfacial layer and establish a control logic to maintain a fixed disengagement zone 
at the top of the DN15 Kühni extraction column. The algorithm follows a defined process:
<img src="https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/blob/main/Joe/Reference/Process%20Flowchart.png?raw=true" alt="Image error" width="600">

The algorithm starts by capturing an image using the Raspberry Pi HQ Camera which prompts the user to select the Region of Interest (ROI) and the two zones which would have 
the possible presence of layers, namely the top layer and the interfacial layer. 

<img src="https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/blob/main/Joe/Reference/ROI%20Selection.png?raw=true" alt="Image error" width="300">

This ensures that the images are calibrated to a certain region of interest and this remains constant throughout the image acquisition phase until the algorithm is restarted 
or rebooted. 

Zone 1 will be the actual region that could have the presence of a top layer and an interfacial layer. Zone 2 will be the actual region that could have the presence of an interfacial layer only. 

<img src="https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/blob/main/Joe/Reference/Zone%20Selection.png?raw=true" alt="Image error" width="300"> 

Zone 2 is a subset of Zone 1 and can be selected just below the outlet of the extract phase in the DN15 extraction column. Both these zones need to be selected carefully 
such that none of the coordinates go outside the reactor at the disengagement zone. The segregation of zones is done to make the detections more robust and trustworthy. The 
detection of the top level is comparatively less cumbersome in comparison to the detection of the interfacial level. Zone 2 basically is the region where the interfacial 
level will be most likely present. The top coordinates of zone 2 is exactly at the outlet of the extract which is the boundary limit for an interlock to trip the pumps. In 
other words, it means that if the interfacial level reaches the top boundary of zone 2, the pumps should be tripped. The segregation of zones is to give a higher priority to 
the droplet candidate of zone 2. However, zone 1 is not completely neglected. Zone 1 shall also give a droplet candidate but it is second in line in terms of priority. If 
two distinct zones are not indicated or if both the zones are selected with the same coordinates, the program will still detect the interfacial level but some of the 
detections may be false detections as now the bubble candidate can be spread over a larger region and the topmost region is really not the intended region to maintain the 
interfacial level.  

Python modules required for operation of Raspberry Pi Optical Level Control (for continuous measurement):
1. cv2 
2. os
3. time
4. numpy
5. picamera2
6. opcua
7. pymodbus
8. collections.deque
9. datetime
10. threading

Python modules required for operation of Raspberry Pi Optical Level Control (for single image analysis):
1. cv2
2. tkinter
3. tkinter.filedialog
4. matplotlib.pyplot
5. matplotlib.gridspace
6. numpy

