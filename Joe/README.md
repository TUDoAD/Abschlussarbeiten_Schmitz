Code of Roberto Joe's master thesis

## Automation of a DN15 Kühni Extraction Column and Reactor using Image-Based Optical Measurement 

The algorithm is designed to extract the level measurements of the top layer and the interfacial layer and establish a control logic to maintain a fixed disengagement zone at the top of the DN15 Kühni extraction column. The algorithm follows a defined process:
<img src="https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/blob/main/Joe/Reference/Process%20Flowchart.png?raw=true" alt="Image error" width="700">

The algorithm starts by capturing an image using the Raspberry Pi HQ Camera which prompts the user to select the Region of Interest (ROI) and the two zones which would have the possible presence of layers, namely the top layer and the interfacial layer. 
(https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/blob/main/Joe/Reference/ROI%20Selection.png)

This ensures that the images are calibrated to a certain region of interest and this remains constant throughout the image acquisition phase until the algorithm is restarted or rebooted. 

Zone 1 will be the actual region that could have the presence of a top layer and an interfacial layer. Zone 2 will be the actual region that could have the presence of an interfacial layer only. 
(https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/blob/main/Joe/Reference/Zone%20Selection.png). 
Zone 2 is a subset of Zone 1 and can be selected just below the outlet of the extract phase in the DN15 extraction column. Both these zones need to be selected carefully such that none of the coordinates go outside the reactor at the disengagement zone. 












Needed Python modules (list may not be complete!):
```
pip install owlready2
pip install pythonnet
pip install win32.com client
```

