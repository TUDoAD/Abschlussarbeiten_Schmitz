Code of Neetika Sains master thesis

## Creation of a digital Twin 

The work in this repository builds upon and adapts the research from Behr, A.S., on DWSIM-EnzymeML-KG. The codes provided here modified from the source work and utilized in the development of a digital twin.


This repository contains the code and the MS Excel files (subdirectory [ELNs](https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/tree/main/Sain/ELNs)) used as data input for the creation of a digital twin as a process simulations in [DWSIM](https://dwsim.org).
Since the workflow is adapted from researchj based on investigation of Enzyme reactions, hence an Enzyme based excel file is also a part of this workflow. The base file is stored here in [EnzymeML](https://enzymeml.github.io/services/).  
Another supplymentary excel file that contains all the relevant data for the creation of a process flow sheet and generation of a reaction is provided as an input and stored in the subdirectory [ELNs] (https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/tree/main/Sain/ELNs). 

The data from the lab is integrated in this supplymentary excel file and the calculation of parameters for the arrhenius equation is perfromed via an additional python script [Vant_hoff_plot_selection.py] [https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/blob/main/Sain/Vant_hoff_plot_selection.py]. This generates the vant hoff plots which are stored in [Plots](https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/tree/main/Sain/Plots). 
This script is integrated in the main file also, but it can be exceuted separtely as well to validate the arrhenius parameters calculated via this script. 

To obtain the process simulation, the ontology [./ontologies/BaseOntology_for_CSTR.owl](https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/blob/main/Sain/ontologies/BaseOntology_for_CSTR.owl) is loaded and extended by the data contained in both Excel-files by the code contained in [ELNs_to_KG_modules.py]
(https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/blob/main/Sain/ELNs_to_KG_modules.py).

This yields a knowledge graph ([./ontologies/KG-DWSIM_CSTR_ELN.owl](https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/blob/main/Sain/ontologies/KG-DWSIM_CSTR_ELN.owl)) that is loaded by the code [DWSIM_modules.py](https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/blob/main/Sain/DWSIM_modules.py) and used to generate the DWSIM-file, stored in the subdirectory [DWSIM](https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/tree/main/Sain/DWSIM).
Furthermore, the data obtained by the process simulation is also stored in the knowledge graph ([./ontologies/KG-DWSIM_CSTR_ELN_output.owl](https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/blob/main/Sain/ontologies/KG-DWSIM_CSTR_ELN_output.owl)) enabling a holistic storage of the related data.

The overall workflow is depicted below, showing the overall data integration. Starting with laboratory data recorded in supplymentary file and relevant ELNs and rate of reaction, data is stored as a variable in Python and in a structured manner with the help of a tailored ontology as a knowledge graph. 
Then, the recorded data is used to automatically generate process simulations, resulting in further insights and eased workflow from laboratory to process simulation data. 

<img src="https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/blob/main/Sain/Image/Picture1.png" alt="Image error" width="700">


The working principle of the code to create the DWSIM-simulation files is depicted in the figure below. This is executed by the `run()` function contained in [DWSIM_modules.py](https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/blob/main/Sain/DWSIM_modules.py)


<img src="https://github.com/TUDoAD/Abschlussarbeiten_Schmitz/blob/main/Sain/Image/FigureB.png" alt="Image error" width="600">



## Installation
Installation of  [DWSIM](https://dwsim.org)-software- Must be done- It is recommended to install the latest version of the DWSIM. Latest version as of today (07/01/2025) is dwsim_8.8.3
Installation of Protege (Ontology Editor) (https://protege.stanford.edu/)- software- If ontology devlopment is part of scope, this software muste be downloaded


Needed Python modules (list may not be complete!):
```
pip install owlready2
pip install pythonnet
pip install win32.com client
```

