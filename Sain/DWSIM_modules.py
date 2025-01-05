# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:04:30 2024

@author: Alexander Behr
"""

import os
import uuid

import clr 

from owlready2 import *

# Importiere Python Module
import pythoncom
import System
pythoncom.CoInitialize()

from System.IO import Directory, Path, File
from System import String, Environment
from System.Collections.Generic import Dictionary

import ELNs_to_KG_modules
# Path to DWSIM-Directory

dwsimpath = os.getenv('LOCALAPPDATA') + "\\DWSIM\\"

clr.AddReference(dwsimpath + "DWSIM")
clr.AddReference(dwsimpath + "CapeOpen.dll")
clr.AddReference(dwsimpath + "DWSIM.Automation.dll")
clr.AddReference(dwsimpath + "DWSIM.Interfaces.dll")
clr.AddReference(dwsimpath + "DWSIM.GlobalSettings.dll")
clr.AddReference(dwsimpath + "DWSIM.SharedClasses.dll")
clr.AddReference(dwsimpath + "DWSIM.Thermodynamics.dll")
clr.AddReference(dwsimpath + "DWSIM.UnitOperations.dll")
clr.AddReference(dwsimpath + "DWSIM.Inspector.dll")
clr.AddReference(dwsimpath + "System.Buffers.dll")
clr.AddReference(dwsimpath + "DWSIM.MathOps.dll")
#clr.AddReference(dwsimpath + "TcpComm.dll")
#clr.AddReference(dwsimpath + "Microsoft.ServiceBus.dll")
clr.AddReference(dwsimpath + "DWSIM.FlowsheetSolver.dll")
clr.AddReference("System.Core")
clr.AddReference("System.Windows.Forms")
clr.AddReference(dwsimpath + "Newtonsoft.Json")
clr.AddReference(dwsimpath + "SkiaSharp.dll")
clr.AddReference("System.Drawing")

from DWSIM.Interfaces.Enums.GraphicObjects import ObjectType
import DWSIM.Thermodynamics.BaseClasses as ThermodynamicsBases
from DWSIM.Thermodynamics import Streams, PropertyPackages
from DWSIM.UnitOperations import UnitOperations, Reactors
from DWSIM.UnitOperations import UnitOperations
from DWSIM.Automation import Automation3
from DWSIM.GlobalSettings import Settings

from enum import Enum
# Paket, um Kalkulationen durchzuführen 
from DWSIM import FlowsheetSolver
# Paket, um ein neues Fließbild zu erstellen und darauf zuzugreifen
from DWSIM import Interfaces
from System import *

from System.Linq import *
from DWSIM import *
#from DWSIM import FormPCBulk
from DWSIM.Interfaces import *
from DWSIM.Interfaces.Enums import*

# Paket, um Fließbild zu zeichnen
from DWSIM.Interfaces.Enums.GraphicObjects import *

# Paket, um neu erstellte Komponenten als JSON datei abzuspeichern
from Newtonsoft.Json import JsonConvert, Formatting

from DWSIM.Thermodynamics import*
from DWSIM.Thermodynamics.BaseClasses import *
from DWSIM.Thermodynamics.PropertyPackages.Auxiliary import *

# Pakte, um Pseudocompound Creator auszuführen
from DWSIM.Thermodynamics.Utilities.PetroleumCharacterization import GenerateCompounds
from DWSIM.Thermodynamics.Utilities.PetroleumCharacterization.Methods import *

# for the pfd to be saved in png format

from SkiaSharp import SKBitmap, SKImage, SKCanvas, SKEncodedImageFormat
from System.IO import MemoryStream
from System.Drawing import Image
from System.Drawing.Imaging import ImageFormat

##
def flowsheet_simulation(onto, pfd_iri):
    #enz_dict, pfd_dict, onto, pfd_iri): 
    working_dir = os.getcwd()
    Directory.SetCurrentDirectory(dwsimpath)
    
    # Create automatin manager
    interf = Automation3()
    
    sim = interf.CreateFlowsheet()
    sim.CreateAndAddPropertyPackage("Soave-Redlich-Kwong (SRK) Advanced")
    

    ## 
    pfd_ind = onto.search_one(iri = pfd_iri)
    pfd_list = pfd_ind.BFO_0000051 # has part
        
    comp_list = [] # lists the components contained in the PFD
    process_streams = [] # lists the stream info for the flowsheet 

            
    # "subst_indv": ontology_individual, "subst_class": ontology_class, "subst_role": role of the individual in the PFD (reactant, product, catalyst,...)
    for module in pfd_list:
        if module.is_a[0].label.first() == "EnergyStream":
            # add energy streams to stream_names for later input in stream creation
            process_streams.append(module)
        
        elif module.is_a[0].label.first() == "MaterialStream":
            materialstream = module.BFO_0000051 # has part
            
            # add material streams to stream_names for later input in stream creation
            process_streams.append(module)
            
            for comp in materialstream:
                mat = comp.RO_0002473.first() # composed primarily of
                subst = mat.is_a.first() 
                role = mat.RO_0000087.first().name # has role
                comp_list.append({"subst_indv":mat, "subst_class": subst, "subst_role":role})
                #print(mat,subst, role) # substance individual, substance class, role [product, reactant]
        try:
            if module.RO_0000087.first().name == "product":# has role
                subst = module.is_a.first()
                role = module.RO_0000087.first() # has role
                comp_list.append({"subst_indv":module, "subst_class": subst, "subst_role":role})#print(module,module.is_a,module.RO_0000087.first().name) # substance individual, substance class, role [product, reactant]
        except:
            pass
    
    #loading components into DWSIM-simulation and filling dictionaries regarding
    # stoichiometric coefficients and reaction order coeffs.
    
    comps = Dictionary[str, float]()
    dorders = Dictionary[str, float]()
    rorders = Dictionary[str, float]()
    
    
    for comp in comp_list:
        # add label of class (= substance name) to the DWSIM-Simulation
        # comp
        subst_class_name = comp["subst_class"].label.first()
        stoich_coeff = comp["subst_indv"].hasStoichiometricCoefficient.first()
        dorder_coeff = comp["subst_indv"].hasDirect_OrderCoefficient.first()
        rorder_coeff = comp["subst_indv"].hasReverse_OrderCoefficient.first()
        
        # add compount to dwsim simulation class
        sim.AddCompound(subst_class_name)
        
        # add coefficents to dictionaries to prepare for creation of reaction
        comps.Add(subst_class_name, stoich_coeff)
        dorders.Add(subst_class_name, dorder_coeff)
        rorders.Add(subst_class_name, rorder_coeff) 
        
        if comp["subst_role"] == "catalyst":
            #has characteristic -> kinetics
            kin_indv = comp["subst_indv"].RO_0000053 
            substrate_indv = []
            for indv in kin_indv: # might be more than one substrate
                # has input -> input = substrate of reaction    
                substrate_indv.append(indv.RO_0002233)
    
    ## Add streams to DWSIM:
    
    # Add starting streams of flow sheet
    stream_info = []
    #for later reference, streams lists the dwsim-object-representation of the streams 
    streams = {}
    # Start at y = 0, x=0
    y_axis = 0
    for stream_indv in process_streams:
        # if the property output of (RO_0002353) returns an empty list -> Start of the flowsheet
        if not stream_indv.RO_0002353:
            #print(stream_indv.label)
            stream_type = stream_indv.is_a[0].label.first()
            stream_name = stream_indv.label.first()
            codestr = """stream = sim.AddObject(ObjectType.{}, 0,{},'{}')
streams['{}'] = stream""".format(stream_type,y_axis,stream_name,stream_name)
            print(codestr)
            #codestr = """stream_info.append({{'type': ObjectType.{}, 'x': 0, 'y': {}, 'name': '{}'}})""".format(stream_type,y_axis,stream_name)
            code = compile(codestr, "<string>","exec")
            exec(code)
            y_axis += 50
            if stream_indv.is_a[0].label.first() == "MaterialStream":
                
                subst_indv = stream_indv.BFO_0000051 # has part
                
                ## set molar flows of compounds
                for sub_stream in subst_indv:
                    substance = sub_stream.RO_0002473.first().is_a.first().label.first() #composed primarily of
                    if sub_stream.hasCompoundMolarFlowUnit.first().replace(" ","") in ["mol/s","mols^-1"]:
                        mol_flow = float(sub_stream.hasCompoundMolarFlow.first())
                    else:
                        print("compound molar flow unit not recognized: {} in stream {}".format(substance,sub_stream.label.first()))
                        mol_flow = 0
                    
                    #print("stream_name: {}, substance:{}, mol_flow:{}".format(stream_name, substance, mol_flow))
                    streams[stream_name].GetAsObject().SetOverallCompoundMolarFlow(substance,mol_flow)
                
                # set overall volume flow
                try: 
                    if stream_indv.hasVolumetricFlowUnit.first().replace(" ","") in ["m3/s","m^3s^-1", "m^3/s", "m3s-1"]:
                        streams[stream_name].GetAsObject().SetVolumetricFlow(float(stream_indv.overallVolumetricFlow.first()))
                except:
                    print(stream_name + ": No volumetric flow defined")
                
                
                ## set temperature
                if subst_indv.first().hasTemperature:
                    if subst_indv.first().hasTemperatureUnit.first() in ["C","c","°c", "°C","Celsius","celsius"]:
                        temp = float(subst_indv.first().hasTemperature.first()) + 273.15
                    else:
                        temp = float(subst_indv.first().hasTemperature.first())
                    streams[stream_name].GetAsObject().SetTemperature(temp)
                
                

    #Add the streams and other objects (mixer, reactor, ...) to the simulation Flowsheet
    stream_info = []
    y_axis = 0
    x_axis = 100   
    codestr = ""
    for stream_indv in process_streams:
        # if the property output of (RO_0002353) returns an empty list -> Start of the flowsheet
        #if not stream_indv.RO_0002353: # output of -> starting streams
        next_modules = stream_indv.RO_0002234 # has output
        for module in next_modules:                
            module_type = module.is_a[0].label.first()
            module_name = module.label.first()
            module_names = list(streams.keys())
            
            # check, if module was already added to the simulation
            # only when true, go further downstream and add the has output streams
            if module_name not in module_names:
                codestr = """stream = sim.AddObject(ObjectType.{},{},{},'{}')\n""".format(module_type,x_axis, y_axis,module_name)
                codestr += """streams['{}'] = stream""".format(module_name)
                print(codestr)
                code = compile(codestr, "<string>","exec")
                exec(code)
                x_axis += 100
                
                # take a look on next stream, going out from last module
                next_streams = module.RO_0002234
                print(next_streams)
                for stream in next_streams:
                    stream_type = stream.is_a[0].label.first()
                    stream_name = stream.label.first()
                    stream_names = [i["name"] for i in stream_info] #why is this created? this condition will alwyas true
                    if stream_name not in stream_names: 
                        codestr = """stream = sim.AddObject(ObjectType.{},{},{},'{}')\n""".format(stream_type,x_axis,y_axis,stream_name)
                        codestr += """streams['{}'] = stream\n""".format(stream_name)
                        print(codestr)
                        code = compile(codestr, "<string>","exec")
                        exec(code)
                        #eval("streams['{}'] = sim.AddObject(ObjectType.{},{},{},'{}')".format(stream_name,stream_type,x_axis,y_axis,stream_name))
                        #codestr = """stream_info.append({{'type': ObjectType.{}, 'x': {}, 'y': {}, 'name': '{}'}})""".format(stream_type,x_axis, y_axis,stream_name)
                        x_axis += 100

    #iterate through pfd_list connect the objects, direction of connection comes
    # with RO_0002234 (has output) and RO_0002353 (output of)
    for pfd_obj in process_streams:
        obj_name = pfd_obj.label.first()  
        obj_1 = streams[obj_name].GetAsObject().GraphicObject
        
        output_objects = pfd_obj.RO_0002234 # has_output -> obj_1 connected to obj_2
        input_objects = pfd_obj.RO_0002353 # output of -> obj_2 connected to obj_1
        
        for out_obj in output_objects:
            obj_2_name = out_obj.label.first()
            obj_2 = streams[obj_2_name].GetAsObject().GraphicObject
            sim.ConnectObjects(obj_1,obj_2, -1,-1)
        
        for inp_obj in input_objects:
            obj_2_name = inp_obj.label.first()
            obj_2 = streams[obj_2_name].GetAsObject().GraphicObject
            sim.ConnectObjects(obj_2,obj_1, -1,-1)
            
    
    ## Add special information to modules
    
    #Adding the Pumps
    Pumps = []
    for module in pfd_list:
        if module.is_a[0].label.first() in ["Pump"]:
  
           Pumps.append(module)
           dwsim_object = streams[module.label.first()].GetAsObject()
           
           dwsim_object.CalcMode = dwsim_object.CalculationMode(
               int(module.hasCalculationType.first()))
           
           dwsim_object.Pout= float(module.hasOutletPressure.first())
           dwsim_object.Eficiencia = float(module.hasEfficiency.first())
           
               
    #Adding the Heaters           
    Heaters = []
    for module in pfd_list:
        if module.is_a[0].label.first() in ["Heater"]:

           Heaters.append(module)
           dwsim_object = streams[module.label.first()].GetAsObject()
           
           dwsim_object.CalcMode = dwsim_object.CalculationMode(
               int(module.hasCalculationType.first()))
           dwsim_object.DeltaP = float(module.hasPressureDrop.first())
           dwsim_object.Eficiencia = float(module.hasEfficiency.first())
           dwsim_object.OutletTemperature = float(module.hasOutletTemperature.first())
         
               
    #Adding the Reactor 
    reactor_list = []
    for module in pfd_list:
        # Add information to reactors
        if module.is_a[0].label.first() in ["RCT_PFR","RCT_Conversion","RCT_Equilibrium","RCT_Gibbs","RCT_CSTR"]:
            # Note: This part of the code is flexible to implement any of the Reactors 
            dwsim_obj = streams[module.label.first()].GetAsObject()
            reactor_list.append(module)
            
            dwsim_obj.ReactorOperationMode = Reactors.OperationMode(int(module.hasTypeOf_OperationMode.first()))
            dwsim_obj.Volume= float(module.hasVolumeValue.first())
            dwsim_obj.Headspace = float(module.hasHeadSpace.first())
    
    ##
    #Arrhenius Parameters
    pre_exponential_factor_class = onto.search_one(label="pre-exponential factor")
    activation_energy_class = onto.search_one(label="activation energy")
    arreh_eq = onto.search_one(label="Arrhenius equation")
    
    for individual in onto.individuals():
        if pre_exponential_factor_class in individual.is_a:
            print(f"Pre-exponential factor (A): {individual.name} - Value: {individual.hasValue}")
            pefa = individual.hasValue # Replace `hasValue` with the actual property name
        if activation_energy_class in individual.is_a:
            print(f"Activation energy (E): {individual.name} - Value: {individual.hasValue}") 
            aec = individual.hasValue
        if arreh_eq in individual.is_a:
            arreq_class = individual
    
    kr1 = sim.CreateKineticReaction('kr0',"this is a saponification reaction", comps, dorders,
                                    rorders, "Ethyl_Acetate", "Liquid", "Molar Concentrations",
                                    "mol/L", "mol/[L.s]", float(pefa[0]), float(aec[0]), 0.0,
                                    0.0, "", "")
    
    sim.AddReaction(kr1)
    sim.AddReactionToSet(kr1.ID, "DefaultSet", True, 0)
    
  
    errors = interf.CalculateFlowsheet4(sim)
    if (len(errors) > 0):
        for e in errors:
            print("Error: " + e.ToString())    

    Directory.SetCurrentDirectory(working_dir)
    
    PFDSurface = sim.GetSurface()
    bmp = SKBitmap(1000, 600)
    canvas = SKCanvas(bmp)
    canvas.Scale(0.5)
    PFDSurface.ZoomAll(bmp.Width, bmp.Height)
    PFDSurface.UpdateCanvas(canvas)
    d = SKImage.FromBitmap(bmp).Encode(SKEncodedImageFormat.Png, 100)
    str1 = MemoryStream()
    d.SaveTo(str1)
    image = Image.FromStream(str1)
    imgPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.Desktop), "Saponification.png")
    image.Save(imgPath, ImageFormat.Png)
    str1.Dispose()
    canvas.Dispose()
    bmp.Dispose()
    
    return sim, interf, streams, pfd_list

##
def extend_knowledgegraph(sim,onto,streams, pfd_list,pfd_iri):
    
    for datProp in ["overallMolarFlow","hasMolarFlowUnit", "hasMolarity","hasMolarityUnit"]:
        onto = ELNs_to_KG_modules.datProp_from_str(datProp,onto)
        
    pfd_ind = onto.search_one(iri = pfd_iri)
    pfd_dict = {} 
    phase_dict = {}
    
    for i in pfd_list:
        pfd_dict[i.label.first()]=i
    
    for stream in streams:
        dwsim_obj = streams[stream].GetAsObject()
        onto_obj = pfd_dict[stream]
        
        if "MaterialStream" in onto_obj.is_a.first().label:
            stream_comp_ids = list(dwsim_obj.ComponentIds)
            stream_composition = list(dwsim_obj.GetOverallComposition())
            molar_flow = dwsim_obj.GetMolarFlow()
            volume_flow = dwsim_obj.GetVolumetricFlow()
            
            ## get phase information
            phases_dict = {}
            for phase_no in range(dwsim_obj.GetNumPhases()):
            
                mol_flow = dict(dwsim_obj.get_Phases())[phase_no].Properties.get_molarflow() #mol/s
                vol_flow = dict(dwsim_obj.get_Phases())[phase_no].Properties.get_volumetric_flow() #m3/s
                
                #print(onto_obj.label)
                if mol_flow and vol_flow:                
                    f = mol_flow / vol_flow /1000 # mol/L
                    conc_dict = {}
                    
                    
                    for i in range(len(list(dwsim_obj.GetPhaseComposition(int(phase_no))))):
                        conc_dict[stream_comp_ids[i]] = f * list(dwsim_obj.GetPhaseComposition(int(phase_no)))[i]
                        #conc_list.append(conc_dict)
                    
                    phase_name = str(dict(dwsim_obj.get_Phases())[int(phase_no)].ComponentName)
                    if phase_name != "Mixture":
                        phases_dict[phase_name] = conc_dict
            #
            phase_dict[str(onto_obj.label.first())] = phases_dict
            ##
            
            ## add information to ontology
            onto_obj.overallVolumetricFlow = [str(volume_flow)]
            onto_obj.hasVolumetricFlowUnit = ["m3/s"]
            onto_obj.overallMolarFlow = [str(molar_flow)]
            onto_obj.hasMolarFlowUnit = ["mol/s"]
            #

            # add Molarities to the sub-material streams
            if onto_obj.BFO_0000051: # has part (partial material stream)
                for submat_stream in onto_obj.BFO_0000051:
                    material_label = submat_stream.RO_0002473[0].is_a[0].label.first()
                    #submat_stream.label.first()
                    conc_dict = phase_dict[onto_obj.label.first()]
                    
                    for phase in conc_dict:
                        key_list = conc_dict[phase].keys()
                        #add molarities
                        if material_label in key_list:
                            submat_stream.hasMolarity = [str(conc_dict[phase][material_label])]
                            submat_stream.hasMolarityUnit = ["mol/L"]
                        
                        # assert phase
                        if "Liquid" in phase: #DWSIM asserts "OverallLiquid" for liquid phases
                            submat_stream.hasAggregateState.append("Liquid")
                        else:
                            submat_stream.hasAggregateState.append(phase)# Vapor,..
         
            
            else: #no partial material stream(s) detected or missing       
                conc_dict = phase_dict[onto_obj.label.first()]
                stream_name = onto_obj.label.first()
                
                for phase in conc_dict:
                    key_list = conc_dict[phase].keys()
                    
                    for subst in key_list:
                        #only extend, if Molarity != 0
                        if conc_dict[phase][subst] != 0:
                            onto, substream_uri = onto_substream_from_name(onto, stream_name, subst)
                            substream = onto.search_one(iri = substream_uri)
                            
                            onto_obj.BFO_0000051.append(substream)#hasPart
                            ## search for the correct substance individual in pfd_dict
                            for key in pfd_dict:
                                if pfd_dict[key].is_a.first().label.first() == subst:
                                    substream.RO_0002473.append(pfd_dict[key]) #consists primarily of
                        
                            # add molarities
                            if subst in key_list:
                                substream.hasMolarity = [str(conc_dict[phase][subst])]
                                substream.hasMolarityUnit = ["mol/L"]
                        
                            # assert phase
                            if "Liquid" in phase: #DWSIM asserts "OverallLiquid" for liquid phases
                                substream.hasAggregateState.append("Liquid")
                            else:
                                substream.hasAggregateState.append(phase)# Vapor,..
                   
    return onto

##

def onto_substream_from_name(onto, stream_name, subst_name):
    uuid_str = "PFD_" + str(uuid.uuid4()).replace("-","_")
    substream = onto.search_one(label = "MaterialStream")(uuid_str)
    substream.label = stream_name + "_" + subst_name
    substream_iri = substream.iri  
    
    return onto, substream_iri



##
def save_simulation(sim,interface, filename):
    fileNameToSave = Path.Combine(os.getcwd(),filename)
    interface.SaveFlowsheet(sim, fileNameToSave, True)

##



##
def run(filename_DWSIM,PFD_uuid,KG_path):#filename_DWSIM,filename_KG):

    onto = owlready2.get_ontology(KG_path).load()
    onto.name = "onto"
    
    filename_KG = KG_path.replace(".owl","_output.owl")
    
    pfd_ind = onto.search_one(iri = "*"+PFD_uuid)
    
    print("Data initialized, ontology loaded...")
    
    pfd_iri = pfd_ind.iri
    sim, interface, streams,pfd_list = flowsheet_simulation(onto,pfd_iri)
    
    print("Storing DWSIM-file: "+filename_DWSIM)
    save_simulation(sim,interface,filename_DWSIM)
    
    print("Integrating new information into Knowledge Graph")
    onto = extend_knowledgegraph(sim, onto, streams, pfd_list, pfd_iri)
    print("Storing Knowledge Graph: "+filename_KG)
    onto.save(file =filename_KG, format ="rdfxml")
        