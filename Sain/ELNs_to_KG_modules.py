# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 13:58:00 2023

@author: Alexander Behr
@author: Elnaz Abbaspour
@author: Neetika Sain 
"""


####################################################
## Ontology Manipulation
####################################################

from owlready2 import *
import uuid

import pyenzyme as pe
import json
import pandas as pd


## 
# To make sure, owlready2 is able to use HermiT for reasoning, configure the path to the java interpreter
# e.g.:
# owlready2.JAVA_EXE = "C://Users//..//Protege-5.5.0-win//Protege-5.5.0//jre//bin//java.exe"
##


def eln_subst_data_to_dict(eln_sheet):
    ext_eln_data = {}
    for col, d in eln_sheet.items():
        if col != "Property":
            sub_name = eln_sheet[eln_sheet['Property'].str.contains('Name')][col].iloc[0]
            sub_name = sub_name.strip().replace(' ','_') if sub_name == str else sub_name 
            if pd.notna(sub_name): ext_eln_data[sub_name] = {}
           # if sub_name in list(ext_eln_data.keys()):
            for index, row in eln_sheet.iterrows():
                if pd.notna(row[col]) and row["Property"] != "Name":
                    ext_eln_data[sub_name][row["Property"]] = row[col]
    
    return ext_eln_data


####
# parsing information from new/additional ELN into dictionary
#
def new_ELN_to_dict(eln_path):
    
    ELN_xlsx = pd.ExcelFile(eln_path)
    eln_sheet = pd.read_excel(ELN_xlsx,'Substances and Reactions')
    
    kin_index = eln_sheet[eln_sheet['Property'].str.contains('Kinetic Parameters', na=False)].index.min()
    eln_sheet_properties = eln_sheet.loc[:kin_index -1].dropna(how='all')
    eln_sheet_kin_params = eln_sheet.loc[kin_index + 1:].dropna(how='all').dropna(axis=1, how='all')
    
    ext_eln_data = {}
    subst_eln_data = {}
    kin_eln_data = {}
    
    # load substances and properties into dictionary
    """
    for col, d in eln_sheet_properties.items():
        if col != "Property":
            sub_name = eln_sheet_properties[eln_sheet_properties['Property'].str.contains('Name')][col].iloc[0].strip()
            subst_eln_data[sub_name] = {}
            for index, row in eln_sheet_properties.iterrows():
                if pd.notna(row[col]) and row["Property"] != "Name":
                    subst_eln_data[sub_name][row["Property"]] = row[col]
     """
    subst_eln_data = eln_subst_data_to_dict(eln_sheet_properties)
    
    ## adding dicts to already existing dict
    for sheet_name in ['Properties for JSON-file', 'Additional Info (Units)']:
        eln_sheet_properties = pd.read_excel(ELN_xlsx,sheet_name)
        add_dict = eln_subst_data_to_dict(eln_sheet_properties)
        subst_eln_data = {key.strip().replace(' ','_'): {**subst_eln_data.get(key, {}), **add_dict.get(key, {})} for key in set(subst_eln_data) | set(add_dict)}    
    
    ##
    
    # extract kinetic parameters into dictionary
    for col, d in eln_sheet_kin_params.items():
        if col != "Property":
            kin_name = eln_sheet_kin_params[eln_sheet_kin_params['Property'].str.contains('Name')][col].iloc[0].strip().replace(' ','_')
            kin_eln_data[kin_name] = {}
            for index, row in eln_sheet_kin_params.iterrows():
                if pd.notna(row[col]) and row["Property"] != "kineticName":
                    kin_eln_data[kin_name][row["Property"]] = row[col]
    
    kinetic_eln_data = eln_subst_data_to_dict(eln_sheet_kin_params)
    
    # load PFD data
    # Sheet PFD
    eln_sheet = pd.read_excel(ELN_xlsx,"PFD")
    pfd_eln_data = {}
    for index, row in eln_sheet.iterrows():
        pfd_eln_data[row["Object-name"].strip()] = {'DWSIM-object type':row["DWSIM-object type"].strip(),
                                       'DWSIM-object argument':int(row["DWSIM-object argument"]) if pd.notna(row["DWSIM-object argument"]) else None, 
                                       'connection':row["output connected to"].strip() if pd.notna(row["output connected to"]) else None,
                                       'overallVolumetricFlow':row["overallVolumetricFlow"] if pd.notna(row["overallVolumetricFlow"]) else None,
                                       'hasVolumetricFlowUnit':row["hasVolumetricFlowUnit"].strip() if pd.notna(row["hasVolumetricFlowUnit"]) else None,
                                       }
    
    # Sheet Material Streams
    eln_sheet = pd.read_excel(ELN_xlsx,"Material Streams")
    matstream_dict = eln_subst_data_to_dict(eln_sheet)    
    for subst in matstream_dict: 
        if "EntersAtObject" in matstream_dict[subst]:
            pfd_eln_data[matstream_dict[subst]["EntersAtObject"]].update({subst:matstream_dict[subst]})
    
    ## Sheet Reactor Specification
    eln_sheet = pd.read_excel(ELN_xlsx,"Reactor Specification")    
    react_dict = {}
    
    for index, row in eln_sheet.iterrows():
        react_dict[row["Property"]] = row["Value"]
    
    try: 
        pfd_eln_data[react_dict["isDWSIMObject"]].update(react_dict)   
    except:
        print('Warning: Sheet Reactor Specification in ELN misses proper DWSIM Object!')
    
    ## Sheet Pump Specification
    eln_sheet = pd.read_excel(ELN_xlsx,"Pump_Specification")
    pump_dict ={}
    
    value_cols = [c for c in eln_sheet.columns if c.startswith('Value')]
    
    for col in value_cols:
        pump_dict ={}
        for i in range(len(eln_sheet)):
            pump_dict[eln_sheet.Property.iloc[i]] = eln_sheet[col].iloc[i]
    
        try:
            pfd_eln_data[pump_dict["isDWSIMObject"]].update(pump_dict)
        except:
            print('Warning: Sheet Pump_Specification in ELN misses proper DWSIM Object!')

    ## Sheet Heater Specification
    eln_sheet = pd.read_excel(ELN_xlsx,"Heater_Specification")
    heater_dict ={}
    
    value_cols = [c for c in eln_sheet.columns if c.startswith('Value')]
    
    for col in value_cols:
        heater_dict ={}
        for i in range(len(eln_sheet)):
            heater_dict[eln_sheet.Property.iloc[i]] = eln_sheet[col].iloc[i]
    
        try:
            pfd_eln_data[heater_dict["isDWSIMObject"]].update(heater_dict)
        except:
            print('Warning: Sheet Heater_Specification in ELN misses proper DWSIM Object!')
    
    
    ##
    # join substances related data and PFD-related data    
    ext_eln_data["substances"] = subst_eln_data   
    ext_eln_data["PFD"] = pfd_eln_data 
    ext_eln_data["kinetics"] = kinetic_eln_data
    # 
    
    return ext_eln_data

####


#####
# Ontology-Extension der Base Ontology #
#####
def base_ontology_extension(path_base_ontology):
    #TODO: Deprecate this function and include the two classes 
    # into the initial base-ontology manually
    # Only supports owl-ontologies
    # load base ontology
    onto_world = owlready2.World()
   # sbo_onto = onto_world.get_ontology("https://raw.githubusercontent.com/EBI-BioModels/SBO/master/SBO_OWL.owl").load()
    onto = onto_world.get_ontology(path_base_ontology).load()
    onto.name = "onto"
    
   # onto.imported_ontologies.append(sbo_onto)
    

    #SBO has some classes that contain XMLs describing mathematical formulas
    # loading the ontology with owlready2 results in the XML-classes of the formulas
    # being interpreted as ontology classes and annotation properties
    # Thus, deletion of these classes takes place here
    
    """
    SBO_annotation_classes = ["apply","bvar","ci","cn","degree","divide","factorial","floor","lambda","lowlimit","minus","plus","power","product","root","selector","semantics","sum","tanh", "times","uplimit", "ln", "log","MathMLmath"]
    for dep_class in SBO_annotation_classes:
        codestr ="class_list = onto.search(iri = '*{}')\nfor indv in class_list: destroy_entity(indv)".format(dep_class)
        code = compile(codestr, "<string>","exec")
        try: 
            exec(code)
        except:
            print(codestr)
    """        
    with onto:
        # Komponenten: DWSIM stellt 6 Datenbanken zur verfügung (DWSIM, ChemSep, Biodiesel, CoolProp, ChEDL and Electrolytes)
        # Daraus ergeben sich 1500 verfügbare Komponenten für die Simulation
        # Datenbanken werden der metadata4Ing Klasse 'ChemicalSubstance' subsumiert
        
        # Bool to state whether the substance is already contained in DWSIM Database
        class hasDWSIMdatabaseEntry(DataProperty):
            label = 'hasDWSIMdatabaseEntry'
            range = [bool]
            pass
        
        class isImportedAs(DataProperty):
            label = 'isImportedAs'
            range = [str]
            pass
        
    return onto

# dynamic creation of substances based on the enzymeML document and the additional eln
def subst_classes_from_dict(enzmldoc, subst_dict, onto):
    #      
    # iterate through each substance from subst_dict and include it in ontology
    
    enzymeML_subst_parameters = ["smiles","inchi"]
    
    for subst in list(subst_dict.keys()):
        
        try: 
            enzml_ID = subst_dict[subst]["hasEnzymeML_ID"]
        except:
            enzml_ID = ''
            
        # include as individual, if label is already present as class
        if onto.search_one(label = subst):
            codestring = """with onto:
                            substance_indv = onto.search_one(label = "{}")('ind_{}')
                            substance_indv.label = 'Sub_{}_{}'
                """.format(subst, subst, subst, enzml_ID)
       
        # include as individual, if part in IRI is already present
        elif onto.search_one(iri = "*{}".format(subst)):
            codestring = """with onto:
                            substance_indv = onto.search_one(iri = "*{}")('ind_{}')
                """.format(subst, subst)
        
        # include as class and individual of class and search in enzymeML doc for
        # the substance
        
        else:   
            try:
                subst_superclass = enzmldoc.getAny(subst_dict[subst]["hasEnzymeML_ID"]).ontology.value.replace(':','_')              
                enzml_name = enzmldoc.getAny(subst_dict[subst]["hasEnzymeML_ID"]).name
            except: 
                subst_superclass = "SBO_0000247" # Simple Chemical
                enzml_name = ''

            
            if enzml_name:
                codestring = """with onto:
                            class {}(onto.search_one(iri = '*{}')):
                                label = '{}'
                                altLabel = '{}' 
                                pass                    
                            substance_indv = {}('ind_{}')
                            substance_indv.label = 'Sub_{}_{}'
                            substance_indv.altLabel = '{}'
                    """.format(subst, subst_superclass, subst, enzml_name, subst, subst,subst, enzml_ID, enzml_name)
            else:
                codestring = """with onto:
                            class {}(onto.search_one(iri = '*{}')):
                                label = '{}'
                                pass                    
                            substance_indv = {}('ind_{}')
                            substance_indv.label = 'Sub_{}_{}'
                    """.format(subst, subst_superclass, subst, subst, subst,subst,enzml_ID)
       
        # compile codestring
        code = compile(codestring, "<string>", "exec")
        # Execute code
        exec(code)
        
        #iterate through enzymeML_subst_parameters and include their properties where suited
        if "hasEnzymeML_ID" in list(subst_dict[subst].keys()):
            for param in enzymeML_subst_parameters:
                try:
                    value = enzmldoc.getAny(subst_dict[subst]["hasEnzymeML_ID"]).dict()[param]
                except:
                    value = None
                
                if value: 
                    codestring = """with onto:
                            onto.search_one(label = '{}').{} = '{}'            
                            """.format(subst, param, value)
                        # compile codestring
                    # print(codestring)
                    code = compile(codestring, "<string>", "exec")
                    # Execute code
                    exec(code)

    return onto

#
def datProp_from_str(data_prop_name, onto):
    # creates new dataproperty based on input string
    # if the label does not already exist as data property
    codestring = """with onto:
        if not onto.search_one(label = "{}"):
            class {}(DataProperty):
                label = '{}'
                pass
        """.format(data_prop_name, data_prop_name,data_prop_name)
    code = compile(codestring, "<string>","exec")
    exec(code)
    return onto
#

def datProp_from_dict(dataProp_dict, onto):
    # Benötigte Relationen bestimmen via set() -> auch bei Mehrfachnennung
    # ist jede Relation aus Dictionary nur max. 1x enthalten in relation_list
    
    relation_set = set()
    #iterate through dataProp_dict keys (all substances in additional eln) and 
    # add all keys as dataProperty
    for i in list(dataProp_dict.keys()):
        relation_set.update(set(dataProp_dict[i].keys()))
    
    
    # Important for adding protein and reactant parameters to ontology that are only contained 
    # in the EnzymeML Excel Sheet
    enzymeML_subst_parameters = ["organism","sequence","ecnumber"]
    relation_set.update(set(enzymeML_subst_parameters))
        
    # only selecting some of the parameters of the EnzymeML substance description
    #relation_set.update(set(enzymeML_subst_parameters))
    
    for rel in relation_set:
        if rel in enzymeML_subst_parameters:
            onto = datProp_from_str("has_" + rel,onto)
        else:
            onto = datProp_from_str(rel.strip().replace(' ','_'),onto)
        
    return onto
    
def subst_set_relations(enzmldoc, subst_dict, onto,PFD_uuid):
    
    # Important for adding protein parameters to ontology that are only contained 
    # in the EnzymeML Excel Sheet
    enzymeML_subst_parameters = ["organism","sequence","ecnumber"]
    prot_dict = enzmldoc.protein_dict
    #PFD_uuid = "DWSIM_" + str(uuid.uuid4()).replace("-","_")
    
    for class_name in list(subst_dict.keys()):
        #iterate through each key of the substance dictionary (each substance)
        # and extend the respective individual with the data properties            
        onto_class = onto.search_one(iri='*ind_'+class_name)  
        
        if "hasEnzymeML_ID" in subst_dict[class_name].keys():
            if subst_dict[class_name]["hasEnzymeML_ID"] in prot_dict.keys():
                # get the data as dictionary from the respective protein:
                prot_dat = prot_dict[subst_dict[class_name]["hasEnzymeML_ID"]].dict()
    
                for prot_param in enzymeML_subst_parameters:
                    codestring = "{}.{}.append('{}')".format(str(onto_class),"has_" + prot_param,str(prot_dat[prot_param]))
                    code = compile(codestring, "<string>","exec")
                    exec(code)
                
        for entry in subst_dict[class_name]: 
            data_prop_type = type(subst_dict[class_name][entry])
            
            if entry == "has_role":
                # assert with has role relationship
                codestring = """with onto:
                    role_indv = onto.search_one(label='{}')('{}')
                    {}.RO_0000087.append(role_indv)

                    pfd_indv = onto.search_one(iri ="*{}")
                    {}.BFO_0000050.append(pfd_indv)
                    """.format(subst_dict[class_name][entry],subst_dict[class_name][entry], str(onto_class), PFD_uuid, str(onto_class))
            
            else:    
                # Assert value directly, if entry is int or float
                # give the entry as string else
                if (data_prop_type == int) or (data_prop_type == float):
                    codestring = "{}.{}.append({})".format(str(onto_class),str(entry), subst_dict[class_name][entry])
                else:
                    codestring = "{}.{}.append('{}')".format(str(onto_class),str(entry), str(subst_dict[class_name][entry]))                
            
            """
            elif entry == "kineticDescription":
                # add reaction(s)
                for reac_ID in entry.replace(" ","").split(","):
                    onto = reactions_to_KG(enzmldoc , reac_ID, onto , PFD_uuid)
            """
            
            
            #print(codestring)
            code = compile(codestring, "<string>","exec")

            exec(code)       
        
    return onto

def kin_ind_from_dict(eln_dict, onto):
    
    kin_dict = eln_dict["kinetics"]
    for kin in list(kin_dict.keys()):
        # kin = label of indv
        kin_indv_uuid = "Kin_" + str(uuid.uuid4()).replace("-","_")
        ## adding rateLaw individual
        kin_type = kin_dict[kin]["rateLaw"] # ontology class
        
        #kin_onto_class = onto.search_one(label = kin_type)
        
        ## Enzyme
        # rateLaw -- characteristic of -> Enzyme
        # indv_rateLaw -- http://purl.obolibrary.org/obo/RO_0000052 -> subst_Enzyme
        Enzyme_name = kin_dict[kin]["kineticOfCompound"].strip().replace(' ','_')             
        subst_id = eln_dict["substances"][Enzyme_name]["hasEnzymeML_ID"]       
        Enz_indv_label = "Sub_" + Enzyme_name + "_" + subst_id
    
        ## Substrate 
        # Might also be more than one substrate, if later there are other kinetics, 
        # this could then be reused
        Substrates = kin_dict[kin]["baseCompound"]
        substrate_indv_label = []
        for i in Substrates.split(","):
            try:
                substrate_indv_label.append("Sub_" + i.strip().replace(' ','_') + "_" + eln_dict["substances"][i.strip().replace(' ','_')]["hasEnzymeML_ID"])
            except:
                try:
                    found = eln_dict["substances"][i.strip().replace(' ','_')]
                    if found:
                        substrate_indv_label.append("Sub_" + i.strip().replace(' ','_') + "_")
                except:
                    print("baseCompound {} in kinetic of {} not found in elndict. EnzymeML_ID missing or comma in baseCompound-Name?".format(i,kin))
                    pass
        
        #print(substrate_indv_label)
        
        #include kinetic type as individual for further relations
        # RO_0000052 = characteristic of -> used to assign kinetic rate law to enzyme
        # RO_0002233 = has input -> Used for input in kinetics ; Substrates are input of reaction      
        if onto.search_one(label = kin_type):
            codestring = """with onto:
                            kin_indv = onto.search_one(label = "{}")('indv_{}')
                            kin_indv.label = "indv_{}"
                            
                            enzyme_indv = onto.search_one(label = "{}")
                            kin_indv.RO_0000052 = enzyme_indv
                            
                """.format(kin_type, kin_indv_uuid, kin, Enz_indv_label)
            
            # adding substrates
            for substrate in substrate_indv_label:
                substr = """\n
                            substrate_indv = onto.search_one(label = "{}")
                            kin_indv.RO_0002233.append(substrate_indv)
                     """.format(substrate)
                codestring = codestring + substr 
                
            
        # include as individual, if part in IRI is already present
        elif onto.search_one(iri = "*{}".format(kin_type)):
            codestring = """with onto:
                            kin_indv = onto.search_one(iri = "*{}")('{}')
                            kin_indv.label = "indv_{}"
                            
                            enzyme_indv = onto.search_one(label = "{}")
                            kin_indv.RO_0000052 = enzyme_indv
                """.format(kin_type, kin_indv_uuid, kin, Enz_indv_label)
            
            # adding substrates
            for substrate in substrate_indv_label:
                substr = """\n
                            substrate_indv = onto.search_one(label = "{}")
                            kin_indv.RO_0002233.append(substrate_indv)
                     """.format(substrate)
                codestring = codestring + substr 

        else:            
            # if not contained in ontology, the kinetics are introduced as subclass of
            # SBO_0000001 (rate law)
            codestring = """with onto:
                        class {}(onto.search_one(iri = '*SBO_0000001')):
                            label = '{}'
                            pass                    
                        kin_indv = {}('{}')
                        kin_indv.label = 'indv_{}'
                        
                        enzyme_indv = onto.search_one(label = "{}")
                        kin_indv.RO_0000052 = enzyme_indv
                """.format(kin_type,kin_type,kin_type,kin_indv_uuid,kin, Enz_indv_label)
            
            # adding substrates
            for substrate in substrate_indv_label:
                substr = """\n
                            substrate_indv = onto.search_one(label = "{}")
                            kin_indv.RO_0002233.append(substrate_indv)
                     """.format(substrate)
                codestring = codestring + substr 
                
        #print(codestring)
        code = compile(codestring, "<string>","exec")
        exec(code)
        
        ## add kinetic equation
        onto = datProp_from_str("has_equation", onto)
        kin_indv = onto.search_one(iri = "*"+kin_indv_uuid)
        kin_indv.has_equation.append(str(kin_dict[kin]["has_equation"]))
        ##
        if kin_type == "Arrhenius equation":
            if "A" in kin_dict[kin]:
                
                ind_name = "A_" + kin_dict[kin]["hasEnzymeML_ID"]
                val = kin_dict[kin]["A"]
                unit = kin_dict[kin]["A_Unit"]
                
                hasVal = onto.search_one(iri = '*hasValue')
                #hasModel= onto.search_one(iri = '*RO_0002615')
                #kin_indv = onto.search_one(label = kin)
                # https://w3id.org/nfdi4cat#pre-exponential_factor
                kin_indv_label =  "indv_"+kin
                
                codestring = """with onto:
                                A_indv = onto.search_one(iri = "*pre-exponential_factor")('{}')
                                A_indv.label = "A"
                                A_indv.{}.append('{}')
                                A_indv.has_unit_string.append('{}')
                                
                                kin_indv = onto.search_one(label = kin_indv_label)
                                kin_indv.hasVariable.append(A_indv)
                    """.format(ind_name, hasVal.name, val, unit)
                    
                    
                #print(codestring)
                code = compile(codestring, "<string>","exec")
                exec(code)
                ## https://w3id.org/nfdi4cat#activation_energy
                if "E" in kin_dict[kin]:
                    
                    ind_name = "E_" + kin_dict[kin]["hasEnzymeML_ID"]
                    val = kin_dict[kin]["E"]
                    unit = kin_dict[kin]["E_Unit"]
                    
                    kin_indv_label =  "indv_"+kin
                    
                    codestring = """with onto:
                                    E_indv = onto.search_one(iri = "*activation_energy")('{}')
                                    E_indv.label = "E"
                                    E_indv.{}.append('{}')
                                    E_indv.has_unit_string.append('{}')
                                    
                                    kin_indv = onto.search_one(label = kin_indv_label)
                                    kin_indv.hasVariable.append(E_indv)
                        """.format(ind_name, hasVal.name, val, unit)
                    code = compile(codestring, "<string>","exec")
                    exec(code)    
                    
                    
        ##
        elif kin_type == "Henri-Michaelis-Menten rate law":
            ## adding Km indv if it is contained
            if "Km" in kin_dict[kin]:
                
                ind_name = "Km_" + kin_dict[kin]["hasEnzymeML_ID"]
                val = kin_dict[kin]["Km"]
                unit = kin_dict[kin]["Km_Unit"]
                
                hasVal = onto.search_one(iri = '*hasValue')
                #hasModel= onto.search_one(iri = '*RO_0002615')
                #kin_indv = onto.search_one(label = kin)
                kin_indv_label =  "indv_"+kin
                
                codestring = """with onto:
                                Km_indv = onto.search_one(iri = "*SBO_0000027")('{}')
                                Km_indv.label = "Km"
                                Km_indv.{}.append('{}')
                                Km_indv.has_unit_string.append('{}')
                                
                                kin_indv = onto.search_one(label = kin_indv_label)
                                kin_indv.hasVariable.append(Km_indv)
                    """.format(ind_name, hasVal.name, val, unit)
                    
                    
                #print(codestring)
                code = compile(codestring, "<string>","exec")
                exec(code)
            ## adding kcat indv if it is contained
            if "kcat" in kin_dict[kin]:
                
                ind_name = "kcat_" + kin_dict[kin]["hasEnzymeML_ID"]
                val = kin_dict[kin]["kcat"]
                unit = kin_dict[kin]["kcat_Unit"]
                
                kin_indv_label =  "indv_"+kin
                
                codestring = """with onto:
                                kcat_indv = onto.search_one(iri = "*SBO_0000025")('{}')
                                kcat_indv.label = "kcat"
                                kcat_indv.{}.append('{}')
                                kcat_indv.has_unit_string.append('{}')
                                
                                kin_indv = onto.search_one(label = kin_indv_label)
                                kin_indv.hasVariable.append(kcat_indv)
                    """.format(ind_name, hasVal.name, val, unit)
                code = compile(codestring, "<string>","exec")
                exec(code)           
                
    return onto

def process_to_KG_from_dict(enzmldoc, eln_dict, onto, PFD_uuid):
    # includes all elements of process flow diagram asserted in additional ELN
    # into the ontology as subclass of ontochem:PhysChemProcessingModule 
    # subclass determined by "DWSIM-object type" entry in additional ELN.
    # ontochem is the ontology based on the nfdi4cat-extension of metadata4ing
    # connection of components via "has_input" and "has_output" object properties
    """
    1. Adds process modules as classes based on their dict-entry 
	   "DWSIM-object type" as subclass of http://www.nfdi.org/nfdi4cat/ontochem#PhysChemProcessingModule
    2. Adds process modules as individual of their respective 
	   classes based on their dict-key
    3. Adds relation process_module_indv -- has_output -> process_module_indv 
	   for each dict-entry "connection" (has output: http://purl.obolibrary.org/obo/RO_0002234)
    4. Searches for Substance names in subdicts of process modules
    -> "EntersAtObject" determines individual of the PFD, 
	    where the substance enters
    --> introduces <process_module_indv + "_" +  Substance_name> as 
	    individual -- part of -> process_module_indv (part of = http://purl.obolibrary.org/obo/BFO_0000050)
    --> With hasEnzymeML_ID and key of dict -> ind -- composed primarily of -> subst_ind (composed primarily of = http://purl.obolibrary.org/obo/RO_0002473)
    -> includes all other information as dataProperty  (see 5.)
    5. includes other information as dataProperty to the individuals
	   Iterates eln_dict["PFD"] and add all missing dataProperties 
	   of the dict to the ontology
	-> excludes "DWSIM-object type", "connection", "EntersAtObject"
	-> excludes all individuals/first level keys
    """
    
    PFD_dict = eln_dict["PFD"]
    subst_list = list(eln_dict["substances"].keys())
    omit_list = ["DWSIM-object type", "DWSIM-object argument", "connection", "EntersAtObject", "isDWSIMObject", "hasEnzymeML_ID"]
        
    
    uuid_dict = {}
    ##
    # Add process modules as classes based on their dict-entry "DWSIM-object type" and add respective individual
    for proc_mod in list(PFD_dict.keys()):
        onto_class_name = PFD_dict[proc_mod]["DWSIM-object type"].strip()
        
        uuid_dict[proc_mod] = "PFD_" + str(uuid.uuid4()).replace("-","_")
        
        # onto_class = eln_dict["PFD"][proc_mod]["DWSIM-object type"]
        # indv = proc_mod
        
        # introduce DWSIM-object type as new class, if not already contained in ontology
        if onto.search_one(label = onto_class_name):
            codestring = """with onto:
                                proc_indv = onto.search_one(label = "{}")('indv_{}')
                                proc_indv.label = 'indv_{}'
                                
                                PFD_indv = onto.search_one(iri= "*{}")
                                proc_indv.BFO_0000050.append(PFD_indv)
             """.format(onto_class_name,uuid_dict[proc_mod],proc_mod,PFD_uuid) 
        else:
            codestring = """with onto:
                                    class {}(onto.search_one(iri = '*PhysChemProcessingModule')):
                                        label = '{}'
                                        comment = "Physical/Chemical processing module represented in a flowsheet of the process simulator DWSIM"
                                        pass                    
                                    proc_indv = {}('indv_{}')
                                    proc_indv.label = 'indv_{}'
                                    
                                    PFD_indv = onto.search_one(iri= "*{}")
                                    proc_indv.BFO_0000050.append(PFD_indv)
             """.format(onto_class_name,onto_class_name,onto_class_name,uuid_dict[proc_mod],proc_mod,PFD_uuid) 
        
        #print(codestring) 
        code = compile(codestring, "<string>","exec")
        exec(code)
    ##
    
    ##        
    # Connect the process module individuals based on their dict-entry "connection"
    # with the relation "has output" RO_0002234
    for proc_mod in list(PFD_dict.keys()):
        # check, if there are any process modules connected to the current selected one
        if type(PFD_dict[proc_mod]["connection"]) == str and str(PFD_dict[proc_mod]["connection"]).strip():
            proc_indv_name = uuid_dict[proc_mod]
            connected_indv_name = uuid_dict[PFD_dict[proc_mod]["connection"].strip()]
            codestring = """with onto:
                proc_indv = onto.search_one(iri = "*{}")
                con_proc_indv = onto.search_one(iri = "*{}")
                
                proc_indv.RO_0002234.append(con_proc_indv)

            """.format(proc_indv_name, connected_indv_name)
            
            #print(codestring)
            code = compile(codestring, "<string>", "exec")
            exec(code)
    ##
    
    ##
    # Create DataProperty if not already contained in ontology
    for proc_mod in PFD_dict:
        for prop_key in list(PFD_dict[proc_mod].keys()):
            if prop_key not in omit_list:            
                if prop_key in subst_list:
                    # This triggers connection of the respective process module with the respective substance
                    # Mostly important for material streams
                    # prop_key is a substance, thus needs to be linked to its individual
                    try: 
                        enz_id = eln_dict["substances"][prop_key]["hasEnzymeML_ID"]
                    except:
                        enz_id = ''
                    
                    combined_ind_uuid = "PFD_" + str(uuid.uuid4()).replace("-","_")
                    combined_ind_name = proc_mod + '_' + prop_key
                    
                    # Add dataProperties of subdictionaries, mostly containing material streams of the substances
                    for key in list(PFD_dict[proc_mod][prop_key].keys()):
                        onto = datProp_from_str(key.strip().replace(' ','_'), onto)
                    
                    #TODO: Alex
                    # Add individual for each proc+substance and connect it to individuals                    
                    codestring = """with onto:
                        proc_indv = onto.search_one(iri = "*{}") 
                        subst_indv = onto.search_one(label = "Sub_{}_{}") 
                        
                        proc_subst_indv = onto.search_one(label = proc_indv.is_a[0].label)('{}')
                        proc_subst_indv.label = "{}"
                        
                        proc_indv.BFO_0000051.append(proc_subst_indv)
                        proc_subst_indv.RO_0002473.append(subst_indv)                           
                        proc_subst_indv.BFO_0000050.append(proc_indv)
                        
                        """.format(uuid_dict[proc_mod],prop_key,enz_id,combined_ind_uuid,combined_ind_name)
                    
                    # add data properties for newly created individual
                    for key in list(PFD_dict[proc_mod][prop_key].keys()):
                        val = PFD_dict[proc_mod][prop_key][key]
                        if (val == int) or (val == float):
                            dataPropstring = """\nproc_subst_indv.{}.append(float({}))""".format(key,val)  
                        else:
                            dataPropstring = """\nproc_subst_indv.{}.append('{}')""".format(key,val)  
  
                        codestring = codestring + dataPropstring
                    #print(codestring)
                else:
                    # No Substance name -> Direct dataProperty assertion
                    onto = datProp_from_str(prop_key.strip().replace(' ','_'),onto)
                    val = PFD_dict[proc_mod][prop_key]
                    if (val == int) or (val == float):
                        codestring = """with onto:
                            proc_indv = onto.search_one(iri = "*{}") 
                            proc_indv.{}.append(float({}))
                            """.format(uuid_dict[proc_mod], prop_key.strip().replace(' ','_'), val)

                    else:
                        codestring = """with onto:
                            proc_indv = onto.search_one(iri = "*{}") 
                            proc_indv.{}.append('{}')
                            """.format(uuid_dict[proc_mod], prop_key.strip().replace(' ','_'), val)
                    
                code = compile(codestring, "<string>", "exec")
                exec(code)    
    return onto

###
def reactions_to_KG(enzmldoc,supp_eln_dict,onto,PFD_uuid):
    
    # add properties of reaction to individual
    # add educts, products, ... subdicts -> based on assigned ontology class
    #onto -> add enzmldoc.reaction_dict[reac_ID]["ontology"]
    # 
    #Get individual of current sheet
    pfd_indv = onto.search_one(iri = "*"+PFD_uuid)
    
    #get all substances of sheet, that have an EnzymeML-ID
    #subst_dict = {EnzymeML-ID: onto.individual, ...}
    subst_dict = {} 
    for indv in pfd_indv.BFO_0000051:
        if indv.hasEnzymeML_ID: 
            subst_dict[indv.hasEnzymeML_ID.first()] = indv 
    #
    #print("\n keys of subst_dict: \n {}".format(list(subst_dict.keys())))
    
    for reac_ID in list(enzmldoc.reaction_dict.keys()):
        reac_obj = enzmldoc.getAny(reac_ID)
        reaction_class = reac_obj.ontology.value.replace(":","_")
        RCT_uuid = "RCT_" + str(uuid.uuid4()).replace("-","_")
        try:
            #with onto:
            #add individual of reaction class to ontology
            rct_indv = onto.search_one(iri ="*"+reaction_class)(RCT_uuid)
            rct_indv.label = reac_obj.name + "_" + reac_ID
            rct_indv.BFO_0000050.append(pfd_indv)
            #Add all other properties of the enzmldoc, but the entries that contain an "ontology" entry   
        except:
            print(reaction_class+" - class not found in ontology while implementing reaction" + reac_ID +" in ontology!")
            
        ## ALEX
        for entry in reac_obj.dict():
            if entry == "educts":
                for i in reac_obj.dict()[entry]:
                    enz_id = i["species_id"]
                    if enz_id in subst_dict.keys():
                        rct_indv.RO_0002233.append(subst_dict[enz_id]) # has input
                
            elif entry == "products":
                for i in reac_obj.dict()[entry]:
                    enz_id = i["species_id"]
                    if enz_id in subst_dict.keys():
                        rct_indv.RO_0002234.append(subst_dict[enz_id]) # has output
            
            elif entry == "modifiers":
                for i in reac_obj.dict()[entry]:
                    enz_id = i["species_id"]
                    if enz_id in subst_dict.keys():
                        rct_indv.RO_0002573.append(subst_dict[enz_id]) # has modifier          
                        kin_indv = subst_dict[enz_id].RO_0000053.first() # has characteristic
                        
                        if subst_dict[enz_id].kineticDescription.first() == reac_ID:
                            #rct_indv --has model -> kin_indv
                            rct_indv.RO_0002615.append(kin_indv) # has model

                        
            else:
                ## add to individual via dataProperty
                if entry not in ["name","ontology"]:
                    onto = datProp_from_str(entry.strip().replace(' ','_'), onto)
                    if reac_obj.dict()[entry]:
                        if type(reac_obj.dict()[entry]) in [float, int]:
                            codestr = """rct_indv.{}.append({})""".format(entry,reac_obj.dict()[entry])
                        else:
                            codestr = """rct_indv.{}.append('{}')""".format(entry,reac_obj.dict()[entry])
                
                        #print(codestr)
                        code = compile(codestr,"<string>","exec")
                        exec(code)
            # else:
                ## add to individual via dataProperty
            
        
    return onto


##

def eln_to_knowledge_graph(enzmldoc, supp_eln_dict, onto, extended_ontology_path):

    ##
    #SBO Term: enzmldoc.getAny("s0").ontology.value
    
    ## include substances in ontology
    # insert substances from dictionary in ontology
    onto = subst_classes_from_dict(enzmldoc, supp_eln_dict["substances"], onto)
    
    PFD_name = enzmldoc.name
    PFD_uuid = "Experiment_" + str(uuid.uuid4()).replace("-","_")
    
    creator_str = ""
    for key in enzmldoc.creator_dict:
        creator_str = creator_str + str(enzmldoc.creator_dict[key].dict()).strip("{").strip("}").replace("'","") + "\n"        
    
    codestring = """with onto:
        PFD_indv = onto.search_one(iri = '*DataProcessingModule')("{}")
        PFD_indv.label = 'Experiment_{}'
        PFD_indv.comment = 'Laboratory experiments and corresponding process flow diagram of {}'
        PFD_indv.comment =""".format(PFD_uuid,PFD_name,PFD_name)
    
    codestring = codestring + '""" Creator(s): \n' + creator_str + '"""'
    
    #print(codestring)
    code = compile(codestring, "<string>", "exec")
    exec(code)
    
    # insert data properties to substance individuals from dictionary
    onto = datProp_from_dict(supp_eln_dict["substances"], onto)

    # insert data properties to substance individuals from dictionary
    onto = subst_set_relations(enzmldoc, supp_eln_dict["substances"], onto, PFD_uuid)
    
    ## include kinetics in ontology        
    onto = kin_ind_from_dict(supp_eln_dict,onto)
    
    ## include Process Flow Diagram in ontology    
    onto = process_to_KG_from_dict(enzmldoc, supp_eln_dict,onto, PFD_uuid)
    
    ## include reactions in ontology
    onto = reactions_to_KG(enzmldoc,supp_eln_dict,onto,PFD_uuid)
    
    # save ontology
    onto.save(file=extended_ontology_path, format="rdfxml")
    
    
    return PFD_uuid



def run(enzml_XLSX_path,pfd_XLSX_path, base_ontology_path, extended_ontology_path):

   enzmldoc = pe.EnzymeMLDocument.fromTemplate(enzml_XLSX_path)
   new_eln_dict = new_ELN_to_dict(pfd_XLSX_path)
   
   onto = base_ontology_extension(base_ontology_path)

   PFD_uuid = eln_to_knowledge_graph(enzmldoc, new_eln_dict, onto, extended_ontology_path)
   
   return PFD_uuid

def eln_to_dict(enzymeML_ELN_path,process_ELN_path):
    enzmldoc = pe.EnzymeMLDocument.fromTemplate(enzymeML_ELN_path)
    enzdict = enzmldoc.dict()
    eln_dict = new_ELN_to_dict(process_ELN_path)
    return enzdict, eln_dict

#TODO: implement UUIDs for substances and also for reaction/kinetic individuals
#TODO: Link to simulation-files and to ELN files via comment/IRI!
    
    