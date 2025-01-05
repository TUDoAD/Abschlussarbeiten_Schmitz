# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 00:43:50 2024

@author: n.sain
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os
import sys
from openpyxl import load_workbook
import win32com.client


xlapp = win32com.client.Dispatch("Excel.Application")
Ea_reference = 48.00 # KJ/mol
K0_reference = 2.86e+07 #L/mol.s
Tolerance = 50 # maximum error defined 

# File path 
file_path = os.path.join(os.getcwd(),'ELNs', 'New-ELN_Kinetik_1_Sain_AB.xlsx')
sheet_0 = 'Data_Input'
save_dir   = os.path.join(os.getcwd(),'Plots')


#Checking if file exists
if not os.path.exists(file_path):
    print('file does not exist')
    sys.exit(0)
    
#Crrating the save directory
if not os.path.exists(save_dir):
    os.makedirs(name=save_dir,exist_ok=True)
    

# Read the data from the Excel file
df = pd.read_excel(file_path, sheet_name=sheet_0)
# Clean up the column names by stripping extra spaces and handling typos
df.columns = df.columns.str.strip()

#%% Defining Functions

# Calculate_ea_k0 function
def calculate_ea_k0(campaign_data):
    x = campaign_data['1/T']
    y = campaign_data['lnK']
    
    # linear regression (y = mx + c)
    if x.empty or y.empty:
        raise ValueError("Inputs for regression cannot be empty")
    
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    # Ea (activation energy) in kJ/mol
    R = 8.314  # J/(mol*K)
    Ea = -slope * R / 1000  #  J to kJ
    
    # K0 (pre-exponential factor) using exp(intercept)
    K0 = np.exp(intercept)
    
    return Ea, K0, slope, intercept

# Function to plot the data and display the equation on the plot
def plot_lnK_vs_1_T(campaign_data, campaign_id, slope, intercept):
    x = campaign_data['1/T']
    y = campaign_data['lnK']
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, label=f"Campaign {campaign_id}", color='blue')
    
    # Plot the best fit line
    plt.plot(x, slope * x + intercept, color='red', label='Best Fit Line')
    
    # Add equation to the plot
   
    equation_text = f"y = {slope:.4f}x + {intercept:.4f}"
    
    
    x_text_position = (x.max() + x.min()) / 2  # Middle of the x-range
    y_text_position = y.max() - 0.1 * (y.max() - y.min())  # Slightly below y-max
    
    # Display the equation on the plot at the calculated position
    plt.text(x_text_position, y_text_position, equation_text, fontsize=16, color='red', 
             bbox=dict(facecolor='white', alpha=0.5))
    
    # Plot formatting
    plt.title(f"lnK vs 1/T for Campaign {campaign_id}")
    plt.xlabel("1/T (1/K)")
    plt.ylabel("lnK")
    plt.legend()
    plt.grid(True)
   
    
    #Save plot to the directory
    plot_title = f"lnK_vs_1_T_Campaign_{campaign_id}"
    file_path_pdf = os.path.join(save_dir, f"{plot_title}.pdf")
    file_path_svg= os.path.join(save_dir,f"{plot_title}.svg")
    
    plt.savefig(file_path_pdf,format='pdf')
    plt.savefig(file_path_svg,format='svg')
    
    plt.show()
#%%

#Add calculated values to new data frame
campaign_df=pd.DataFrame([], columns=['Campaign_id','Calculated_Ea','Calculated_K0'])

# Loop through each campaign and plot lnK vs 1/T
campaign_ids = df['Campaign_ID'].unique()  
for idx, campaign_id in enumerate(campaign_ids):
    campaign_data = df[df['Campaign_ID'] == campaign_id]
    
    # Calculate Ea and K0 for the current campaign
    Ea, K0, slope, intercept = calculate_ea_k0(campaign_data)
    
    # Plot lnK vs 1/T for the current campaign and show the equation
    plot_lnK_vs_1_T(campaign_data, campaign_id, slope, intercept)
    
    # Print Ea and K0
    print(f"Campaign {campaign_id}: Ea = {Ea:.2f} kJ/mol, K0 = {K0:.2e}")
    campaign_df.loc[idx]=[campaign_id,Ea,K0]
    
    
#%%

#Erorr Calculation 

def percentage_error(calculated,reference):
    return abs ((calculated-reference)/ reference)*100

#Calcualte the percentage error for Ea and K0
campaign_df['Ea_error_%']=campaign_df['Calculated_Ea'].apply(lambda x:percentage_error(x,Ea_reference))
campaign_df['K0_error_%']=campaign_df['Calculated_K0'].apply(lambda x:percentage_error(x,K0_reference))

#Filter campaigns where both Ea and K0 less than 5%
filtered_df=campaign_df[(campaign_df['Ea_error_%']< Tolerance) & (campaign_df['K0_error_%'] < Tolerance)]

#Display campaigns that satisfy the condition 
best_campaigns = filtered_df['Campaign_id'].tolist()

print(f"Best Campaign(s) with erorr <5% : {best_campaigns}")
print(filtered_df)

#%%



finished = False

while not finished:
    try:
        #Returning value to excel file 
        wb = xlapp.workbooks.open(file_path)
        sheet_1 = 'Substances and Reactions'
        sheet = wb.Sheets(sheet_1)
        
        for i in range(len(filtered_df)):
            Ea_selected = filtered_df['Calculated_Ea'].iloc[i]
            K0_selected = filtered_df['Calculated_K0'].iloc[i]
            
            cell_address_1 = f"B{18+i}"  
            cell_address_2 = f"B{16+i}"  
            
           
            sheet.Range(cell_address_1).Value = Ea_selected  
            sheet.Range(cell_address_2).Value = K0_selected  
          
        wb.RefreshAll()
        wb.Save()
        wb.Close()
        xlapp.Quit()
        finished = True
    
    except:

        print('Trying to save. !!!')
    