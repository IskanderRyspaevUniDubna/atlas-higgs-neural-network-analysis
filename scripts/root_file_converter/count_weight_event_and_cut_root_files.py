import os
import numpy as np
import pandas as pd
import uproot

def read_txt_config(config_file_name, config_file_path):
    lines = []
    with open(str(config_file_path)+'/'+str(config_file_name)) as file:
        for line in file:
            lines.append(line.strip())
    return lines

directory_path = (
    '/lustre/home/user/r/ryspaev/HOME/neural_network_analysis/IskanderRyspaevDiploma2025/'
    'atlas-higgs-neural-network-analysis/data/oficialMC_v34_minintuples_v1'
)

files = [f for f in os.listdir(directory_path) 
         if os.path.isfile(os.path.join(directory_path, f))
        ]

var_list = read_txt_config(
        'var_list.txt',
        (
            '/lustre/home/user/r/ryspaev/HOME/neural_network_analysis/IskanderRyspaevDiploma2025/'
            'atlas-higgs-neural-network-analysis/data/config_files'
        )
    )

for file_name in files:
    nom_loose = 'nominal_Loose;1'
    root_file = uproot.open(
        str(directory_path) + '/' + str(file_name) + ':' + str(nom_loose)
    )
    
    Weight_Event = (np.array(root_file['weight_mc']) * 
                    np.array(root_file['xsec_weight']) * 
                    np.array(root_file['weight_jvt']) * 
                    np.array(root_file['weight_pileup']) * 
                    np.array(root_file['weight_leptonSF']) * 
                    np.array(root_file['weight_bTagSF_DL1r_Continuous'])
                   ) / np.array(root_file['totalEventsWeighted'])

    run_number = np.array(root_file['runNumber'])
    for i in range(len(run_number)):
        if(run_number[i]<290000):
            Weight_Event[i] = 36207.66 * Weight_Event[i]
        elif((run_number[i]>=290000) and (run_number[i]<310000)):
            Weight_Event[i] = 44307.4 * Weight_Event[i]
        elif(run_number[i]>=310000):
            Weight_Event[i] = 58450.1 * Weight_Event[i]        
    
    data = {}
    data['Weight_Event'] = np.array(Weight_Event)
    for var in var_list:
        data[var] = np.array(root_file[var])
    if(file_name == '346676_tH_Selected.root'):
        data['class_label'] = np.ones(len(run_number))
    else:
        data['class_label'] = np.zeros(len(run_number))
    with uproot.recreate(
        (
            '/lustre/home/user/r/ryspaev/HOME/neural_network_analysis/IskanderRyspaevDiploma2025/'
            'atlas-higgs-neural-network-analysis/data/cuted_with_weight_event/'
        ) + file_name.replace('.root','') + '_cuted_with_Weight_Event' + '.root') as file:
        file['tree'] = data
    print(str(file_name) + str(" : ") + str(np.sum(np.array(Weight_Event))))