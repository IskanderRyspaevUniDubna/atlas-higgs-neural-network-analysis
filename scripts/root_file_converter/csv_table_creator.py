import os
import uproot
import pandas as pd


def read_txt_config(config_file_name, config_file_path):
    lines = []
    with open(str(config_file_path)+'/'+str(config_file_name)) as file:
        for line in file:
            lines.append(line.strip())
    return lines

dir_path = (
    '/lustre/home/user/r/ryspaev/HOME/neural_network_analysis/IskanderRyspaevDiploma2025/'
    'atlas-higgs-neural-network-analysis/data/cuted_with_weight_event'
)

files = [f for f in os.listdir(dir_path) 
         if os.path.isfile(os.path.join(dir_path, f))
        ]

var_list = read_txt_config(
        'var_list.txt',
        (
            '/lustre/home/user/r/ryspaev/HOME/neural_network_analysis/IskanderRyspaevDiploma2025/'
            'atlas-higgs-neural-network-analysis/data/config_files'
        )
    )

var_list.append('Weight_Event')

var_list.append('class_label')

combined_table = []
for file_name in files:
    nom_loose = "tree;1"
    root_file = uproot.open(
        str(dir_path)+"/"+str(file_name)+":"+str(nom_loose)
    )
    csv_table = root_file.arrays(var_list, library="pd")
    combined_table.append(csv_table)

result = pd.concat(combined_table, ignore_index=True)
result.to_csv(
    (
        '/lustre/home/user/r/ryspaev/HOME/neural_network_analysis/IskanderRyspaevDiploma2025/'
        'atlas-higgs-neural-network-analysis/data/output_table/'
    )
    + 'converted_root_data.csv', index=False
)