import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
import json

# Assuming you have your matrices as pandas dataframes
def data(label):
    p = json.load(open("/tmp/tP" + label + ".json", "r"))
    q = json.load(open("/tmp/tQ" + label + ".json", "r"))
    muhat = json.load(open("/tmp/tMuHat" + label + ".json", "r"))
    # Convert dataframes to lists of lists
    return Dataset.from_dict({'p': [p],
                              'q': [q],
                              'couplings': [muhat]})

# Combine datasets into a DatasetDict
# they are really all just functions of muhat with the J_m, J_f projections
dataset_dict = DatasetDict({
    'K': data('Kids'),
    'M': data('Marriage'),
    'KM': data('KM')
})

hf_rwtoken = "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"
dataset_dict.push_to_hub("StefanHubner/DivorceData", token = hf_rwtoken)


# new data

file_path = './240716_0500__RES_07_15_adjusted10s_incldummy.xlsx'

def couplings(i):
    return pd.read_excel(file_path,
                         sheet_name = 'statics_' + str(i),
                         header=[2, 3, 4, 5],
                         index_col=[0, 1, 2, 3, 4]
                        ).fillna(0.0)

def trans(sex, i):
    trans = pd.read_excel(file_path,
                          sheet_name = sex + '_' + str(i),
                          header = [2, 3],
                          index_col = [1, 2, 3, 4, 5, 6, 7])
    trans = trans.drop(trans.columns[0], axis = 1)
    return trans.fillna(0.0)

trans("women", 1).shape
couplings(1).loc[1996].shape
