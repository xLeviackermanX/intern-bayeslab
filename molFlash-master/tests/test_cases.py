import json
from molflash.generator.utils.Seq2Seq_preprocess import string_to_int,int_to_string,to_categorical,preprocess_data,input_data,prepareData

data_path = '/home/bayeslabs/moltorch_architecture/mol/Generator/dataset.csv'
smi = '<SRXN> C O c 1 c c 2 c ( c ( Cl ) c 1 O C ) C C N ( C ) C C 2 c 1 c c c c c 1 <ERXN>'

def test_input_data():
    dataset = input_data(data_path)
    assert dataset is not None and type(dataset).__name__ == 'Tensor'


input_vocab = {}
with open('/home/bayeslabs/molFlash/molflash/generator/Seq2Seq/input_vocab.json', 'r') as f:
    input_vocab = json.load(f)
inv_vocab = {v:k for k,v in input_vocab.items()}

def test_string_to_int():
    smi_int = string_to_int(smi.split(' '),50,input_vocab)
    assert smi_int is not None and type(smi_int).__name__ == 'list'

smi_int = string_to_int(smi.split(' '),50,input_vocab)

def test_int_to_string():
    int_smi = int_to_string(smi_int,inv_vocab)
    assert int_smi is not None and type(int_smi).__name__ == 'list'



# test_input_data()
# test_string_to_int()
# test_int_to_string()
prepareData()