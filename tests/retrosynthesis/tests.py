import pandas as pd
from molflash.utils.preprocess import PreprocessingFunc

data_path = '/home/bayeslabs/New_Arch/proc_train_singleprod.csv'
smile = "[Cl:1][C:2]1[N:3]=[CH:4][C:5]2[C:10]([CH:11]=1)=[C:9]([N+:12]([O-])=O)[CH:8]=[CH:7][CH:6]=2.O.[OH-].[Na+]"
reaction = "[Cl:1][C:2]1[N:3]=[CH:4][C:5]2[C:10]([CH:11]=1)=[C:9]([N+:12]([O-])=O)[CH:8]=[CH:7][CH:6]=2.O.[OH-].[Na+]>>[Cl:1][C:2]1[N:3]=[CH:4][C:5]2[C:10]([CH:11]=1)=[C:9]([NH2:12])[CH:8]=[CH:7][CH:6]=2"
template = "[NH2;D1;+0:1]-[c:2]>>O=[N+;H0;D3:1](-[O-])-[c:2]"
product = "[Cl:1][C:2]1[N:3]=[CH:4][C:5]2[C:10]([CH:11]=1)=[C:9]([NH2:12])[CH:8]=[CH:7][CH:6]=2"

def test_smileToFPS():
    fps = PreprocessingFunc.smileToFPS(smile)
    print(fps)
    print(type(fps))
    assert fps is not None and type(fps).__name__ == 'list' 

def test_fingerprint_mols():
    mol_fps = PreprocessingFunc.fingerprint_mols(smile, fp_dim=1024)
    print(mol_fps)
    print(type(mol_fps))

def test_fingerprint_reactions():
    rxn_fps = PreprocessingFunc.fingerprint_reactions(reaction, fp_dim=1024)
    print(rxn_fps)

def test_fps_prep():
    fp = PreprocessingFunc.fps_prep((smile,reaction))
    print(fp)
    print(type(fp))

data = ((product,reaction),1)
def test_fps_preprocess():
    data_fps = PreprocessingFunc.fps_preprocess(data)
    print(data_fps)
    print(type(data_fps))
    assert data_fps is not None and type(data_fps).__name__ == 'tuple'


def test_get_labels():
    df = pd.read_csv(data_path, nrows = 10)

    rxns = list(df['rxn_smiles'])
    temps = list(df['retro_templates'])
    labelled_data = PreprocessingFunc.get_labels((rxns,temps))
    print(labelled_data)
    print(type(labelled_data))
    assert labelled_data is not None and type(labelled_data).__name__ == 'tuple'






test_smileToFPS()
test_fingerprint_mols()
test_fingerprint_reactions()
test_fps_prep()
test_fps_preprocess()
test_get_labels()

