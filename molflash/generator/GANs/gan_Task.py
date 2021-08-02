#flash imports
#other pytorch rdkit imports

class GAN(flash.Task):
    """
    does pretained prediction and calls finetuning for property filtration and then model tuning.
    """
    def __init__(self, vocabulary, config):
        """

        :param vocabulary: initialize atom vocabulary
        :param config: model latent dimensions, encoder, its model configs(batches,hidden size etc..)
        """

    def forward(self, n_batch):
        """

        :param n_batch:
        :return:
        """
        out = self.sample(n_batch)
        return out

    def encode_smiles(self, smiles_in, encoder=None):
        """

        :param smiles_in: input smiles
        :param encoder: load the encoder selected
        :return: latent vector
        """

        model = load_model(model_version=encoder)

        #call preprocess before sending to encoder


    def compute_gradient_penalty(self, real_samples,
                                 fake_samples, discriminator):
        """Calculates the gradient penalty loss for GAN """
        return gradient_penalty


    def sample_pretrainprediction(self, n_batch, max_length=100):
        """
        Run the pretrained Generator loaded
        :param n_batch:
        :param max_length:
        :return:
        """
        self.decoder=load_model(filename)
        self.Gen.eval()

        """
        #do property filtration and put for training"""
        gan_finetune(new_smiles)



def load_model(model_version=None,modelpath):
    """

    :param model_version: if required
    :param modelpath: path of the encoder.py
    :return:
    """

    # Import model
    model = load(model_name=path)
    return model





class CustomDatamoudule(DataModule):

    def get_data(self,args):
        """
        :return: csv file after reading
        """
        return pd.read_csv(args.datafile)



    def clean_data(self,data):
        """
        cleaning like - getting unique rationales, removing invalid smiles
        :return: cleaned dataset rationales
        """

        rationales = data['rationales']
        rationales = PreprocessingFunc.unique_rationales(rationales)
        rationales = PreprocessingFunc.rem_invalidsmiles(rationales)
        return rationales




    def from_dataset(self,configparams):
        """
        create data format as required for gans and encoder based
        :return:
        """
        data=get_data(configparams.file)
        rationales=clean_data(data)

        return rationale_dataset


if __name__=="__main__":
    #config params
