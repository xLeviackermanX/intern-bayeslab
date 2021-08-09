#mports




class Discriminator(pl.PytorchlightningModule):

    """Initialize the Discriminator"""
    def __init__(self, data_shape=(1, 512)):
        super(Discriminator, self).__init__()
        self.data_shape = data_shape
        
        if model=="gcn":
            self.model = gcn()
        elif model=="lstm":
            self.model=="lstm"

    def forward(self, x):
        validity = self.model(x)
        return validity


class Generator(pl.PytorchlightningModule):
    def __init__(self, data_shape=None, latent_dim=None):
        """
        Initialize the input data shape, model encoder for generator,latent dimension size
        """

        self.model = gcn(modelargs)

    def forward(self, x):
        out = self.model(x)
        return out


