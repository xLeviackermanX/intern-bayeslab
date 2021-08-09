class GANTrainer(pl.LightningModule):
    """
    trainer upon the pretrained model
    """

    def __init__(self, config):
        self.config = config
        self.latent_size = self.config.latent_vector_dim
        
        
    def property_filter(self):
        #property prediction and filtered molecules is returned"




    def _train_epoch(self, model, tqdm_data,
                     optimizer_disc=None, optimizer_gen=None):
                         """
                         compute the gradients , loss ,optimize for each epoch
                         """

    def _train(self, model, train_loader, val_loader=None, logger=None):

        """
        setup optimizers, schedules
        :param model: pretrained model
        :param train_loader:
        :param val_loader:
        :param logger:
        :return: model save checkpoint
        """

        device = model.device

        for epoch in range(self.config.train_epochs):
            postfix = self._train_epoch(model, tqdm_data,
                                        optimizer_disc, optimizer_gen)
            if val_loader:

                torch.save(
                    model.state_dict(),
                    self.config.model_save[:-3] + '_{0:03d}.pt'.format(epoch)
                )
                model = model.to(device)


    def get_collate_fn(self, model):
        device = self.get_collate_device(model)

        def collate(data):
            tensors = torch.tensor([t for t in data],
                                   dtype=torch.float64, device=device)
            return tensors

        return collate

    def _get_dataset_info(self, data, name=None):
        df = pd.DataFrame(data)
        maxlen = df.iloc[:, 0].map(len).max()
        ctr = Counter(''.join(df.unstack().values))
        charset = ''
        for c in list(ctr):
            charset += c
        return {"maxlen": maxlen, "charset": charset, "name": name}

    def fit(self,
            model,
            train_data,
            val_data=None):
        from ddc_pub import ddc_v3 as ddc
        self.generator = model.Generator
        self.discriminator = model.Discriminator
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.discriminator.cuda()
            self.generator.cuda()

        logger = Logger() if self.config.log_file is not None else None

        if self.config.heteroencoder_version == 'new':
            # Train the heteroencoder first
            print("Training heteroencoder.")
            currentDirectory = os.getcwd()
            path = #save_new_model_path
            encoder_checkpoint_path = #base checkpoint path

            # Convert all SMILES to binary RDKit mols to be
            #  compatible with the heteroencoder
            heteroencoder_mols = [Chem.rdchem.Mol
                                      .ToBinary(Chem.MolFromSmiles(smiles))
                                  for smiles in train_data]
            
            # Dataset information
            dataset_info = self._get_dataset_info(
                train_data, name="heteroencoder_train_data")
            
            # Initialize heteroencoder with default parameters
            # Train heteroencoder
            encoder_model.fit(modelparams)

            heteroencoder_model.save(path)

        heteroencoder = load_model(
            model_version=self.config.heteroencoder_version)
        print("Training GAN.")
        mols_in = [Chem.rdchem.Mol.ToBinary(
            Chem.MolFromSmiles(smiles)) for smiles in train_data]
        latent_train = heteroencoder.transform(
            heteroencoder.vectorize(mols_in))
        # Now encode the GAN training set to latent vectors

        latent_train = latent_train.reshape(latent_train.shape[0],
                                            self.latent_size)

        if val_data is not None:
            mols_val = [Chem.rdchem.Mol.ToBinary(Chem.MolFromSmiles(smiles))
                        for smiles in val_data]
            latent_val = heteroencoder.transform(
                heteroencoder.vectorize(mols_val))
            latent_val = latent_val.reshape(latent_val.shape[0],
                                            self.latent_size)

        train_loader = self.get_dataloader(model,
                                           LatentMolsDataset(latent_train),
                                           shuffle=True)
        val_loader = None if val_data is None else self.get_dataloader(
            model, LatentMolsDataset(latent_val), shuffle=False
        )

        self._train(model, train_loader, val_loader, logger)
        return model