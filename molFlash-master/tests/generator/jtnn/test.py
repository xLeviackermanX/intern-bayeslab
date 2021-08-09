from molflash.generator.G2G.data import JTNNDataModule
from molflash.models.jtnn.jtnn_vae import JTNNVAE
from molflash.generator.G2G.jtnn_autoencoder import JTNNTask
import flash

file = "/home/trinity/github-repos/new/molFlash/tests/dataset/train_15L.csv"

dm = JTNNDataModule(file)
print(dm.vocab.size())
model = JTNNTask(dm.vocab,450,32,20,3,JTNNVAE)
trainer = flash.Trainer(max_epochs=200, gpus=1, fast_dev_run=False)
trainer.fit(model, dm)