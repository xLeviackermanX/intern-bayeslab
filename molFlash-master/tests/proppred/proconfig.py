config:
  type: proppred
  modelName: "GCN"
  filePath: "/home/bayeslabs/New_Arch/proppred/bbb.csv"
  loss_fn: nn.BCELoss()
  batch_size: 32
  epochs: 5
  gpus: 0
  logger:
  optimizer: optim.Adam
  preprocessing:
  lr: 0.001
  dropout: 0.1
  split: [0.8,0.1,0.1]

  
