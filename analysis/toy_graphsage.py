import os.path as osp

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torchmetrics import Accuracy

from torch_geometric.data.lightning import LightningNodeData
from torch_geometric.datasets import Reddit
from torch_geometric.nn import GraphSAGE
import torch_geometric.transforms as T

from vqniche.dataloaders.transforms import init_data_transforms, SetExperimentDataKeys
from vqniche.dataloaders.in_memory_dataset_blob import InMemoryDatasetBlob
from vqniche.models.graphsage import GraphSAGE


class Model(pl.LightningModule):
    def __init__(self, in_channels: int, out_channels: int,
                 hidden_channels: int = 256, num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.gnn = GraphSAGE(in_channels, hidden_channels, num_layers,
                             out_channels, dropout=dropout,
                             norm=BatchNorm1d(hidden_channels))

        self.train_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.val_acc = Accuracy(task='multiclass', num_classes=out_channels)
        self.test_acc = Accuracy(task='multiclass', num_classes=out_channels)

    def forward(self, x, edge_index):
        return self.gnn(x, edge_index)

    def training_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index)[:data.batch_size]
        y = data.y[:data.batch_size]
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(y_hat.softmax(dim=-1), y)
        self.log('train_acc', self.train_acc, prog_bar=True, on_step=False,
                 on_epoch=True)
        return loss

    def validation_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index)[:data.batch_size]
        y = data.y[:data.batch_size]
        self.val_acc(y_hat.softmax(dim=-1), y)
        self.log('val_acc', self.val_acc, prog_bar=True, on_step=False,
                 on_epoch=True)

    def test_step(self, data, batch_idx):
        y_hat = self(data.x, data.edge_index)[:data.batch_size]
        y = data.y[:data.batch_size]
        self.test_acc(y_hat.softmax(dim=-1), y)
        self.log('test_acc', self.test_acc, prog_bar=True, on_step=False,
                 on_epoch=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.1)


if __name__ == '__main__':
    # dataset = Reddit(osp.join('data', 'Reddit'))
    # data = dataset[0]
    # Set seed for reproducibility
    seed = 0
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('medium')

    DataKeyTransform = SetExperimentDataKeys(       
                            feature_name="cell_gene_counts",
                            label_name="cell_types",
                            edge_index_name="spatial_delaunay"
                        )

    data_transforms = init_data_transforms(
                            data_transform_names=[
                                    'NormalizeFeatures',
                                    'RandomNodeSplit'],
                            norm_method='read_depth',
                            val_ratio=0.1,
                            test_ratio=0.1
                        )
    transform = T.Compose([DataKeyTransform] + data_transforms)
    dataset = InMemoryDatasetBlob(
                    name="sss2-1b_1p",
                    transform=transform
                )
    print(dataset)

    data = dataset[0]
    print(data)
    
    datamodule = LightningNodeData(
        data,
        loader='neighbor',
        num_neighbors=[25, 10],
        batch_size=1024,
        num_workers=4,
        shuffle=False
    )

    model = Model(dataset.num_node_features, dataset.num_classes)
    # model = GraphSAGE(
    #     name='GraphSAGE',
    #     in_channels=data.num_features,
    #     out_channels=data.num_classes,        
    #     hidden_channels=256,
    #     num_layers=2,
    #     dropout=0.5,
    #     loss_kwargs={'reduction': 'mean'})

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    strategy = pl.strategies.SingleDeviceStrategy(device=device)
    checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_acc', save_top_k=1,
                                              mode='max')
    trainer = pl.Trainer(strategy=strategy, devices=1, max_epochs=20,
                         callbacks=[checkpoint], deterministic=True)

    trainer.fit(model, datamodule)
    trainer.test(ckpt_path='best', datamodule=datamodule)