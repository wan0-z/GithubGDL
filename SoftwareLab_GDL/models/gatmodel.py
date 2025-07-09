import torch
import torch.nn.functional as F
from torch import nn
import lightning as L
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import networkx as nx
import torch_geometric.utils as pyg_utils
import numpy as np
import matplotlib as mpl

class BRepGATClassifier(L.LightningModule):
    def __init__(self, 
                 in_node_features=14,      # face_attr feature dim
                 in_edge_features=15,      # edge_attr feature dim
                 hidden_dim=640,
                 num_classes=27,           # change to your actual number of classes
                 num_layers=6,
                 dropout=0.3,
                 lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = in_node_features if i == 0 else hidden_dim
            out_dim = hidden_dim if i < num_layers - 1 else num_classes
            apply_relu = i < num_layers - 1

            self.gat_layers.append(
                GATv2Conv(
                    in_channels=(in_dim, in_dim),
                    out_channels=out_dim,
                    edge_dim=in_edge_features,
                    heads=1,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=True
                )
            )
            if apply_relu:
                self.gat_layers.append(nn.ReLU())
        self.dropout = nn.Dropout(p=dropout)
        self.lr = lr

    def forward(self, data):
        x, edge_index, edge_attr = data.face_attr, data.edge_index, data.edge_attr
        for layer in self.gat_layers:
            if isinstance(layer, GATv2Conv):
                x = layer(x, edge_index, edge_attr)
            else:
                x = layer(x)
        return x  # shape: [num_nodes, num_classes]

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        y = batch.cls_label.long()  # 也要加 long()
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log('train_loss', loss)
        self.log('train_acc', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        y = batch.cls_label.long()  # 也要加 long()
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        logits = self(batch)
        y = batch.cls_label.long()
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss
    def visualize_attention(self, batch, layer_idx=0):

        layer = self.gat_layers[layer_idx]
        if isinstance(layer, GATv2Conv):
            device = next(self.parameters()).device

            x = batch.face_attr.to(device)
            edge_index = batch.edge_index.to(device)
            edge_attr = batch.edge_attr.to(device) if hasattr(batch, 'edge_attr') else None

            if edge_attr is not None:
                out, alpha = layer(x, edge_index, edge_attr, return_attention_weights=True)
            else:
                out, alpha = layer(x, edge_index, return_attention_weights=True)

            alpha_weights = alpha[1].detach().cpu().numpy().flatten()

            # 归一化权重
            alpha_weights = (alpha_weights - alpha_weights.min()) / (alpha_weights.max() - alpha_weights.min() + 1e-8)

            # 转换为颜色列表
            cmap = plt.cm.Blues
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            colors = cmap(norm(alpha_weights))

            import torch_geometric.utils as pyg_utils
            G = pyg_utils.to_networkx(batch.to('cpu'), to_undirected=True)

            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G)

            # 使用颜色列表作为edge_color
            nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color=colors, width=2)
            plt.title(f"Layer {layer_idx} Attention")
            plt.axis('off')

            self.logger.experiment.add_figure(f"attention/layer_{layer_idx}", plt.gcf(), self.global_step)
            plt.close()

    def on_train_batch_end(self, outputs, batch, batch_idx):
    # 假设 self.gat_layers 是 GAT 层列表
        x = batch.face_attr
        for i, layer in enumerate(self.gat_layers):
            if isinstance(layer, GATv2Conv):
                x, attn = layer(x, batch.edge_index, batch.edge_attr, return_attention_weights=True)
            else:
                x = layer(x)
            self.logger.experiment.add_histogram(f"layer_{i}_output", x, self.global_step)

    def on_validation_epoch_end(self):
        batch = next(iter(self.trainer.val_dataloaders))
        self.visualize_attention(batch, layer_idx=0)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
