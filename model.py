import torch
from weighted_gat import WeightedGATConv
from gru_set2set import GRUSet2Set


class FGNN(torch.nn.Module):
    def __init__(self, config):
        super(FGNN, self).__init__()
        self.heads = config["WGAT_heads"]
        self.hidden_size = config["hidden_size"]
        self.item_embedding = torch.nn.Embedding(config["item_num"], embedding_dim=self.hidden_size)
        self.WGAT1 = WeightedGATConv(in_channels=config["hidden_size"],
                                     out_channels=config["hidden_size"],
                                     heads=self.heads,
                                     concat=False,
                                     negative_slope=config["leaky_relu"],
                                     dropout=config["dropout"],
                                     bias=True,
                                     weighted=True)
        self.WGAT2 = WeightedGATConv(in_channels=config["hidden_size"],
                                     out_channels=config["hidden_size"],
                                     heads=self.heads,
                                     concat=False,
                                     negative_slope=config["leaky_relu"],
                                     dropout=config["dropout"],
                                     bias=True,
                                     weighted=True)
        self.WGAT3 = WeightedGATConv(in_channels=config["hidden_size"],
                                     out_channels=config["hidden_size"],
                                     heads=self.heads,
                                     concat=False,
                                     negative_slope=config["leaky_relu"],
                                     dropout=config["dropout"],
                                     bias=True,
                                     weighted=True)
        self.set2set = GRUSet2Set(in_channels=config["hidden_size"], processing_steps=3)
        self.linear = torch.nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x - 1, data.edge_index, data.batch, data.edge_attr
        x = self.item_embedding(x)

        x = self.WGAT1(x, edge_index, edge_attr)
        x = self.WGAT2(x, edge_index, edge_attr)
        x = self.WGAT3(x, edge_index, edge_attr)

        q_star = self.set2set(x, batch)
        scores = self.linear(q_star) @ self.item_embedding.weight.T
        return scores
