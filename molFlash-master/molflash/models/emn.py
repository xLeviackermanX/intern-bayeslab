import torch
from torch import nn
import math

class EMN(nn.Module):

    def __init__(self, modeltype='Classification', message_passes=6,
                 edge_embedding_size=40,
                 edge_emb_hidden_dim=120,
                 att_hidden_dim=80,
                 msg_hidden_dim=80,
                 gather_width=100,
                 gather_att_hidden_dim=80,
                 gather_emb_hidden_dim=80,
                 out_hidden_dim=60):
        self.modeltype = modeltype
        super(EMN, self).__init__()
        self.node_features = 40
        self.edge_emb_depth = 3
        self.att_depth = 3
        self.edge_features = 4
        self.out_features = 1
        self.msg_depth = 3
        self.gather_att_depth = 3
        self.gather_emb_depth = 3
        self.out_depth = 2
        self.out_layer_shrinkage = 1.0
        self.gather_att_dropout_p = 0.0
        self.gather_emb_dropout_p = 0.0
        self.edge_emb_dropout_p = 0.0
        self.out_dropout = 0.0
        self.out_dropout_p = 0.0
        self.msg_dropout_p = 0.0
        self.att_dropout_p = 0.0
        self.message_passes = message_passes
        self.edge_embedding_size = edge_embedding_size
        self.gather = GraphGather(
            edge_embedding_size, gather_width,
            self.gather_att_depth, gather_att_hidden_dim, self.gather_att_dropout_p,
            self.gather_emb_depth, gather_emb_hidden_dim, self.gather_emb_dropout_p
        )
        out_layer_sizes = [
            round(out_hidden_dim * (self.out_layer_shrinkage ** (i / (self.out_depth - 1 + 1e-9)))) for i in range(self.out_depth)
        ]
        self.out_nn = FeedForwardNetwork(gather_width, out_layer_sizes, self.out_features, dropout_p=self.out_dropout_p)
        self.embedding_nn = FeedForwardNetwork(
            self.node_features * 2 + self.edge_features, [edge_emb_hidden_dim] * self.edge_emb_depth, edge_embedding_size,
            dropout_p = self.edge_emb_dropout_p
        )

        self.emb_msg_nn = FeedForwardNetwork(
            edge_embedding_size, [msg_hidden_dim] * self.msg_depth, edge_embedding_size, dropout_p = self.msg_dropout_p
        )
        self.att_msg_nn = FeedForwardNetwork(
            edge_embedding_size, [att_hidden_dim] * self.att_depth, edge_embedding_size, dropout_p = self.att_dropout_p
        )

        self.gru = nn.GRUCell(edge_embedding_size, edge_embedding_size, bias=False)


    def preprocess_edges(self, nodes, node_neighbours, edges):
        cat = torch.cat([nodes, node_neighbours, edges], dim=1)
        return torch.tanh(self.embedding_nn(cat))


    def propagate_edges(self, edges, ingoing_edge_memories, ingoing_edges_mask):
        BIG_NEGATIVE = -1e6
        energy_mask = ((1 - ingoing_edges_mask).float() * BIG_NEGATIVE).unsqueeze(-1)
        cat = torch.cat([edges.unsqueeze(1), ingoing_edge_memories], dim=1)
        embeddings = self.emb_msg_nn(cat)

        edge_energy = self.att_msg_nn(edges)
        ing_memory_energies = (self.att_msg_nn(ingoing_edge_memories)) + energy_mask
        energies = torch.cat([edge_energy.unsqueeze(1), ing_memory_energies], dim=1)
        attention = torch.softmax(energies, dim=1)

        message = (attention * embeddings).sum(dim=1)
        return self.gru(message)

    def readout(self, hidden_nodes, input_nodes, node_mask):
        graph_embeddings = self.gather(hidden_nodes, input_nodes, node_mask)
        return self.out_nn(graph_embeddings)

    def forward(self,inp_data):
        adjacency, nodes, edges = inp_data
        edges_b_idx, edges_n_idx, edges_nhb_idx = adjacency.nonzero().unbind(-1)
        n_edges = edges_n_idx.shape[0]
        adj_of_edge_batch_indices = adjacency.clone().long()
        r = torch.arange(n_edges) + 1

        # moving the tensor to gpu (if needed)
        r = r.to(adj_of_edge_batch_indices.device)


        adj_of_edge_batch_indices[edges_b_idx, edges_n_idx, edges_nhb_idx] = r

        ingoing_edges_eb_idx = (torch.cat([
            row[row.nonzero()] for row in adj_of_edge_batch_indices[edges_b_idx, edges_nhb_idx, :]
        ]) - 1).squeeze()

        edge_degrees = adjacency[edges_b_idx, edges_nhb_idx, :].sum(-1).long()
        ingoing_edges_igeb_idx = torch.cat([i * torch.ones(d) for i, d in enumerate(edge_degrees)]).long()
        ingoing_edges_ige_idx = torch.cat([torch.arange(i) for i in edge_degrees]).long()

        batch_size = adjacency.shape[0]
        n_nodes = adjacency.shape[1]
        max_node_degree = adjacency.sum(-1).max().int()
        edge_memories = torch.zeros(n_edges, self.edge_embedding_size)
        # edge_memories = edge_memories.to("cuda")
        ingoing_edge_memories = torch.zeros(n_edges, max_node_degree, self.edge_embedding_size)
        ingoing_edges_mask = torch.zeros(n_edges, max_node_degree)


        # moving to gpu (if needed)
        edge_memories, ingoing_edge_memories, ingoing_edges_mask = edge_memories.to(edges.device), ingoing_edge_memories.to(edges.device), ingoing_edges_mask.to(edges.device)

        edge_batch_nodes = nodes[edges_b_idx, edges_n_idx, :]
        edge_batch_neighbours = nodes[edges_b_idx, edges_nhb_idx, :]
        edge_batch_edges = edges[edges_b_idx, edges_n_idx, edges_nhb_idx, :]
        edge_batch_edges = self.preprocess_edges(edge_batch_nodes, edge_batch_neighbours, edge_batch_edges)


        ingoing_edges_nhb_idx = edges_nhb_idx[ingoing_edges_eb_idx]
        ingoing_edges_receiving_edge_n_idx = edges_n_idx[ingoing_edges_igeb_idx]
        not_same_idx = (ingoing_edges_receiving_edge_n_idx != ingoing_edges_nhb_idx).nonzero()
        ingoing_edges_eb_idx = ingoing_edges_eb_idx[not_same_idx].squeeze()
        ingoing_edges_ige_idx = ingoing_edges_ige_idx[not_same_idx].squeeze()
        ingoing_edges_igeb_idx = ingoing_edges_igeb_idx[not_same_idx].squeeze()

        ingoing_edges_mask[ingoing_edges_igeb_idx, ingoing_edges_ige_idx] = 1
        for i in range(self.message_passes):
            ingoing_edge_memories[ingoing_edges_igeb_idx, ingoing_edges_ige_idx, :] = \
                edge_memories[ingoing_edges_eb_idx, :]
            edge_memories = self.propagate_edges(edge_batch_edges, ingoing_edge_memories.clone(), ingoing_edges_mask)

        node_mask = (adjacency.sum(-1) != 0)

        node_sets = torch.zeros(batch_size, n_nodes, max_node_degree, self.edge_embedding_size)
        if next(self.parameters()).is_cuda:
            node_sets = node_sets.to(edges.device)

        edge_batch_edge_memory_indices = torch.cat(
            [torch.arange(row.sum()) for row in adjacency.view(-1, n_nodes)]
        ).long()

        node_sets[edges_b_idx, edges_n_idx, edge_batch_edge_memory_indices, :] = edge_memories
        graph_sets = node_sets.sum(2)
        output = self.readout(graph_sets, graph_sets, node_mask)
        # print(self.modeltype)
        if self.modeltype == "Classification":
            sigmoid = nn.Sigmoid()
            output = sigmoid(output)
            # output = (output >= 0.5).float()
            # output = output
            # output = output.detach()
            return output


        return output




class FeedForwardNetwork(nn.Module):

    def __init__(self, in_features, hidden_layer_sizes, out_features, activation='SELU', bias=False, dropout_p=0.0):
        super(FeedForwardNetwork, self).__init__()

        Activation = nn.SELU
        Dropout = nn.AlphaDropout
        init_constant = 1.0


        layer_sizes = [in_features] + hidden_layer_sizes + [out_features]

        layers = []
        for i in range(len(layer_sizes) - 2):
            layers.append(Dropout(dropout_p))
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], bias))
            layers.append(Activation())
        layers.append(Dropout(dropout_p))
        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1], bias))

        self.seq = nn.Sequential(*layers)

        for i in range(1, len(layers), 3):

            nn.init.normal_(layers[i].weight, std=math.sqrt(init_constant / layers[i].weight.size(1)))

    def forward(self, input):
        # input = input.to("cuda")
        return self.seq(input)

    def __repr__(self):
        ffnn = type(self).__name__
        in_features = self.seq[1].in_features
        hidden_layer_sizes = [linear.out_features for linear in self.seq[1:-1:3]]
        out_features = self.seq[-1].out_features
        if len(self.seq) > 2:
            activation = str(self.seq[2])
        else:
            activation = 'None'
        bias = self.seq[1].bias is not None
        dropout_p = self.seq[0].p
        return '{}(in_features={}, hidden_layer_sizes={}, out_features={}, activation={}, bias={}, dropout_p={})'.format(
            ffnn, in_features, hidden_layer_sizes, out_features, activation, bias, dropout_p
        )

class GraphGather(nn.Module):

    def __init__(self, node_features, out_features,
                 att_depth=2, att_hidden_dim=100, att_dropout_p=0.0,
                 emb_depth=2, emb_hidden_dim=100, emb_dropout_p=0.0):
        super(GraphGather, self).__init__()


        self.att_nn = FeedForwardNetwork(
            node_features * 2, [att_hidden_dim] * att_depth, out_features, dropout_p=att_dropout_p, bias=False
        )
        self.emb_nn = FeedForwardNetwork(
            node_features, [emb_hidden_dim] * emb_depth, out_features, dropout_p=emb_dropout_p, bias=False
        )

    def forward(self, hidden_nodes, input_nodes, node_mask):
        cat = torch.cat([hidden_nodes, input_nodes], dim=2)
        energy_mask = (node_mask == 0).float() * 1e6
        energies = self.att_nn(cat) - energy_mask.unsqueeze(-1)
        attention = torch.sigmoid(energies)
        embedding = self.emb_nn(hidden_nodes)
        return torch.sum(attention * embedding, dim=1)
