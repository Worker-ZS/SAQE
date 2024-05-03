import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add, scatter_mean, scatter_max, scatter_min

from torchdrug import layers, utils
from torchdrug.layers import functional


class GeneralizedRelationalConv(layers.MessagePassingBase):

    eps = 1e-6

    message2mul = {
        "transe": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, query_input_dim, message_func="distmult",
                 aggregate_func="pna", layer_norm=False, activation="relu", dependent=True):
        super(GeneralizedRelationalConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.query_input_dim = query_input_dim
        self.message_func = message_func
        self.aggregate_func = aggregate_func
        self.dependent = dependent

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        if self.aggregate_func == "pna":
            self.linear = nn.Linear(input_dim * 13, output_dim)
        else:
            self.linear = nn.Linear(input_dim * 2, output_dim)
        if dependent:
            self.relation_linear = nn.Linear(query_input_dim, num_relation * input_dim)
        else:
            self.relation = nn.Embedding(num_relation, input_dim)

    def message(self, graph, input):
        assert graph.num_relation == self.num_relation

        batch_size = len(graph.query)
        node_in, node_out, relation = graph.edge_list.t()
        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation, self.input_dim)
        else:
            relation_input = self.relation.weight.expand(batch_size, -1, -1)
        relation_input = relation_input.transpose(0, 1)
        node_input = input[node_in]
        edge_input = relation_input[relation]

        if self.message_func == "transe":
            message = edge_input + node_input
        elif self.message_func == "distmult":
            message = edge_input * node_input
        elif self.message_func == "rotate":
            node_re, node_im = node_input.chunk(2, dim=-1)
            edge_re, edge_im = edge_input.chunk(2, dim=-1)
            message_re = node_re * edge_re - node_im * edge_im
            message_im = node_re * edge_im + node_im * edge_re
            message = torch.cat([message_re, message_im], dim=-1)
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        message = torch.cat([message, graph.boundary])

        return message

    def aggregate(self, graph, message):
        node_out = graph.edge_list[:, 1]
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=graph.device)])
        edge_weight = torch.cat([graph.edge_weight, torch.ones(graph.num_node, device=graph.device)])
        edge_weight = edge_weight.unsqueeze(-1).unsqueeze(-1)
        degree_out = graph.degree_out.unsqueeze(-1).unsqueeze(-1) + 1

        if self.aggregate_func == "sum":
            update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "mean":
            update = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
        elif self.aggregate_func == "max":
            update = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
        elif self.aggregate_func == "pna":
            mean = scatter_mean(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            sq_mean = scatter_mean(message ** 2 * edge_weight, node_out, dim=0, dim_size=graph.num_node)
            max = scatter_max(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            min = scatter_min(message * edge_weight, node_out, dim=0, dim_size=graph.num_node)[0]
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        return update

    def message_and_aggregate(self, graph, input):
        if graph.requires_grad or self.message_func == "rotate":
            return super(GeneralizedRelationalConv, self).message_and_aggregate(graph, input)

        assert graph.num_relation == self.num_relation

        batch_size = len(graph.query)
        input = input.flatten(1)
        boundary = graph.boundary.flatten(1)

        degree_out = graph.degree_out.unsqueeze(-1) + 1
        if self.dependent:
            relation_input = self.relation_linear(graph.query).view(batch_size, self.num_relation, self.input_dim)
            relation_input = relation_input.transpose(0, 1).flatten(1)
        else:
            relation_input = self.relation.weight.repeat(1, batch_size)
        adjacency = graph.adjacency.transpose(0, 1)

        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        if self.aggregate_func == "sum":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = update + boundary
        elif self.aggregate_func == "mean":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            update = (update + boundary) / degree_out
        elif self.aggregate_func == "max":
            update = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            update = torch.max(update, boundary)
        elif self.aggregate_func == "pna":
            sum = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
            sq_sum = functional.generalized_rspmm(adjacency, relation_input ** 2, input ** 2, sum="add", mul=mul)
            max = functional.generalized_rspmm(adjacency, relation_input, input, sum="max", mul=mul)
            min = functional.generalized_rspmm(adjacency, relation_input, input, sum="min", mul=mul)
            mean = (sum + boundary) / degree_out
            sq_mean = (sq_sum + boundary ** 2) / degree_out
            max = torch.max(max, boundary)
            min = torch.min(min, boundary)
            std = (sq_mean - mean ** 2).clamp(min=self.eps).sqrt()
            features = torch.cat([mean.unsqueeze(-1), max.unsqueeze(-1), min.unsqueeze(-1), std.unsqueeze(-1)], dim=-1)
            features = features.flatten(-2)
            scale = degree_out.log()
            scale = scale / scale.mean()
            scales = torch.cat([torch.ones_like(scale), scale, 1 / scale.clamp(min=1e-2)], dim=-1)
            update = (features.unsqueeze(-1) * scales.unsqueeze(-2)).flatten(-2)
        else:
            raise ValueError("Unknown aggregation function `%s`" % self.aggregate_func)

        return update.view(len(update), batch_size, -1)

    def combine(self, input, update):
        output = self.linear(torch.cat([input, update], dim=-1))
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output


class CompositionalGraphConv(layers.MessagePassingBase):

    message2mul = {
        "sub": "add",
        "distmult": "mul",
    }

    def __init__(self, input_dim, output_dim, num_relation, message_func="mult", layer_norm=False, activation="relu"):
        super(CompositionalGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_relation = num_relation
        self.message_func = message_func

        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = None
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation

        self.loop_relation = nn.Embedding(1, input_dim)
        self.linear = nn.Linear(3 * input_dim, output_dim)
        self.relation_linear = nn.Linear(input_dim, output_dim)

    def message(self, graph, input):
        assert graph.num_relation == self.num_relation

        relation_input = graph.relation_input
        node_in, node_out, relation = graph.edge_list.t()
        node_input = torch.cat([input[node_in], input])
        edge_input = torch.cat([relation_input[relation], self.loop_relation.weight.repeat(graph.num_node, 1)])
        edge_input = edge_input.unsqueeze(1)

        if self.message_func == "sub":
            message = node_input - edge_input
        elif self.message_func == "mult":
            message = node_input * edge_input
        elif self.message_func == "corr":
            node_input = torch.fft.rfft(node_input)
            edge_input = torch.fft.rfft(edge_input)
            message = torch.fft.irfft(node_input.conj() * edge_input, n=input.shape[-1])
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)

        graph.relation_input = self.relation_linear(relation_input)

        return message

    def aggregate(self, graph, message):
        batch_size = message.shape[1]
        node_in, node_out, relation = graph.edge_list.t()
        edge_weight = graph.edge_weight * 2 / (graph.degree_in[node_in] * graph.degree_out[node_out]) ** 0.5
        edge_weight = torch.cat([edge_weight, torch.ones(graph.num_node, device=self.device)])
        edge_weight = edge_weight.unsqueeze(-1).unsqueeze(-1)
        node_out = node_out * 3 + relation % 2
        node_out = torch.cat([node_out, torch.arange(graph.num_node, device=self.device) * 3 + 2])
        update = scatter_add(message * edge_weight, node_out, dim=0, dim_size=graph.num_node * 3)
        update = update.view(graph.num_node, 3, batch_size, self.input_dim)
        update = update.transpose(1, 2).reshape(graph.num_node, batch_size, -1)

        return update

    def message_and_aggregate(self, graph, input):
        if graph.requires_grad or self.message_func == "corr":
            return super(CompositionalGraphConv, self).message_and_aggregate(graph, input)

        assert graph.num_relation == self.num_relation

        batch_size = len(graph.query)
        input = input.flatten(1)
        relation_input = torch.cat([graph.relation_input, self.loop_relation.weight])
        relation_input = relation_input.repeat(1, batch_size)
        node_in, node_out, relation = graph.edge_list.t()
        edge_weight = graph.edge_weight * 2 / (graph.degree_in[node_in] * graph.degree_out[node_out]) ** 0.5
        edge_weight = torch.cat([edge_weight, torch.ones(graph.num_node, device=self.device)])
        node_in = torch.cat([node_in, torch.arange(graph.num_node, device=self.device)])
        node_out = torch.cat([node_out * 3 + relation % 2, torch.arange(graph.num_node, device=self.device) * 3 + 2])
        loop = torch.ones(graph.num_node, dtype=torch.long, device=self.device) * graph.num_relation
        relation = torch.cat([relation, loop])
        adjacency = utils.sparse_coo_tensor(torch.stack([node_in, node_out, relation]), edge_weight,
                                            (graph.num_node, graph.num_node * 3, graph.num_relation + 1))
        adjacency = adjacency.transpose(0, 1)

        if self.message_func == "sub":
            relation_input = -relation_input
        if self.message_func in self.message2mul:
            mul = self.message2mul[self.message_func]
        else:
            raise ValueError("Unknown message function `%s`" % self.message_func)
        update = functional.generalized_rspmm(adjacency, relation_input, input, sum="add", mul=mul)
        update = update.view(graph.num_node, 3, batch_size, self.input_dim)
        update = update.transpose(1, 2).reshape(graph.num_node, batch_size, -1)

        graph.relation_input = self.relation_linear(graph.relation_input)

        return update

    def combine(self, input, update):
        output = self.linear(update)
        if self.layer_norm:
            output = self.layer_norm(output)
        if self.activation:
            output = self.activation(output)
        return output
