import torch
from torch import nn
from torch.nn import functional as F

from torchdrug import core, data, layers, utils
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from .data import Stack


@R.register("model.LS-QE")
class QueryExecutor(nn.Module, core.Configurable):
    stack_size = 2
    def __init__(self, model, logic="product", dropout_ratio=0.25, num_mlp_layer=2):
        super(QueryExecutor, self).__init__()
        self.model = RelationProjection(model, num_mlp_layer)
        self.symbolic_model = SymbolicTraversal()
        self.logic = logic
        self.dropout_ratio = dropout_ratio

    def traversal_dropout(self, graph, h_prob, r_index):
        sample, h_index = h_prob.nonzero().t()
        r_index = r_index[sample]
        any = -torch.ones_like(h_index)
        pattern = torch.stack([h_index, any, r_index], dim=-1)
        inverse_pattern = torch.stack([any, h_index, r_index ^ 1], dim=-1)
        pattern = torch.cat([pattern, inverse_pattern])
        edge_index = graph.match(pattern)[0]
        h_index, t_index = graph.edge_list.t()[:2]
        degree_h = h_index.bincount()
        degree_t = t_index.bincount()
        h_index, t_index = graph.edge_list[edge_index, :2].t()
        must_keep = (degree_h[h_index] <= 1) | (degree_t[t_index] <= 1)
        edge_index = edge_index[~must_keep]
        is_sampled = torch.rand(len(edge_index), device=self.device) <= self.dropout_ratio
        edge_index = edge_index[is_sampled]
        edge_mask = ~functional.as_mask(edge_index, graph.num_edge)
        return graph.edge_mask(edge_mask)
    def execute(self, graph, query, all_loss=None, metric=None):
        batch_size = len(query)
        self.stack = Stack(batch_size, self.stack_size, graph.num_node, device=self.device)
        self.symbolic_stack = Stack(batch_size, self.stack_size, graph.num_node, device=self.device)
        self.var = Stack(batch_size, query.shape[1], graph.num_node, device=self.device)
        self.symbolic_var = Stack(batch_size, query.shape[1], graph.num_node, device=self.device)
        self.IP = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        all_sample = torch.ones(batch_size, dtype=torch.bool, device=self.device)
        op = query[all_sample, self.IP]
        while not op.is_stop().all():
            is_operand = op.is_operand()
            is_intersection = op.is_intersection()
            is_union = op.is_union()
            is_negation = op.is_negation()
            is_projection = op.is_projection()
            if is_operand.any():
                h_index = op[is_operand].get_operand()
                self.apply_operand(is_operand, h_index, graph.num_node)
            if is_intersection.any():
                self.apply_intersection(is_intersection)
            if is_union.any():
                self.apply_union(is_union)
            if is_negation.any():
                self.apply_negation(is_negation)
            if not (is_operand | is_negation | is_intersection | is_union).any() and is_projection.any():
                r_index = op[is_projection].get_operand()
                self.apply_projection(is_projection, graph, r_index, all_loss=all_loss, metric=metric)
            op = query[all_sample, self.IP]
        if (self.stack.SP > 1).any():
            raise ValueError("More operands than expected")
    def forward(self, graph, query, all_loss=None, metric=None):
        self.execute(graph, query, all_loss=all_loss, metric=metric)
        t_prob = self.stack.pop()
        t_logit = ((t_prob + 1e-10) / (1 - t_prob + 1e-10)).log()
        return t_logit
    def visualize(self, graph, full_graph, query):
        self.execute(graph, query)
        var_probs = self.var.stack
        easy_answers = self.symbolic_var.stack
        self.execute(full_graph, query)
        all_answers = self.symbolic_var.stack
        return var_probs, easy_answers, all_answers

    def apply_operand(self, mask, h_index, num_node):
        h_prob = functional.one_hot(h_index, num_node)
        self.stack.push(mask, h_prob)
        self.symbolic_stack.push(mask, h_prob)
        self.var.push(mask, h_prob)
        self.symbolic_var.push(mask, h_prob)
        self.IP[mask] += 1

    def apply_intersection(self, mask):
        y_prob, sym_y_prob = self.stack.pop(mask), self.symbolic_stack.pop(mask)
        x_prob, sym_x_prob = self.stack.pop(mask), self.symbolic_stack.pop(mask)
        z_prob = self.conjunction(x_prob, y_prob)
        sym_z_prob = self.conjunction(sym_x_prob, sym_y_prob)
        self.stack.push(mask, z_prob)
        self.symbolic_stack.push(mask, sym_z_prob)
        self.var.push(mask, z_prob)
        self.symbolic_var.push(mask, sym_z_prob)
        self.IP[mask] += 1

    def apply_union(self, mask):
        y_prob, sym_y_prob = self.stack.pop(mask), self.symbolic_stack.pop(mask)
        x_prob, sym_x_prob = self.stack.pop(mask), self.symbolic_stack.pop(mask)
        z_prob = self.disjunction(x_prob, y_prob)
        sym_z_prob = self.disjunction(sym_x_prob, sym_y_prob)
        self.stack.push(mask, z_prob)
        self.symbolic_stack.push(mask, sym_z_prob)
        self.var.push(mask, z_prob)
        self.symbolic_var.push(mask, sym_z_prob)
        self.IP[mask] += 1

    def apply_negation(self, mask):
        x_prob, sym_x_prob = self.stack.pop(mask), self.symbolic_stack.pop(mask)
        y_prob = self.negation(x_prob)
        sym_y_prob = self.negation(sym_x_prob)
        self.stack.push(mask, y_prob)
        self.symbolic_stack.push(mask, sym_y_prob)
        self.var.push(mask, y_prob)
        self.symbolic_var.push(mask, sym_y_prob)
        self.IP[mask] += 1

    def apply_projection(self, mask, graph, r_index, all_loss=None, metric=None):
        h_prob, sym_h_prob = self.stack.pop(mask), self.symbolic_stack.pop(mask)
        if all_loss is not None:
            graph = self.traversal_dropout(graph, sym_h_prob, r_index)
        h_prob = h_prob.detach()
        t_prob = self.model(graph, h_prob, r_index, all_loss=all_loss, metric=metric)
        sym_t_prob = self.symbolic_model(graph, sym_h_prob, r_index, all_loss=all_loss, metric=metric)
        self.stack.push(mask, t_prob)
        self.symbolic_stack.push(mask, sym_t_prob)
        self.var.push(mask, t_prob)
        self.symbolic_var.push(mask, sym_t_prob)
        self.IP[mask] += 1

    def conjunction(self, x, y):
        if self.logic == "godel":
            return torch.min(x, y)
        elif self.logic == "product":
            return x * y
        elif self.logic == "lukasiewicz":
            return (x + y - 1).clamp(min=0)
        else:
            raise ValueError("Unknown fuzzy logic `%s`" % self.logic)

    def disjunction(self, x, y):
        if self.logic == "godel":
            return torch.max(x, y)
        elif self.logic == "product":
            return x + y - x * y
        elif self.logic == "lukasiewicz":
            return (x + y).clamp(max=1)
        else:
            raise ValueError("Unknown fuzzy logic `%s`" % self.logic)

    def negation(self, x):
        return 1 - x


@R.register("model.RelationProjection")
class RelationProjection(nn.Module, core.Configurable):
    def __init__(self, model, num_mlp_layer=2):
        super(RelationProjection, self).__init__()
        self.model = model
        self.query = nn.Embedding(model.num_relation, model.input_dim)
        self.mlp = layers.MLP(model.output_dim, [model.output_dim] * (num_mlp_layer - 1) + [1])
        self.alpha=0.6
        self.beta=0.4
    def forward(self, graph, h_prob, r_index, all_loss=None, metric=None):
        query = self.query(r_index)
        graph = graph.clone()
        with graph.graph():
            graph.query = query
        input = torch.einsum("bn, bd -> nbd", h_prob, query)
        output = self.model(graph, input, all_loss=all_loss, metric=metric)
        t_prob = F.sigmoid(self.mlp(output["node_feature"]).squeeze(-1))
        t_prob = self.alpha*t_prob+self.beta*F.sigmoid(self.mlp(output["node_feature1"]).squeeze(-1))
        return t_prob.t()


@R.register("model.Symbolic")
class SymbolicTraversal(nn.Module, core.Configurable):
    def forward(self, graph, h_prob, r_index, all_loss=None, metric=None):
        batch_size = len(h_prob)
        any = -torch.ones_like(r_index)
        pattern = torch.stack([any, any, r_index], dim=-1)
        edge_index, num_edges = graph.match(pattern)
        num_nodes = graph.num_node.repeat(batch_size)
        graph = data.PackedGraph(graph.edge_list[edge_index], num_nodes=num_nodes, num_edges=num_edges)
        adjacency = utils.sparse_coo_tensor(graph.edge_list.t()[:2], graph.edge_weight,
                                            (graph.num_node, graph.num_node))
        t_prob = functional.generalized_spmm(adjacency.t(), h_prob.view(-1, 1), sum="max").clamp(min=0)
        return t_prob.view_as(h_prob)
