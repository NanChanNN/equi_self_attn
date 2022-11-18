import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm

import itertools
import math
import numpy as np

import dgl
import dgl.function as fn
from dgl.nn.functional import edge_softmax

default_MLP_archi = {}
global_total_counter = 0
global_rank2_counter = 0

def truncate_normal_initialization(weights, scale=1.0):
    fan_out, fan_in = weights.shape
    scale = scale / max(1, fan_in)
    a = -2
    b = 2
    std = math.sqrt(scale) / truncnorm.std(a=a, b=b, loc=0, scale=1)
    sample_weights = truncnorm.rvs(a=a, b=b, loc=0, scale=std, size=(fan_out * fan_in))
    sample_weights = sample_weights.reshape(fan_out, fan_in)
    with torch.no_grad():
        weights.copy_(torch.from_numpy(sample_weights).float().to(weights.device))


def build_MLP_network(in_dim, out_dim, archi: dict = default_MLP_archi):
    """
    Build a MLP network whose architecture is indicated in 'archi'
    :param in_dim: input dimension
    :param out_dim: output dimension
    :param archi: the architecture dictionary
    :return:
    """
    mlp_net = []
    net_in = in_dim
    net_out = out_dim
    hidden_config = archi['hidden']
    final_config = archi['final']

    if archi['in_LN']:
        mlp_net.append(nn.LayerNorm(net_in))

    for i in range(archi['layer'] - 1):
        net_hidden = hidden_config['dim']
        mlp_net.append(nn.Linear(net_in, net_hidden, bias=hidden_config['bias']))
        
        if hidden_config['bias'] and (hidden_config['init'] not in ['none', 'kaiming_uni']):
            nn.init.constant_(mlp_net[-1].bias, 0.)
            
        if hidden_config['init'] == 'lecun':
            truncate_normal_initialization(mlp_net[-1].weight, scale=1.)
        elif hidden_config['init'] == 'relu':
            truncate_normal_initialization(mlp_net[-1].weight, scale=2.)
        elif hidden_config['init'] == 'leaky':
            torch.nn.init.kaiming_normal_(mlp_net[-1].weight)
        elif hidden_config['init'] == 'glorot':
            torch.nn.init.xavier_uniform_(mlp_net[-1].weight, gain=1.)
        elif hidden_config['init'] == 'kaiming_uni':
            torch.nn.init.kaiming_uniform_(mlp_net[-1].weight)
        elif hidden_config['init'] == 'gating':
            torch.nn.init.constant_(mlp_net[-1].weight, 0.)
            if hidden_config['bias']:
                nn.init.constant_(mlp_net[-1].bias, 1.) 
        
        if hidden_config['norm'] == 'LN':
            mlp_net.append(nn.LayerNorm(net_hidden))
        elif hidden_config['norm'] == 'BN':
            mlp_net.append(nn.BatchNorm1d(net_hidden))
        else:
            assert hidden_config['norm'] == 'none'
        
        mlp_net.append(nn.Dropout(hidden_config['drop']))

        mlp_net.append(hidden_config['act'])

        net_in = net_hidden

    mlp_net.append(nn.Linear(net_in, net_out, bias=final_config['bias']))

    if final_config['bias'] and (final_config['init'] not in ['none', 'kaiming_uni']):
        nn.init.constant_(mlp_net[-1].bias, 0.)
    if final_config['init'] == 'lecun':
        truncate_normal_initialization(mlp_net[-1].weight, scale=1.)
    elif final_config['init'] == 'relu':
        truncate_normal_initialization(mlp_net[-1].weight, scale=2.)
    elif final_config['init'] == 'leaky':
        torch.nn.init.kaiming_normal_(mlp_net[-1].weight)
    elif final_config['init'] == 'glorot':
        torch.nn.init.xavier_uniform_(mlp_net[-1].weight, gain=1.)
    elif final_config['init'] == 'kaiming_uni':
        torch.nn.init.kaiming_uniform_(mlp_net[-1].weight)
    elif final_config['init'] == 'gating':
        torch.nn.init.constant_(mlp_net[-1].weight, 0.)
        if final_config['bias']:
            nn.init.constant_(mlp_net[-1].bias, 1.)
    elif final_config['init'] == 'residue':
        torch.nn.init.constant_(mlp_net[-1].weight, 0.)
    
    mlp_net.append(nn.Dropout(final_config['drop']))
    
    if final_config['act'] is not None:
        mlp_net.append(final_config['act'])

    return nn.Sequential(*mlp_net)


class SODInvariantScalars(nn.Module):
    """ NN parameterized SO(d)-invariant scalar function.
    g(VV^T, scalars, minors(V)) -> scalar
    """
    def __init__(self, m_in_vec: int, m_in_s: int, out_dim: int, net_archi: dict = default_MLP_archi):
        """
        :param m_in_vec: channels of input vector features
        :param m_in_s: channels of input scalar features
        :param out_dim: dimension of the output feature
        :param net_archi: the dimension of hidden layers in the network
        """
        super(SODInvariantScalars, self).__init__()
        self.m_in_vec = m_in_vec
        self.m_in_s = m_in_s
        self.out_dim = out_dim
        
        triu_idx = torch.triu_indices(m_in_vec, m_in_vec)
        assert triu_idx.shape[1] == int(m_in_vec * (m_in_vec + 1) / 2)
        self.register_buffer('triu_idx', triu_idx)

        sub_idx = torch.combinations(torch.arange(m_in_vec), 3, False)
        if sub_idx.shape[0] != 0:
            self.register_buffer('sub_idx', torch.flatten(sub_idx))
        else:
            self.register_buffer('sub_idx', None)

        self.in_dim = m_in_s + self.triu_idx.shape[1] + sub_idx.shape[0]
        self.net = build_MLP_network(self.in_dim, self.out_dim, net_archi)

    def __repr__(self):
        return f"SODInvariantScalars(m_in_vec={self.m_in_vec}, m_in_s={self.m_in_s}, " \
               f"out_dim={self.out_dim}, hidden_dim={self.hidden_dim})"

    @staticmethod
    def _compute_determinant(A: torch.Tensor):
        """ Compute the determinant of batched 3x3 matrices
        :param A: ..., 3, 3
        :return:
        """
        output = A[..., 0, 0] * A[..., 1, 1] * A[..., 2, 2] + \
                 A[..., 0, 1] * A[..., 1, 2] * A[..., 2, 0] + \
                 A[..., 0, 2] * A[..., 1, 0] * A[..., 2, 1] - \
                 A[..., 0, 2] * A[..., 1, 1] * A[..., 2, 0] - \
                 A[..., 0, 1] * A[..., 1, 0] * A[..., 2, 2] - \
                 A[..., 0, 0] * A[..., 1, 2] * A[..., 2, 1]
        return output

    def forward(self, features: dict):
        """
        :param features: dict,
            'vec': B, m_in_vec, 3
            'scalar': B, m_in_s, 1
        :return:
            B, out_dim
        """
        net_input_features = []

        if 'vec' in features:
            v_mat = features['vec']  # B, m_in_vec, 3
            B = v_mat.shape[0]
            inner_product = torch.einsum('...ik, ...jk->...ij', v_mat, v_mat)  # B, m_in_vec, m_in_vec
            inner_product = inner_product[..., self.triu_idx[0], self.triu_idx[1]]  # B, m_in_vec * (m_in_vec + 1) / 2
            net_input_features.append(inner_product)

            if self.sub_idx is not None:
                sub_mat = torch.index_select(v_mat, 1, self.sub_idx)
                sub_mat = sub_mat.reshape(B, -1, 3, 3)
                sub_det = self._compute_determinant(sub_mat)
                net_input_features.append(sub_det)      # B, (n,3)

        if 'scalar' in features:
            s_vec = features['scalar'][..., 0]
            net_input_features.append(s_vec)

        net_input_features = torch.cat(net_input_features, dim=-1)
        scalar_weights = self.net(net_input_features)  # B, out_dim

        return scalar_weights
    

class ODInvariantScalars(nn.Module):
    """NN parameterized O(d)-invariant scalar function.
    g(VV^T, scalars) -> scalar
    """

    def __init__(self, m_in_vec: int, m_in_s: int, out_dim: int, net_archi: dict = default_MLP_archi):
        """
        :param m_in_vec: channels of input vector features
        :param m_in_s: channels of input scalar features
        :param out_dim: dimension of the output feature
        :param hidden_dim: the dimension of hidden layers in the network
        """
        super(ODInvariantScalars, self).__init__()
        self.m_in_vec = m_in_vec
        self.m_in_s = m_in_s
        self.in_dim = int((m_in_vec + 1) * m_in_vec / 2) + m_in_s
        # self.in_dim = m_in_vec * m_in_vec + m_in_s
        self.out_dim = out_dim

        self.net = build_MLP_network(self.in_dim, self.out_dim, net_archi)

        # indices for the upper triangular part of the inner product matrix
        triu_idx = torch.triu_indices(m_in_vec, m_in_vec)
        self.register_buffer('triu_idx', triu_idx)

    def __repr__(self):
        return f"ODInvariantScalars(m_in_vec={self.m_in_vec}, m_in_s={self.m_in_s}, " \
               f"out_dim={self.out_dim}, hidden_dim={self.hidden_dim})"

    def forward(self, features: dict):
        """
        :param features: dict,
            'vec': B, m_in_vec, 3
            'scalar': B, m_in_s, 1 (optional)
        :return:
            B, out_dim
        """
        net_input_features = []

        if 'vec' in features:
            v_mat = features['vec']     # B, m_in_vec, 3
            B = v_mat.shape[0]
            inner_product = torch.einsum('...ik, ...jk->...ij', v_mat, v_mat)   # B, m_in_vec, m_in_vec
            inner_product = inner_product[..., self.triu_idx[0], self.triu_idx[1]]  # B, m_in_vec * (m_in_vec + 1) / 2
            # inner_product = inner_product.reshape(B, self.m_in_vec * self.m_in_vec)
            net_input_features.append(inner_product)
        
        if 'scalar' in features:
            s_vec = features['scalar'][..., 0]
            net_input_features.append(s_vec)
        
        net_input_features = torch.cat(net_input_features, dim=-1)
        scalar_weights = self.net(net_input_features)  # B, out_dim

        return scalar_weights


class SO3EquivariantVector(nn.Module):
    """SO(3) Equivariant Vector Function. Possibly update scalar vectors, depending on whether
    the output channels are zeros or not.
    The updated scalar vector is extracted from the output of ODInvariantScalars
    """

    def __init__(self, m_in_vec: int, m_out_vec: int, m_in_s: int, m_out_s: int, invariant_mod: str, 
                 cross_product: bool, input_LN: bool = False, net_archi: dict = default_MLP_archi):
        """
        :param m_in_vec: channels of input vector features
        :param m_out_vec: channels of output vector features
        :param m_in_s: channels of input scalar features
        :param m_out_s: channels of output scalar features
        """
        super(SO3EquivariantVector, self).__init__()
        assert (m_out_vec == 0) or (m_in_vec >0)
        assert invariant_mod in ['OD', 'SOD'], 'unresolved invariant module %s'%(str(invariant_mod))

        self.m_in_vec = m_in_vec
        self.m_out_vec = m_out_vec
        if cross_product:
            self.cross_product = True
        else:
            self.cross_product = False
        if self.cross_product:
            self.m_cross_prod = int(m_in_vec * (m_in_vec - 1) / 2)
        else:
            self.m_cross_prod = 0
        self.m_in_s = m_in_s
        self.m_out_s = m_out_s
        self.input_LN = input_LN
        
        if self.input_LN:
            self.input_layer_norm = SO3LayerNorm({'vec': m_in_vec, 'scalar':m_in_s})

        self.normalize_term = self.m_cross_prod + self.m_in_vec
        self.out_dim_vec = self.m_out_vec * (self.m_cross_prod + self.m_in_vec)
        self.out_dim_s = self.m_out_s
        self.out_dim = self.out_dim_vec + self.out_dim_s

        if invariant_mod == 'OD':
            self.scalar_nets = ODInvariantScalars(self.m_in_vec, self.m_in_s, self.out_dim, net_archi=net_archi)
        else:
            self.scalar_nets = SODInvariantScalars(self.m_in_vec, self.m_in_s, self.out_dim, net_archi=net_archi)

        # cross product indices
        cross_prod_idx = torch.triu_indices(m_in_vec, m_in_vec, offset=1)
        self.register_buffer('cross_prod_idx', cross_prod_idx)

    def __repr__(self):
        return f"SO3EquivariantVector(m_in_vec={self.m_in_vec}, m_out_vec={self.m_out_vec}, m_in_s={self.m_in_s}," \
               f"m_out_s={self.m_out_s})"

    def forward(self, features: dict):
        """
        :param features: dict
            'vec': B, m_in_vec, 3
            'scalar': B, m_in_s, 1 (optional)
        :return: dict
            'vec': B, m_out_vec, 3
            'scalar': B, m_out_s, 1
        """
        if self.input_LN:
            features = self.input_layer_norm(features)
        
        weights = self.scalar_nets(features)  # B, out_dim
        vec_weights, s_weights = torch.split(weights, [self.out_dim_vec, self.out_dim_s], dim=-1)
        new_features = {}

        if self.out_dim_vec > 0:
            v_mat = features['vec']
            B = v_mat.shape[0]
            vec_weights = vec_weights.reshape(B, self.m_out_vec, self.m_cross_prod + self.m_in_vec)
            if self.m_in_vec > 1 and self.cross_product:
                c0, c1 = self.cross_prod_idx
                mat0 = torch.gather(v_mat, 1, c0[None, :, None].expand(B, -1, 3))  # B, m_cross_prod, 3
                mat1 = torch.gather(v_mat, 1, c1[None, :, None].expand(B, -1, 3))  # B, m_cross_prod, 3
                cross_prods = torch.linalg.cross(mat0, mat1, dim=-1)  # B, m_cross_prod, 3
                cat_mat = torch.cat([v_mat, cross_prods], dim=-2)  # B, m_cross_prod + m_in_vec ,3
            else:
                cat_mat = v_mat
             # B, m_out_vec, 3
            new_features['vec'] = torch.einsum('...ij, ...jk->...ik', vec_weights, cat_mat) / self.normalize_term

        if self.out_dim_s > 0:
            new_features['scalar'] = s_weights.unsqueeze(-1)    # B, m_out_s, 1

        return new_features
    

class PairwiseSO3Conv(nn.Module):
    """ Generate pairwise features.
    f_ji = h('f_j', 's_j', x_i - x_j, edge_ji)   -> {'vec' , 'scalar'}
    """

    def __init__(self, m_in_vec: int, m_out_vec: int, m_in_s: int, m_out_s: int, invariant_mod: str,
                 cross_product: bool, edge_dim: int = 0, net_archi: dict = default_MLP_archi):
        """
        :param m_in_vec: channels of input vector features
        :param m_out_vec: channels of output vector features
        :param m_in_s: channels of input scalar features
        :param m_out_s: channels of output scalar features
        :param edge_dim: dimensions of edge features.
        """
        super(PairwiseSO3Conv, self).__init__()
        self.m_in_vec = m_in_vec
        self.m_out_vec = m_out_vec
        self.m_in_s = m_in_s
        self.m_out_s = m_out_s
        self.edge_dim = edge_dim

        net_in_vec = m_in_vec + 1  # cat(f_j, x_i - x_j)
        net_in_s = m_in_s + edge_dim  # cat(s_j, edge_ji)
        self.net = SO3EquivariantVector(net_in_vec, m_out_vec, net_in_s, m_out_s, invariant_mod, cross_product,
                                        net_archi=net_archi)

    def __repr__(self):
        return f"PairwiseSO3Conv(m_in_vec={self.m_in_vec}, m_out_vec={self.m_out_vec}, m_in_s={self.m_in_s}," \
               f"m_out_s={self.m_out_s}, edge_dim={self.edge_dim})"

    def udf_edge(self):
        """Pairwise convolution features. This function is set up as a User Defined Function in DGL.
        :return:
            node -> edge function handle
        """

        def fnc(edges):
            input_feat_dict = {}
            rel = (edges.dst['x'] - edges.src['x'])  # relative distance - num_edges, 3

            vec_feats = []
            if 'vec' in edges.src:
                vec_feats.append(edges.src['vec'])  # num_edges, m_in_vec, 3
            vec_feats.append(rel[:, None, :])
            input_feat_dict['vec'] = torch.cat(vec_feats, dim=1)  # num_edges, m_in + 1, 3

            add_feat = []
            if 'scalar' in edges.src:
                add_feat.append(edges.src['scalar'])  # num_edges, m_in_s, 1
                assert add_feat[-1].shape[-2] == self.m_in_s
            if 'w' in edges.data and self.edge_dim != 0:
                add_feat.append(edges.data['w'].unsqueeze(-1))  # num_edges, edge_dim, 1
                assert add_feat[-1].shape[-2] == self.edge_dim
            if len(add_feat) != 0:
                input_feat_dict['scalar'] = torch.cat(add_feat, dim=-2)  # num_edges, m_in_s + edge_dim, 1

            out_dict = self.net(input_feat_dict)  # num_edges, m_out, 3
            return out_dict

        return fnc

    def forward(self, features: dict, G):
        """
        :param features: dict, input features
            'vec' - B, m_in_vec, 3
            'scalar' - B, m_in_s, 1
        :param G: dgl graph object
        :return:
            dict:
            'vec' - n_edges, m_out_vec, 3
            'scalar' - n_edges, m_out_s, 1
        """
        with G.local_scope():
            if 'vec' in features:
                G.ndata['vec'] = features['vec']
            if 'scalar' in features:
                G.ndata['scalar'] = features['scalar']
            G.apply_edges(self.udf_edge())

            return {key: G.edata[key] for key in ['vec', 'scalar'] if key in G.edata}


class AttentionModule(nn.Module):
    """An SO(3)-equivariant self-attention module for DGL graphs. Implementing attention"""

    def __init__(self, heads: int):
        """
        :param heads: Number of attention heads.
        """
        super(AttentionModule, self).__init__()
        self.heads = heads
        self.attn_dropout = nn.Dropout(p=0.0)

    def __repr__(self):
        return f"AttentionModule(heads={self.heads})"

    def vectorize_dict(self, data_dict):
        """
        Vectorize data in the data_dict and concatenate them together.
        :param data_dict:
            'vec': B, m_vec, 3
            'scalar': B, m_s, 1
        :return:
            B, heads, m_vec // heads * 3 + m_s // heads * 1
        """
        container = []
        for key, value in data_dict.items():
            B, m_in, dim = value.shape
            assert m_in % self.heads == 0, 'm_in is not divisible by heads.'
            container.append(value.reshape(B, self.heads, -1))
        return torch.cat(container, dim=-1)

    def forward(self, q: dict, k: dict, v: dict, G, features:dict = None):
        """
        :param q: dict, query
            'vec': B, m_qk_vec, 3
            'scalar': B, m_qk_s, 1
        :param k: dict, key
            'vec': B, m_qk_vec, 3
            'scalar': B, m_qk_s, 1
        :param v: dict, value
            'vec': B, m_v_vec, 3
            'scalar': B, m_V_s, 1
        :param G:
            A DGL graph
        :return: dict
            {'vec': B, m_v_vec, 3,
            'scalar': B, m_v_s, 1}
        """
        with G.local_scope():
            G.ndata['query'] = self.vectorize_dict(q)    # num_nodes, heads, dim
            G.edata['key'] = self.vectorize_dict(k)     # num_edges, heads, dim
            div_term = math.sqrt(G.ndata['query'].shape[-1])

            # Compute the attention weights
            G.apply_edges(fn.e_dot_v('key', 'query', 'attn'))   # num_edges, heads, 1
            attn = G.edata.pop('attn') / div_term  # num_edges, heads, 1
            attn = edge_softmax(G, attn)  # num_edges, heads, 1
            attn = self.attn_dropout(attn)

            # Apply attention weights to value embeddings
            output_dict = {}
            for data_type, data_item in v.items():
                num_edges, m_in, dim = data_item.shape
                assert m_in % self.heads == 0, 'm_in is not divisible by heads in the value embedding.'
                G.edata[data_type] = data_item.reshape(num_edges, self.heads, -1, dim) * attn.unsqueeze(-1)
                G.update_all(fn.copy_e(data_type, 'msg'), fn.sum('msg', data_type))  # num_nodes, heads, m_in/heads, dim
                output_dict[data_type] = G.ndata[data_type].reshape(-1, m_in, dim)
            
            return output_dict


class SO3EquivariantAttenRes(nn.Module):
    """ SO(3) Equivariant Attention Module. Apply SO(3)-Equivariant attentions on the neighborhood of each
    node. It also contains skip self-interaction
    """

    def __init__(self, m_in: dict, m_qk: dict, m_v: dict, m_out: dict, invariant_mod: str, cross_product: bool,
                 edge_dim: int = 0, heads: int = 1, recurrent: bool = False, recur_drop={'vec': 0.0, 'scalar': 0.0},
                 skip_type: str = 'cat', input_LN=False, q_archi: dict = default_MLP_archi,
                 k_archi: dict = default_MLP_archi, v_archi: dict = default_MLP_archi,
                 out_archi: dict = default_MLP_archi):
        """
        :param m_in: dict, channels of input vectors and scalars
        :param m_qk: dict, channels of vectors and scalars in the query and key
        :param m_v: dict, channels of vectors and scalars in the value
        :param m_out: dict, channels of output vectors and scalars
        :param edge_dim: dimension of edge features
        :param heads: number of heads in the attention mechanism
        """
        super(SO3EquivariantAttenRes, self).__init__()
        self.m_in = m_in
        self.m_qk = m_qk
        self.m_v = m_v
        self.m_out = m_out
        self.edge_dim = edge_dim
        self.heads = heads
        self.recurrent = recurrent
        self.skip_type = skip_type
        self.input_LN = input_LN

        if self.input_LN:
            self.input_layer_norm = SO3LayerNorm(m_in)

        self.query_net = SO3EquivariantVector(m_in['vec'], m_qk['vec'], m_in['scalar'], m_qk['scalar'],
                                              invariant_mod, cross_product, net_archi=q_archi)
        self.key_net = PairwiseSO3Conv(m_in['vec'], m_qk['vec'], m_in['scalar'], m_qk['scalar'], invariant_mod,
                                       cross_product, edge_dim=edge_dim, net_archi=k_archi)
        self.value_net = PairwiseSO3Conv(m_in['vec'], m_v['vec'], m_in['scalar'], m_v['scalar'], invariant_mod,
                                         cross_product, edge_dim=edge_dim, net_archi=v_archi)
        self.attn_mod = AttentionModule(heads)

        if self.skip_type == 'cat':
            self.skip_module = SkipCat()
            self.out_net = SO3EquivariantVector(m_in['vec'] + m_v['vec'], m_out['vec'], m_in['scalar'] + m_v['scalar'],
                                                m_out['scalar'], invariant_mod, cross_product, net_archi=out_archi)
        elif self.skip_type == 'sum':
            self.skip_module = SkipSum()
            self.out_net = SO3EquivariantVector(m_v['vec'], m_out['vec'], m_v['scalar'], m_out['scalar'], invariant_mod,
                                                cross_product, net_archi=out_archi)
        elif self.skip_type == 'gate':
            self.skip_module = SkipGate(m_in, m_v)
            self.out_net = SO3EquivariantVector(m_v['vec'], m_out['vec'], m_v['scalar'], m_out['scalar'], invariant_mod,
                                                cross_product, net_archi=out_archi)
        elif self.skip_type == 'none':
            self.skip_module = None
            self.out_net = SO3EquivariantVector(m_v['vec'], m_out['vec'], m_v['scalar'], m_out['scalar'], invariant_mod,
                                                cross_product, net_archi=out_archi)
        else:
            raise ValueError('Unknown skip type in SO3EquivariantAttenRes: {}'.format(self.skip_type))

        if self.recurrent:
            assert (m_in['vec'] == m_out['vec']) and (m_in['scalar'] == m_out['scalar'])
            self.recurrent_drop = SO3Dropout(m_out, p=recur_drop)

    def __repr__(self):
        return f"SO3EquivariantAtten(m_in={self.m_in}, m_qk={self.m_qk}, m_v={self.m_v}, m_out={self.m_out}, " \
               f"edge_dim={self.edge_dim}, heads={self.heads})"

    def forward(self, features: dict, G):
        """
        :param features: dict
            'vec' - B, m_in['vec'], 3
            'scalar' - B, m_in['scalar'], 1
        :param G: graph
        :return: dict
            'vec' - B, m_out['vec'], 3
            'scalar' - B, m_out['scalar'], 1
        """
        if self.input_LN:
            features = self.input_layer_norm(features)

        queries = self.query_net(features)  # dict {B, m_qk['vec'], 3; B, m_qk['scalar'], 1}
        keys = self.key_net(features, G)  # dict {B, m_qk['vec'], 3; B, m_qk['scalar'], 1}
        values = self.value_net(features, G)  # dict {B, m_v['vec'], 3; B, m_v['scalar'], 1}
        updated_feat = self.attn_mod(queries, keys, values, G, features)  # dict {B, m_v['vec'], 3; B, m_v['scalar'], 1}

        if self.skip_module is not None:
            updated_feat = self.skip_module(features, updated_feat)
        updated_feat = self.out_net(updated_feat)

        if self.recurrent:
            # Maybe add drop out on features. Also, it seems that adding features output from layer norm seems not a
            # good idea
            updated_feat = self.recurrent_drop(updated_feat)
            updated_feat = {key: value + features[key] for key, value in updated_feat.items()}

        return updated_feat


class SkipCat(nn.Module):
    def __init__(self):
        """ Concatenate two features dict.
        """
        super(SkipCat, self).__init__()

    @staticmethod
    def forward(features_1: dict, features_2: dict):
        updated_feat = {}
        for data_type in ['vec', 'scalar']:
            container = []
            if data_type in features_1:
                container.append(features_1[data_type])
            if data_type in features_2:
                container.append(features_2[data_type])
            if len(container) != 0:
                updated_feat[data_type] = torch.cat(container, dim=-2)
        return updated_feat


class SkipSum(nn.Module):
    def __init__(self):
        """ Add two features dict together,
        """
        super(SkipSum, self).__init__()

    @staticmethod
    def forward(features_1: dict, features_2: dict):
        updated_feat = {}
        for data_type in ['vec', 'scalar']:
            tmp_results = 0.
            if data_type in features_1:
                tmp_results += features_1[data_type]
            if data_type in features_2:
                tmp_results += features_2[data_type]
            if not isinstance(tmp_results, float):
                updated_feat[data_type] = tmp_results


gate_archi = {
    'layer': 1,  # number of layers.
    'in_LN': False,
    'hidden': {},
    'final': {
        'init': 'gating',
        'bias': True,
        'act': nn.Sigmoid(),
        'drop': 0.0,
    }
}


class SkipGate(nn.Module):
    def __init__(self, m_in: dict, m_update: dict):
        """ Gated mechanism.
        """
        super(SkipGate, self).__init__()
        self.m_in = m_in
        self.m_update = m_update
        self.gate_map = SO3EquivariantVector(m_in_vec=m_in['vec'], m_out_vec=0, m_in_s=m_in['scalar'],
                                             m_out_s=m_update['vec'] + m_update['scalar'], net_archi=gate_archi)

    def forward(self, features: dict, updated: dict):
        """
        :param features: dict
            'vec': B, m_in['vec'], 3
            'scalar': B, m_in['scalar'], 1
        :param updated: dict
            'vec': B, m_update['vec'], 3
            'scalar': B, m_update['scalar'], 1
        :return:
        """
        output = {}
        gate = self.gate_map(features)['scalar']    # B, m_update['vec'] + m_update['scalar'], 1
        if 'vec' in updated:
            output['vec'] = updated['vec'] * gate[:, :self.m_update['vec'], :]
        if 'scalar' in updated:
            output['scalar'] = updated['scalar'] * gate[:, self.m_update['vec']:, :]
        return output


class SO3Dropout(nn.Module):
    """SO3 Dropout modules"""

    def __init__(self, m_in: dict, p: dict):
        """
        :param m_in: dict, channels of input vectors and scalars.
        """
        super(SO3Dropout, self).__init__()
        self.m_in = m_in
        self.dropout_rate = p
        self.dropout_mods = nn.ModuleDict()
        self.eps = 1e-12
        for data_type, channel in m_in.items():
            if channel != 0:
                self.dropout_mods[data_type] = nn.Dropout(p=self.dropout_rate[data_type])

    def __repr__(self):
        return f"SO3Dropout(m_in={self.m_in}, dropout_rate={self.dropout_rate})"

    def forward(self, features: dict, **kwargs):
        """
        :param features: dict
            'vec': B, m_in['vec'], 3
            'scalar': B, m_in['scalar'], 1
        :param kwargs:
        :return: dict
            'vec': B, m_in['vec'], 3
            'scalar', B, m_in[scalar'], 1
        """
        new_features = {}
        if 'vec' in self.dropout_mods:
            data_item = features['vec']
            norm = torch.sqrt(torch.sum(torch.square(data_item), dim=-1, keepdim=True) + self.eps)  # B, m_in[*], 1
            phase = data_item / norm  # B, m_in[*], dim
            transformed = self.dropout_mods['vec'](norm)  # B, m_in[*], 1
            new_features['vec'] = transformed * phase  # B, m_in[*], dim
        if 'scalar' in self.dropout_mods:
            data_item = features['scalar']  # B, m_in[*], dim
            new_features['scalar'] = self.dropout_mods['scalar'](data_item)  # B, m_in[*], dim
        return new_features


class NormBias(nn.Module):
    """Norm-based SO(3)-equivariant nonlinearity with only learned biases."""

    def __init__(self, m_in: dict, non_lin=nn.ReLU(), shifted: dict = {'vec': 'none', 'scalar': 'none'},
                 init: dict = {'vec': 'rand', 'scalar': 'rand'}):
        """
        :param m_in: dict, channels of input vectors and scalars
        :param non_lin: non-linearity
        :param shifted: normalization technique, ['LN', 'BN', or 'none']
        """
        super(NormBias, self).__init__()
        for key, value in shifted.items(): assert value in ['LN', 'none', 'BN'], 'Unresolved shifted type' + key
        for key, value in init.items(): assert value in ['rand', 'zero'], 'Unresolved shifted type ' + key

        self.m_in = m_in
        self.bias = nn.ParameterDict()
        self.shift_module = nn.ModuleDict()
        self.eps = 1e-12
        self.non_lin = non_lin
        self.shift_type = shifted

        for data_type, channel in m_in.items():
            if channel != 0:
                if init[data_type] == 'rand':
                    self.bias[data_type] = nn.Parameter(torch.rand(1, channel), requires_grad=True)
                else:
                    self.bias[data_type] = nn.Parameter(torch.zeros(1, channel), requires_grad=True)
                if shifted[data_type] == 'LN':
                    self.shift_module[data_type] = nn.LayerNorm(channel)
                elif shifted[data_type] == 'BN':
                    self.shift_module[data_type] = nn.BatchNorm1d(channel)

    def __repr__(self):
        return f"NormBias(m_in={self.m_in}, non_lin={self.non_lin}, shifted={self.shifted})"

    def forward(self, features: dict, **kwargs):
        """
        :param features: dict
            'vec': B, m_in['vec'], 3
            'scalar': B, m_in['scalar'], 1
        :param kwargs:
        :return: dict
            'vec': B, m_in['vec'], 3
            'scalar', B, m_in[scalar'], 1
        """
        new_features = {}
        if 'vec' in self.bias:
            data_item = features['vec']
            norm = torch.sqrt(torch.sum(torch.square(data_item), dim=-1) + self.eps)  # B, m_in[*]
            phase = data_item / norm.unsqueeze(-1)  # B, m_in[*], dim
            if self.shift_type != 'none':
                norm = self.shift_module['vec'](norm)
            transformed = self.non_lin(norm + self.bias['vec']).unsqueeze(-1)  # B, m_in[*], 1
            new_features['vec'] = transformed * phase  # B, m_in[*], dim
        if 'scalar' in self.bias:
            data_item = features['scalar'][..., 0]  # B, m_in[*]
            if self.shift_type != 'none':
                data_item = self.shift_module['scalar'](data_item)
            data_item = self.non_lin(data_item + self.bias['scalar'])
            new_features['scalar'] = data_item.unsqueeze(-1)
        return new_features


class SO3LayerNorm(nn.Module):
    def __init__(self, m_in):
        """
        :param m_in: dict, channels of input vectors and scalars
        """
        super(SO3LayerNorm, self).__init__()

        self.m_in = m_in
        self.LN_modules = nn.ModuleDict()
        self.eps = 1e-12

        for data_type, channel in m_in.items():
            if channel != 0:
                self.LN_modules[data_type] = nn.LayerNorm(channel)

    def forward(self, features, **kwargs):
        """
               :param features: dict
                   'vec': B, m_in['vec'], 3
                   'scalar': B, m_in['scalar'], 1
               :param kwargs:
               :return: dict
                   'vec': B, m_in['vec'], 3
                   'scalar', B, m_in[scalar'], 1
               """
        new_features = {}
        if 'vec' in self.LN_modules:
            data_item = features['vec']     # B, m_in[*], dim
            norm = torch.sqrt(torch.sum(torch.square(data_item), dim=-1) + self.eps)  # B, m_in[*]
            phase = data_item / norm.unsqueeze(-1)  # B, m_in[*], dim
            transformed = self.LN_modules['vec'](norm)  # B, m_in[*]
            new_features['vec'] = transformed.unsqueeze(-1) * phase  # B, m_in[*], dim
        if 'scalar' in self.LN_modules:
            data_item = features['scalar'][..., 0]  # B, m_in[*]
            data_item = self.LN_modules['scalar'](data_item)    # B, m_in[*]
            new_features['scalar'] = data_item.unsqueeze(-1)    # B, m_in[*], 1

        return new_features
