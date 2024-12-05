# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from copy import deepcopy
from torch_scatter import scatter
from torch_geometric.utils import to_dense_batch, degree
from torch_geometric.data import Data, Batch

NODE_MASK_TOKEN = torch.arange(0, 9 * 512, 512, dtype=torch.long)
EDGE_MASK_TOKEN = torch.arange(0, 3 * 512, 512, dtype=torch.long).unsqueeze(0).repeat(5,1)

def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros(
            [padlen, padlen], dtype=x.dtype).fill_(float('-inf'))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)

def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)

def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)

def pad_pos_unsqueeze(x, padlen):
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)

@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(-1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x

def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
              item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y,
              ) for item in items]
    idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat([pad_3d_unsqueeze(
        i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                        for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                          for i in in_degrees])

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree, # for undirected graph
        x=x,
        edge_input=edge_input,
        y=y,
    )

ratio_convert = {0.05:0.1, 0.1:0.2, 0.15:0.3, 0.2:0.45, 0.25:0.50, 0.3:0.5, 0.5:0.5}
def lego_collator(items, 
                  max_node=512,
                  multi_hop_max_dist=20,
                  spatial_pos_max=20,
                  noise_scale=0.05,
                  perturb_ratio=0.1, 
                  perturb_strategy="random"):
        assert perturb_strategy in ["random", "lego", "lego_maskonly", "lego_noiseonly"]

        if perturb_strategy == "random":
            # local structure ratio convert to all atom ratio
            perturb_ratio = ratio_convert[perturb_ratio]
        else:
            pass
        
        item_res = []
        tg_data_list = []
        for item in items:
            if item is not None and item.x.size(0) <= max_node:
                item_res.append(item)
                tg_data_list.append(Data(x=item.x,edge_index=item.edge_index))
        tg_batch = Batch.from_data_list(tg_data_list)
        # items = [
            # item for item in items if item is not None and item.x.size(0) <= max_node]

        items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
                item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y, item.pos, item.edge_index
                ) for item in item_res]
        idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys, poses, _ = zip(*items)

        for idx, _ in enumerate(attn_biases):
            attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
        max_node_num = max(i.size(0) for i in xs)
        max_dist = max(i.size(-2) for i in edge_inputs)
        y = torch.cat(ys)
        x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
        edge_input = torch.cat([pad_3d_unsqueeze(
            i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
        attn_bias = torch.cat([pad_attn_bias_unsqueeze(
            i, max_node_num + 1) for i in attn_biases])
        attn_edge_type = torch.cat(
            [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
        spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                            for i in spatial_poses])
        in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                            for i in in_degrees])

        label_pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])
        noisy_pos = deepcopy(label_pos)

        node_type_edges = []
        for idx in range(len(items)):
            node_atom_type = items[idx][6][:, 0]
            n_nodes = items[idx][6].shape[0]            
            node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
            node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
            node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
            node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
            node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
            node_atom_edge = convert_to_single_emb(node_atom_edge)

            node_type_edges.append(node_atom_edge.long())
        node_type_edge = torch.cat(node_type_edges)

        
        # mask local structures
        bsz = len(ys)
        in_degrees_batch = torch.cat(in_degrees)
        if perturb_strategy.startswith("lego"):
            non_terminal_atoms_mask = in_degrees_batch > 1
            # non_terminal_atoms_batch = torch.where(non_terminal_atoms_mask == True)[0]
            non_terminal_atoms_batch = torch.arange(in_degrees_batch.size(0))[non_terminal_atoms_mask]
            # non_terminal_atoms_per_graph = tg_batch.batch[non_terminal_atoms_mask].bincount()
            # -> this line will cause error, if the length of last bin is 0, leading to len(non_terminal_atoms_per_graph) != bsz
            non_terminal_atoms_per_graph = scatter(non_terminal_atoms_mask.to(torch.int64), tg_batch.batch, reduce="sum")

            ls_perturbed_per_graph = torch.max(torch.ones(bsz), (non_terminal_atoms_per_graph * perturb_ratio).to(torch.int64)).to(torch.int64)
            central_mask_on_non_terminal = torch.zeros_like(non_terminal_atoms_batch).bool()
            accum = 0
            for i, (h, n) in enumerate(zip(non_terminal_atoms_per_graph, ls_perturbed_per_graph)):
                sample_id_batch_i = torch.randperm(h)[:n]
                central_mask_on_non_terminal[sample_id_batch_i + accum] = True
                accum += non_terminal_atoms_per_graph[i].item()
            
            central_mask_in_batch = torch.zeros_like(in_degrees_batch).bool()
            central_mask_in_batch[non_terminal_atoms_mask] = central_mask_on_non_terminal
            central_atoms = torch.arange(in_degrees_batch.size(0))[central_mask_in_batch]

        else:
            atoms_per_graph = tg_batch.batch.bincount()
            atoms_perturbed_per_graph = torch.max(torch.ones(bsz), (atoms_per_graph * perturb_ratio).to(torch.int64)).to(torch.int64)
            central_mask = torch.zeros_like(tg_batch.batch).bool()
            accum = 0
            for i, (h, n) in enumerate(zip(atoms_per_graph, atoms_perturbed_per_graph)):
                sample_id_batch_i = torch.randperm(h)[:n]
                central_mask[sample_id_batch_i + accum] = True
                accum += atoms_per_graph[i].item()
            central_atoms = torch.arange(in_degrees_batch.size(0))[central_mask]

        masked_edges_index = torch.cat([torch.where(tg_batch.edge_index[0,:]==central_atoms[i])[0] for i in range(central_atoms.size(0))], dim=0)
        masked_edges = tg_batch.edge_index[:,masked_edges_index]
        leaf_atoms = masked_edges[1]

        all_mask_in_batch = torch.zeros_like(central_mask_in_batch).bool()
        all_mask_in_batch[torch.unique(torch.cat([leaf_atoms, central_atoms]))] = True
        
        padded_atom_mask_central, _ = to_dense_batch(central_mask_in_batch, tg_batch.batch)
        padded_atom_mask_all, _ = to_dense_batch(all_mask_in_batch, tg_batch.batch)

        # padded_atom_mask_central = torch.zeros([bsz, max_node_num], dtype=torch.bool)
        # padded_atom_mask_all = torch.zeros([bsz, max_node_num], dtype=torch.bool)
        # masked_edges = torch.tensor([], dtype=torch.int64)
        # for i in range(len(items)):
        #     central_idx, leaf_idx, masked_edges_i = get_central_atom(num_nodes=xs[i].size(0), edge_index=edge_indexs[i], in_degree=in_degrees[i], ratio=self.perturb_ratio, strategy=self.perturb_strategy)
        #     if central_idx is not None and len(central_idx) > 0: 
        #         padded_atom_mask_central[i, central_idx] = True
        #         padded_atom_mask_all[i, torch.cat([central_idx,leaf_idx], dim=0)] = True

        #         # edge_input[i][masked_edges_i[0], masked_edges_i[1], :, :] = 0
        #         # edge_input[i][masked_edges_i[1], masked_edges_i[0], :, :] = 0

        #         masked_edges = torch.cat([masked_edges, masked_edges_i + i*max_node_num], dim=-1)

        expanded_mask_i = padded_atom_mask_all.unsqueeze(2)
        expanded_mask_j = padded_atom_mask_all.unsqueeze(1)
        padded_edge_mask_all = expanded_mask_i | expanded_mask_j
        if perturb_strategy == "lego_maskonly":
            x[padded_atom_mask_all] = NODE_MASK_TOKEN
            edge_input[padded_edge_mask_all] = EDGE_MASK_TOKEN
        elif perturb_strategy == "lego_noiseonly":
            gaussian_noise = torch.randn_like(noisy_pos) * torch.tensor(noise_scale)
            noisy_pos[padded_atom_mask_all] += gaussian_noise[padded_atom_mask_all]
        else:
            x[padded_atom_mask_all] = NODE_MASK_TOKEN
            edge_input[padded_edge_mask_all] = EDGE_MASK_TOKEN
            # gaussian_noise = torch.randn_like(noisy_pos) * torch.tensor(noise_scale)
            uniform_noise = (torch.rand_like(noisy_pos) * 2 - 1) * torch.tensor(noise_scale)
            noisy_pos[padded_atom_mask_all] += uniform_noise[padded_atom_mask_all] 

        del tg_data_list 
        del tg_batch     
        del central_mask_in_batch
        del all_mask_in_batch
        del padded_atom_mask_all
        del expanded_mask_i
        del expanded_mask_j
        del padded_edge_mask_all

        return dict(
            idx=torch.LongTensor(idxs),
            attn_bias=attn_bias,
            attn_edge_type=attn_edge_type,
            spatial_pos=spatial_pos,
            in_degree=in_degree,
            out_degree=in_degree, # for undirected graph
            x=x,
            edge_input=edge_input,
            y=y,

            label_pos=label_pos,
            node_type_edge=node_type_edge,

            pos=noisy_pos,
            central_mask=padded_atom_mask_central,
            # all_mask=padded_atom_mask_all,
            masked_edges=masked_edges,
        )


def collator_3d(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):

    if not hasattr(items[0], "edge_input"):
        items = [
            item for item in items if item is not None and item.x.size(0) <= 256]

        # for lba prediction task, cause the dataset we processed does not provide edge info
        items = [(item.idx, item.attn_bias, item.x, item.y, item.pos, item.train_mean, item.train_std
            ) for item in items]
        idxs, attn_biases, xs, ys, poses, means, stds = zip(*items)


        # for idx, _ in enumerate(attn_biases):
            # attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')

        max_node_num = max(i.size(0) for i in xs)
        y = torch.cat(ys)
        x = torch.cat([pad_2d_unsqueeze(i.unsqueeze(-1), max_node_num) for i in xs])

        attn_bias = torch.cat([pad_attn_bias_unsqueeze(
            i, max_node_num + 1) for i in attn_biases])

        pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])

        node_type_edges = []
        for idx in range(len(items)):
            node_atom_type = items[idx][2][:]
            n_nodes = items[idx][2].shape[0]
            node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
            node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
            node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
            node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
            node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
            node_atom_edge = convert_to_single_emb(node_atom_edge)

            node_type_edges.append(node_atom_edge.long())
        node_type_edge = torch.cat(node_type_edges)

        new_len = len(items)
        attn_edge_type = torch.zeros([new_len, max_node_num, max_node_num, 3])
        spatial_pos = torch.zeros([new_len, max_node_num, max_node_num])
        in_degree = torch.zeros([new_len, max_node_num])
        edge_input = torch.zeros([new_len, max_node_num, max_node_num, 5, 3])

        return dict(
            idx=torch.LongTensor(idxs),
            attn_bias=attn_bias,
            attn_edge_type=attn_edge_type,
            spatial_pos=spatial_pos,
            in_degree=in_degree,
            out_degree=in_degree, # for undirected graph
            x=x,
            edge_input=edge_input,
            y=y,

            pos=pos,
            node_type_edge=node_type_edge,
            mean=means[0],
            std=stds[0]
        )
    
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    if hasattr(items[0], "train_mean"):
        items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
                item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y, item.pos, item.train_mean, item.train_std
                ) for item in items]
        idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys, poses, means, stds = zip(*items)
    else:
        items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
                item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y, item.pos
                ) for item in items]
        idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys, poses = zip(*items)
        means, stds = [None], [None]

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat([pad_3d_unsqueeze(
        i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                        for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                          for i in in_degrees])

    pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])

    node_type_edges = []
    for idx in range(len(items)):

        node_atom_type = items[idx][6][:, 0]
        n_nodes = items[idx][6].shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)

        node_type_edges.append(node_atom_edge.long())
    node_type_edge = torch.cat(node_type_edges)

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree, # for undirected graph
        x=x,
        edge_input=edge_input,
        y=y,

        pos=pos,
        node_type_edge=node_type_edge,
        mean=means[0],
        std=stds[0]
    )

def collator_3d_qm9(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
              item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y, item.pos,
              item.train_mean, item.train_std, item.type

              ) for item in items]
    idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys, poses, means, stds, types = zip(*items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)
    y = torch.cat(ys)
    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    edge_input = torch.cat([pad_3d_unsqueeze(
        i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                        for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                          for i in in_degrees])

    pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])

    node_type_edges = []
    for idx in range(len(items)):

        node_atom_type = items[idx][6][:, 0]
        n_nodes = items[idx][6].shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)

        node_type_edges.append(node_atom_edge.long())
    node_type_edge = torch.cat(node_type_edges)

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree, # for undirected graph
        x=x,
        edge_input=edge_input,
        y=y,

        pos=pos,
        node_type_edge=node_type_edge,

        mean=means[0],
        std=stds[0],
        type=types[0],
    )


def collator_3d_md17(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    items = [(item.idx, item.attn_bias, item.x, item.y, item.pos, item.dy) for item in items]
    idxs, attn_biases,xs, ys, poses, dys = zip(*items)

    max_node_num = max(i.size(0) for i in xs)
    y = torch.cat(ys)
    dy = torch.cat(dys)
    # x = torch.cat([pad_2d_unsqueeze(i.unsqueeze(-1), max_node_num) for i in xs])
    x = torch.cat([(i+1).unsqueeze(-1).unsqueeze(0) for i in xs], dim=0)

    # attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        # i, max_node_num + 1) for i in attn_biases])
    attn_bias = torch.cat([i.unsqueeze(0) for i in attn_biases])

    # pos = torch.cat([pad_pos_unsqueeze(i, max_node_num) for i in poses])
    pos = torch.cat([i.unsqueeze(0) for i in poses])

    node_type_edges = []
    for idx in range(len(items)):

        node_atom_type = items[idx][2]
        n_nodes = items[idx][2].shape[0]
        node_atom_i = node_atom_type.unsqueeze(-1).repeat(1, n_nodes)
        node_atom_i = pad_spatial_pos_unsqueeze(node_atom_i, max_node_num).unsqueeze(-1)
        node_atom_j = node_atom_type.unsqueeze(0).repeat(n_nodes, 1)
        node_atom_j = pad_spatial_pos_unsqueeze(node_atom_j, max_node_num).unsqueeze(-1)
        node_atom_edge = torch.cat([node_atom_i, node_atom_j], dim=-1)
        node_atom_edge = convert_to_single_emb(node_atom_edge)

        node_type_edges.append(node_atom_edge.long())
    node_type_edge = torch.cat(node_type_edges)

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=None,
        spatial_pos=None,
        in_degree=None,
        out_degree=None, # for undirected graph
        x=x,
        edge_input=None,
        y=y,
        dy=dy,

        pos=pos,
        node_type_edge=node_type_edge,
    )


def get_central_atom(num_nodes, edge_index, in_degree, ratio, strategy="random"):
    """
    Randomly pick non-terminal/central atoms to form local structures in a molecule.
    """
    if num_nodes == 1:
        # no mask for CH4
        return None, None, None

    if strategy == "random":
        ls_perturbed = max(1, int(num_nodes * ratio))
        perturb_idx = torch.randperm(num_nodes)[:ls_perturbed]
        return perturb_idx, perturb_idx, []

    ls_perturbed = max(1, int(num_nodes * ratio))
    terminal_atoms = in_degree <= 1
    non_terminal_atoms = torch.arange(num_nodes)[~terminal_atoms]
    # terminal_masked = torch.sum(terminal_atoms) > num_masked
    cnt = 0
    perm_list = non_terminal_atoms[torch.randperm(non_terminal_atoms.numel())].tolist()
    central_idx, leaf_idx = [], []
    masked_edges = torch.tensor([], dtype=torch.int64)
    while cnt < ls_perturbed and len(perm_list) > 0:
        m = perm_list[0]
        central_idx.append(m)
        perm_list.remove(m)

        e0 = torch.where(edge_index[0] == m)[0]
        m_e = edge_index[:, e0]
        leafs = m_e[1,:]
        masked_edges = torch.cat([masked_edges, m_e], dim=1)

        for leaf in leafs: 
            try:
                perm_list.remove(leaf)
            except:
                pass
        leaf_idx = leaf_idx + list(leafs.tolist())  # one-element tensor t.tolist() will lead to a number, not a list
        cnt += 1  
    leaf_idx = list(set(leaf_idx))

    return torch.tensor(central_idx, dtype=torch.int64), torch.tensor(leaf_idx, dtype=torch.int64), masked_edges