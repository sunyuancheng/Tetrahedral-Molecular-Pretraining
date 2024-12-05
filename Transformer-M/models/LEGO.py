import logging

import torch
import torch.nn as nn
from fairseq import utils
import numpy as np
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    LayerNorm,
)
from fairseq.utils import safe_hasattr

from ..modules import (
    init_params,
    TransformerMEncoder
)
from ..modules.transformer_m_encoder import TransformerMEncoder


logger = logging.getLogger(__name__)

@register_model("LEGO_pretrain")
class LEGOPretrainModel(FairseqEncoderModel):
    """
    Class for training a Masked Language Model. It also supports an
    additional sentence level prediction if the sent-loss argument is set.
    """

    def __init__(self, args, encoder):
        super().__init__(encoder)
        self.args = args

        # if specified then apply bert initialization on the model. We need
        # to explictly call this to make sure that the output embeddings
        # and projection layers are also correctly initialized
        if getattr(args, "apply_init", False):
            self.apply(init_params)
        self.encoder_embed_dim = args.encoder_embed_dim

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # Arguments related to dropout
        parser.add_argument(
            "--mode-prob", type=str, default="0.2,0.2,0.6", help="probability of {2D+3D, 2D, 3D} mode for joint training"
        )
        parser.add_argument(
            "--add-3d", action='store_true', help="add 3D attention bias"
        )
        parser.add_argument(
            "--no-2d", action='store_true', help="remove 2D encodings"
        )
        parser.add_argument(
            "--num-3d-bias-kernel", type=int, default=128, metavar="D", help="number of kernel in 3D attention bias"
        )
        parser.add_argument(
            "--droppath-prob", type=float, metavar="D", help="stochastic path probability", default=0.0
        )

        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for" " attention weights",
        )
        parser.add_argument(
            "--act-dropout",
            type=float,
            metavar="D",
            help="dropout probability after" " activation in FFN",
        )

        # Arguments related to hidden states and self-attention
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )

        # Arguments related to input and output embeddings
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--share-encoder-input-output-embed",
            action="store_true",
            help="share encoder input" " and output embeddings",
        )
        parser.add_argument(
            "--encoder-learned-pos",
            action="store_true",
            help="use learned positional embeddings in the encoder",
        )
        parser.add_argument(
            "--no-token-positional-embeddings",
            action="store_true",
            help="if set, disables positional embeddings" " (outside self attention)",
        )
        parser.add_argument(
            "--num-segment", type=int, metavar="N", help="num segment in the input"
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )

        # Arguments related to sentence level prediction
        parser.add_argument(
            "--sentence-class-num",
            type=int,
            metavar="N",
            help="number of classes for sentence task",
        )
        parser.add_argument(
            "--sent-loss",
            action="store_true",
            help="if set," " calculate sentence level predictions",
        )

        # Arguments related to parameter initialization
        parser.add_argument(
            "--apply-init",
            action="store_true",
            help="use custom param initialization for Transformer-M",
        )

        # misc params
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="Which activation function to use for pooler layer.",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )

    def max_positions(self):
        return self.encoder.max_positions

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        base_architecture(args)

        if not safe_hasattr(args, "max_positions"):
            try:
                args.max_positions = args.tokens_per_sample
            except:
                args.max_positions = args.max_nodes

        logger.info(args)

        encoder = LEGOModel(args)

        return cls(args, encoder)

    def forward(self, batched_data, **kwargs):
        return self.encoder(batched_data, **kwargs)



class LEGOModel(FairseqEncoder):

    def __init__(self, args):
        super().__init__(dictionary=None)
        self.max_positions = args.max_positions

        self.molecule_encoder = LEGOModelEncoder(
            num_atoms=args.num_atoms,
            num_in_degree=args.num_in_degree,
            num_out_degree=args.num_out_degree,
            num_edges=args.num_edges,
            num_spatial=args.num_spatial,
            num_edge_dis=args.num_edge_dis,
            edge_type=args.edge_type,
            multi_hop_max_dist=args.multi_hop_max_dist,
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            max_seq_len=self.max_positions,
            num_segments=args.num_segment,
            use_position_embeddings=not args.no_token_positional_embeddings,
            encoder_normalize_before=args.encoder_normalize_before,
            apply_init=args.apply_init,
            activation_fn=args.activation_fn,
            learned_pos_embedding=args.encoder_learned_pos,
            sandwich_ln=args.sandwich_ln,
            droppath_prob=args.droppath_prob,
            add_3d=args.add_3d,
            num_3d_bias_kernel=args.num_3d_bias_kernel,
            no_2d=args.no_2d,
            mode_prob=args.mode_prob,
        )


        self.embed_out = None
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        self.proj_out = None

        self.lm_head_transform_weight = nn.Linear(
            args.encoder_embed_dim, args.encoder_embed_dim
        )
        self.activation_fn = utils.get_activation_fn(args.activation_fn)
        self.layer_norm = LayerNorm(args.encoder_embed_dim)


        self.embed_out = nn.Linear(
            args.encoder_embed_dim, 1, bias=False
        )

    def forward(self, batched_data, perturb=None, segment_labels=None, masked_tokens=None, **unused):

        inner_states, atom_output = self.molecule_encoder(
            batched_data,
            segment_labels=segment_labels,
            perturb=perturb,
        )

        x = inner_states[-1].transpose(0, 1)

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))
        x = self.embed_out(x)
        x = x + self.lm_output_learned_bias

        return x, atom_output, {
            "inner_states": inner_states,
        }

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


class LEGOModelEncoder(TransformerMEncoder):
    """
    Most parts are same with TransformerMEncoder.
    Comment Line 292 - 296.
    In LEGO, all the graphs are in 2D&3D mode.
    Adjust the 2D and 3D mask: stay None in training.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, batched_data, perturb=None, segment_labels: torch.Tensor=None,
                last_state_only: bool=False, position=None, token_embeddings=None,
                attn_mask=None):
        
        data_x = batched_data["x"]
        n_mol, n_atom = data_x.size()[:2]
        padding_mask = (data_x[:,:,0]).eq(0) # B x T x 1
        padding_mask_cls = torch.zeros(n_mol, 1, device=padding_mask.device, dtype=padding_mask.dtype)
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        # B x (T+1) x 1
        mask_dict = {0: [1, 1], 1: [1, 0], 2: [0, 1]}
        mask_2d = mask_3d = None
        # if self.training:
        #     mask_choice = np.random.choice(np.arange(3), n_mol, p=self.mode_prob)
        #     mask = torch.tensor([mask_dict[i] for i in mask_choice]).to(batched_data['pos'])
        #     mask_2d = mask[:, 0]
        #     mask_3d = mask[:, 1]

        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.atom_feature(batched_data, mask_2d=mask_2d)

        if perturb is not None:
            x[:, 1:, :] += perturb

        # x: B x T x C

        attn_bias = self.molecule_attn_bias(batched_data, mask_2d=mask_2d)

        delta_pos = None
        if self.molecule_3d_bias is not None and not (batched_data["pos"] == 0).all():
            attn_bias_3d, merged_edge_features, delta_pos = self.molecule_3d_bias(batched_data)
            if mask_3d is not None:
                merged_edge_features, delta_pos = merged_edge_features * mask_3d[:, None, None], delta_pos * mask_3d[:, None, None, None]
                attn_bias_3d = attn_bias_3d.masked_fill_(((attn_bias_3d != float('-inf')) * (1 - mask_3d[:, None, None, None])).bool(), 0.0)
            attn_bias[:, :, 1:, 1:] = attn_bias[:, :, 1:, 1:] + attn_bias_3d
            x[:, 1:, :] = x[:, 1:, :] + merged_edge_features * 0.01

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask, self_attn_bias=attn_bias)
            if not last_state_only:
                inner_states.append(x)

        atom_output = None
        if delta_pos is not None:
            atom_output = self.atom_proc(x[1:, :, :], attn_bias[:, :, 1:, 1:], delta_pos)
            if mask_3d is not None:
                mask_3d_only = (mask == torch.tensor([0.0, 1.0]).to(mask)[None, :]).all(dim=-1)
                atom_output = atom_output * mask_3d_only[:, None, None]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), atom_output
        else:
            return inner_states, atom_output



@register_model_architecture("LEGO_pretrain", "lego")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.0)

    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.num_segment = getattr(args, "num_segment", 2)

    args.sentence_class_num = getattr(args, "sentence_class_num", 2)
    args.sent_loss = getattr(args, "sent_loss", False)

    args.apply_init = getattr(args, "apply_init", False)

    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)

    args.sandwich_ln = getattr(args, "sandwich_ln", False)
    args.droppath_prob = getattr(args, "droppath_prob", 0.0)


@register_model_architecture("LEGO_pretrain", "lego_base")
def bert_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.share_encoder_input_output_embed = getattr(
        args, "share_encoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.num_segment = getattr(args, "num_segment", 2)

    args.encoder_layers = getattr(args, "encoder_layers", 12)

    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)

    args.sentence_class_num = getattr(args, "sentence_class_num", 2)
    args.sent_loss = getattr(args, "sent_loss", False)

    args.apply_init = getattr(args, "apply_init", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.sandwich_ln = getattr(args, "sandwich_ln", False)
    args.droppath_prob = getattr(args, "droppath_prob", 0.0)

    args.add_3d = getattr(args, "add_3d", False)
    args.num_3d_bias_kernel = getattr(args, "num_3d_bias_kernel", 128)
    args.no_2d = getattr(args, "no_2d", False)
    args.mode_prob = getattr(args, "mode_prob", "0.2,0.2,0.6")
    base_architecture(args)