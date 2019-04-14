import math

import torch
import torch.nn as nn
import utils
import torch.nn.functional as F

class ConvEncoder(nn.Module):
    """Convolutional encoder"""
    def __init__(
        self, dictionary, embed_dim=512, max_positions=1024,
        convolutions=((512, 3),) * 20, dropout=0.1, attention=False,
        attention_nheads=1, left_pad=True,
    ):
        super().__init__()
        self.dictionary = dictionary
        self.dropout = dropout
        self.num_attention_layers = None
        self.left_pad = left_pad

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        self.embed_positions = PositionalEmbedding(
            max_positions,
            embed_dim,
            self.padding_idx,
            left_pad=self.left_pad,
        )

        def expand_bool_array(val):
            if isinstance(val, bool):
                # expand True into [True, True, ...] and do the same with False
                return [val] * len(convolutions)
            return val

        attention = expand_bool_array(attention)

        in_channels = convolutions[0][0]
        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.attproj = nn.ModuleList()
        for i, (out_channels, kernel_size) in enumerate(convolutions):
            self.projections.append(
                Linear(in_channels, out_channels) if in_channels != out_channels else None
            )
            self.convolutions.append(
                ConvTBC(in_channels, out_channels * 2, kernel_size, dropout=dropout)
            )

            self.attention.append(
                SelfAttention(out_channels, embed_dim, attention_nheads) if attention[i] else None
            )
            in_channels = out_channels

        self.fc2 = Linear(in_channels, embed_dim)

    def forward(self, src_tokens, src_lengths):
        # embed tokens and positions
        x = self.embed_tokens(src_tokens) + self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        input_embedding = x.transpose(0, 1)

        # project to size of convolution
        x = self.fc1(x)

        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()  # -> T x B
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # temporal convolutions
        for proj, conv, attention in zip(self.projections, self.convolutions, self.attention):
            residual = x if proj is None else proj(x)

            if encoder_padding_mask is not None:
                x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

            x = F.dropout(x, p=self.dropout, training=self.training)
            padding_l = (conv.kernel_size[0] - 1) // 2
            padding_r = conv.kernel_size[0] // 2
            x = F.pad(x, (0, 0, 0, 0, padding_l, padding_r))
            x = conv(x)
            x = F.glu(x, dim=2)
            if attention is not None:
                x = attention(x)
            x = (x + residual) * math.sqrt(0.5)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # project back to size of embedding
        x = self.fc2(x)

        if encoder_padding_mask is not None:
            encoder_padding_mask = encoder_padding_mask.t()  # -> B x T
            x = x.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

        # scale gradients (this only affects backward, not forward)
        x = GradMultiply.apply(x, 1.0 / (2.0 * self.num_attention_layers))

        # add output to input embedding for attention
        y = (x + input_embedding.transpose(0, 1)) * math.sqrt(0.5)

        return {
            'encoder_out': (x, y),
            'encoder_padding_mask': encoder_padding_mask,  # B x T
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        encoder_out['encoder_out'] = tuple(
            eo.index_select(0, new_order) for eo in encoder_out['encoder_out']
        )

        if encoder_out['encoder_padding_mask'] is not None:
            encoder_out['encoder_padding_mask'] = \
                encoder_out['encoder_padding_mask'].index_select(0, new_order)

        if 'pretrained' in encoder_out:
            encoder_out['pretrained']['encoder_out'] = tuple(
                eo.index_select(0, new_order)
                for eo in encoder_out['pretrained']['encoder_out']
            )

        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class ConvDecoder(nn.Module):
    """Convolutional decoder"""
    def __init__(
        self, dictionary, embed_dim=512, out_embed_dim=256, max_positions=1024,
        convolutions=((512, 3),) * 8, attention=True, dropout=0.1,
        selfattention=False, attention_nheads=1, selfattention_nheads=1,
        project_input=False, gated_attention=False, downsample=False,
        pretrained=False, trained_decoder=None, left_pad=False,
    ):
        super().__init__()
        self.dictionary = dictionary
        self.onnx_trace = False
        self.register_buffer('version', torch.Tensor([2]))
        self.pretrained = pretrained
        self.pretrained_decoder = trained_decoder
        self.dropout = dropout
        self.left_pad = left_pad
        self.need_attn = True
        in_channels = convolutions[0][0]

        def expand_bool_array(val):
            if isinstance(val, bool):
                # expand True into [True, True, ...] and do the same with False
                return [val] * len(convolutions)
            return val

        attention = expand_bool_array(attention)
        selfattention = expand_bool_array(selfattention)

        if not isinstance(attention, list) or len(attention) != len(convolutions):
            raise ValueError('Attention is expected to be a list of booleans of '
                             'length equal to the number of layers.')

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)

        self.embed_positions = PositionalEmbedding(
            max_positions,
            embed_dim,
            padding_idx,
            left_pad=self.left_pad,
        )

        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        self.attention = nn.ModuleList()
        self.selfattention = nn.ModuleList()
        self.attproj = nn.ModuleList()
        for i, (out_channels, kernel_size) in enumerate(convolutions):
            self.projections.append(
                Linear(in_channels, out_channels) if in_channels != out_channels else None
            )
            self.convolutions.append(
                LinearizedConv1d(
                    in_channels, out_channels * 2, kernel_size,
                    padding=(kernel_size - 1), dropout=dropout,
                )
            )

            self.attention.append(
                DownsampledMultiHeadAttention(
                    out_channels, embed_dim, attention_nheads,
                    project_input=project_input, gated=False, downsample=False,
                ) if attention[i] else None
            )

            self.attproj.append(
                Linear(out_channels, embed_dim, dropout=dropout) if attention[i] else None
            )
            self.selfattention.append(
                SelfAttention(
                    out_channels, embed_dim, selfattention_nheads,
                    project_input=project_input, gated=gated_attention,
                    downsample=downsample,
                ) if selfattention[i] else None
            )
            in_channels = out_channels

        self.fc2 = Linear(in_channels, out_embed_dim)
        self.fc3 = Linear(out_embed_dim, num_embeddings, dropout=dropout)

        # model fusion
        if self.pretrained:
            # independent gates are learned from the concatenated input
            self.gate1 = nn.Sequential(Linear(out_embed_dim*2, out_embed_dim), nn.Sigmoid())
            self.gate2 = nn.Sequential(Linear(out_embed_dim*2, out_embed_dim), nn.Sigmoid())
            # pretrained and trained models are joined
            self.joining = nn.Sequential(
                Linear(out_embed_dim*2, out_embed_dim*2),
                LayerNorm(out_embed_dim*2),
                nn.GLU(),
                Linear(out_embed_dim, out_embed_dim*2),
                LayerNorm(out_embed_dim*2),
                nn.GLU(),
                Linear(out_embed_dim, out_embed_dim),
                LayerNorm(out_embed_dim)
            )
            # pretrained model contains an output layer that is nhid -> vocab size
            # but the models are combined in their hidden state
            # the hook stores the output of the pretrained model forward
            self.pretrained_outputs = {}

            def save_output():
                def hook(a, b, output):
                    self.pretrained_outputs["out"] = output
                return hook

            self.pretrained_decoder.fc2.register_forward_hook(save_output())

    def forward(self, prev_output_tokens, encoder_out_dict):
        encoder_out = encoder_out_dict['encoder']['encoder_out']
        trained_encoder_out = encoder_out_dict['pretrained'] if self.pretrained else None

        encoder_a, encoder_b = self._split_encoder_out(encoder_out)

        # embed positions
        positions = self.embed_positions(prev_output_tokens)

        # embed tokens and positions
        x = self.embed_tokens(prev_output_tokens) + positions
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x.transpose(0, 1)

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # temporal convolutions
        avg_attn_scores = None
        for proj, conv, attention, selfattention, attproj in zip(
            self.projections, self.convolutions, self.attention, self.selfattention, self.attproj
        ):
            residual = x if proj is None else proj(x)

            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x)
            x = F.glu(x, dim=2)

            # attention
            if attention is not None:
                r = x
                x, attn_scores = attention(attproj(x) + target_embedding, encoder_a, encoder_b)
                x = x + r
                if not self.training and self.need_attn:
                    if avg_attn_scores is None:
                        avg_attn_scores = attn_scores
                    else:
                        avg_attn_scores.add_(attn_scores)

            if selfattention is not None:
                x = selfattention(x)

            x = (x + residual) * math.sqrt(0.5)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        # project back to size of vocabulary
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        if not self.pretrained:
            x = self.fc3(x)

        # fusion gating
        if self.pretrained:
            trained_x, _ = self.pretrained_decoder.forward(prev_output_tokens, trained_encoder_out)
            y = torch.cat([x, self.pretrained_outputs["out"]], dim=-1)
            gate1 = self.gate1(y)
            gate2 = self.gate2(y)
            gated_x1 = gate1 * x
            gated_x2 = gate2 * self.pretrained_outputs["out"]
            fusion = torch.cat([gated_x1, gated_x2], dim=-1)
            fusion = self.joining(fusion)
            fusion_output = self.fc3(fusion)
            return fusion_output, avg_attn_scores
        else:
            return x, avg_attn_scores

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

    def _split_encoder_out(self, encoder_out):
        """Split and transpose encoder outputs."""
        # transpose only once to speed up attention layers
        encoder_a, encoder_b = encoder_out
        encoder_a = encoder_a.transpose(0, 1).contiguous()
        encoder_b = encoder_b.transpose(0, 1).contiguous()
        result = (encoder_a, encoder_b)
        return result

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def get_normalized_probs(self, net_output, log_probs, sample):
        """Get normalized probabilities (or log probs) from a net's output."""

        if hasattr(self, 'adaptive_softmax') and self.adaptive_softmax is not None:
            if sample is not None:
                assert 'target' in sample
                target = sample['target']
            else:
                target = None
            out = self.adaptive_softmax.get_log_prob(net_output[0], target=target)
            return out.exp_() if not log_probs else out

        logits = net_output[0]
        if log_probs:
            return utils.log_softmax(logits, dim=-1, onnx_trace=self.onnx_trace)
        else:
            return utils.softmax(logits, dim=-1, onnx_trace=self.onnx_trace)

    def upgrade_state_dict(self, state_dict):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict

