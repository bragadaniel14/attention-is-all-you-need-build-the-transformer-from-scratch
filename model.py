"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    ret_dict = {tok:idx for tok,idx in zip(specials, range(len(specials)))}
    count = len(specials)
    for sentence in sentences:
        for token in sentence.split():
            if token not in ret_dict:
                ret_dict[token] = count
                count += 1
    return ret_dict

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    return {idx:tok for tok,idx in token_to_id.items()}

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    return [token_to_id.get(tok,token_to_id[unk_token]) for tok in sentence.split()]

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    return [id_to_token[idx] for idx in ids]

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    new_ids = ids[:max_len]
    return new_ids + [pad_id] * max(0, max_len-len(new_ids))

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    return torch.stack([torch.tensor(sent,dtype=torch.long) for sent in padded_sequences], dim=0)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    return embeddings * math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors
    return torch.pow(10000, torch.tensor([-2*d/d_model for d in range(d_model // 2)]))

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1
    return torch.arange(max_len).view((max_len,1)).float()

# Step 10 - fill_even_indices_with_sin
import torch

def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    pe[:, 0::2] = torch.sin(div_term*position)
    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    pe[:, 1::2] = torch.cos(position*div_term)
    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix
    pe = torch.zeros((max_len, d_model), dtype=torch.float)
    div_term =compute_positional_div_term(d_model)
    position_column = build_position_index_column(max_len)

    pe = fill_even_indices_with_sin(pe, position_column, div_term)
    pe = fill_odd_indices_with_cos(pe, position_column, div_term)

    return pe

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.
    _, L, _ = embedded_batch.shape
    return embedded_batch + positional_encoding[:L, :]

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    mask = token_ids != pad_id
    B, L = token_ids.shape
    return mask.view((B,1,1,L))

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    causal_mask = torch.full((seq_len, seq_len), False)
    row, col = torch.tril_indices(seq_len,seq_len)
    causal_mask[row, col] = True
    return causal_mask.view(1,1, seq_len, seq_len)

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    
    return query @ key.transpose(-2,-1)

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    inf_mask = torch.zeros(mask.shape)
    inf_mask[~mask] = -torch.inf
    scores += inf_mask
    return scores

# Step 20 - softmax_attention_weights
import torch

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf
    softmax = torch.softmax(masked_scores, axis=-1)
    return torch.where(softmax.isnan(), 0, softmax)

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    return attention_weights @ value

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # TODO: chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    raw_scores = compute_raw_attention_scores(query, key)
    scores = scale_attention_scores(raw_scores, query.shape[-1])
    if mask is not None:
        scores = mask_attention_scores_with_neg_inf(scores, mask)
    attention_weights = softmax_attention_weights(scores)

    return apply_attention_weights_to_values(attention_weights, value), attention_weights

    """
    
    new_query = query @  key.transpose(-2,-1)
    new_query /= math.sqrt(query.shape[-1])
    if mask is not None:
        mask_fill = torch.where(mask, 0, -torch.inf)
        new_query += mask_fill
    softmax = torch.softmax(new_query, axis=-1)
    softmax = torch.where(softmax.isnan(), 0, softmax)
    return softmax @ value, softmax
    
    """

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    B,L, d_model = tensor.shape
    return tensor.view(B,L, num_heads, -1)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return split_tensor.transpose(-3,-2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    B,num_heads, L, d_k = multi_head_tensor.shape
    return multi_head_tensor.transpose(-3,-2).reshape(B, L, -1)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    return x @ weight.T + (0 if bias is None else bias)

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    return apply_linear_projection(x, w_q, b_q), apply_linear_projection(x, w_k, b_k), apply_linear_projection(x, w_v, b_v)

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    B, L, d_model = q.shape
    return transpose_heads_before_sequence(split_last_dim_into_heads(q,num_heads)), \
        transpose_heads_before_sequence(split_last_dim_into_heads(k,num_heads)),\
        transpose_heads_before_sequence(split_last_dim_into_heads(v,num_heads))

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    return scaled_dot_product_attention(q_h,k_h,v_h, mask)
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    return apply_linear_projection(merge_heads_back_to_model_dim(context), w_o,b_o)

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    q_h, k_h,v_h = split_qkv_into_heads(query @ w_q, key @ w_k, value @ w_v, num_heads)
    context, weights = multi_head_scaled_dot_product_attention(q_h,k_h,v_h,mask)
    merged_context = merge_heads_and_project_output(context, w_o, None) 
    return merged_context

# Step 32 - apply_ffn_first_linear_and_relu
def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    return torch.relu(x @ w1 + b1)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    return hidden @ w2 + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    return apply_ffn_second_linear(apply_ffn_first_linear_and_relu(x, w1, b1), w2,b2)

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    mean = torch.mean(x, axis=-1, keepdim=True)
    var = ((x-mean)**2).mean(axis=-1, keepdim=True)
    return mean, var

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    mean, var = compute_layer_norm_mean_and_variance(x)
    return ((x-mean)/torch.sqrt(var + eps))* gamma + beta

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    return normalize_and_scale_with_gamma_beta(residual_input+sublayer_output, gamma, beta,eps)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    return x * keep_mask / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    merged_context = assemble_multi_head_attention_forward(x,x,x,w_q,w_k,w_v,w_o,num_heads, src_mask)
    return apply_residual_add_and_norm(x, merged_context, gamma, beta)

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    suboutput = position_wise_feed_forward_network(x,w1,b1,w2,b2)
    return apply_residual_add_and_norm(x, suboutput, gamma, beta)

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    w_q, w_k, w_v, w_o, gamma, beta = layer_params['w_q'],layer_params['w_k'],layer_params['w_v'],layer_params['w_o'],layer_params['attn_gamma'],layer_params['attn_beta']
    output = encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask)
    w1,b1,w2,b2,gamma,beta = layer_params['w1'],layer_params['b1'],layer_params['w2'],layer_params['b2'],layer_params['ffn_gamma'],layer_params['ffn_beta']
    return encoder_layer_feed_forward_sublayer(output,w1,b1,w2,b2,gamma,beta)

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    for params in encoder_layer_params_list:
        x = assemble_encoder_layer(x, params, num_heads, src_mask)
    return x

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    merged_context = assemble_multi_head_attention_forward(y,y,y,w_q,w_k,w_v,w_o,num_heads, tgt_mask)
    return apply_residual_add_and_norm(y, merged_context, gamma, beta)

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    if src_mask is not None and src_mask.dim() == 2:
        B, L = src_mask.shape
        src_mask = src_mask.view(B, 1, 1, L)
    merged_context = assemble_multi_head_attention_forward(y,encoder_output,encoder_output,w_q,w_k,w_v,w_o,num_heads, src_mask)
    return apply_residual_add_and_norm(y, merged_context, gamma, beta)

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    output = position_wise_feed_forward_network(y,w1,b1,w2,b2)
    return apply_residual_add_and_norm(output, y, gamma, beta)

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""
    # TODO: chain the three decoder sublayers using params from layer_params.
    w_q, w_k, w_v, w_o, gamma, beta = layer_params['w_q_self'],layer_params['w_k_self'],layer_params['w_v_self'],layer_params['w_o_self'],layer_params['self_gamma'],layer_params['self_beta']
    y = decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask)
    w_q, w_k, w_v, w_o, gamma, beta = layer_params['w_q_cross'],layer_params['w_k_cross'],layer_params['w_v_cross'],layer_params['w_o_cross'],layer_params['cross_gamma'],layer_params['cross_beta']
    y = decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask)
    w1,b1,w2,b2,gamma,beta = layer_params['w1'],layer_params['b1'],layer_params['w2'],layer_params['b2'],layer_params['ffn_gamma'],layer_params['ffn_beta']
    return decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta)

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
    for params in decoder_layer_params_list:
        y = assemble_decoder_layer(y, encoder_output, params, num_heads, src_mask, tgt_mask)
    return y

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    mult = decoder_output @ output_projection_weight.T + (0 if output_projection_bias is None else output_projection_bias)
    return mult

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    return token_embedding_weight.transpose(0,1)

# Step 50 - apply_log_softmax_over_vocab
def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    return logits - torch.log(torch.sum(torch.exp(logits), axis=-1,keepdim=True))

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    # TODO: embed src+tgt, add PE, build masks, run encoder/decoder, project to log probs.
    src_embedding = model_params.get('src_embedding', model_params.get('token_embedding'))
    tgt_embedding = model_params.get('tgt_embedding', model_params.get('token_embedding'))
  
    encoder_layers = model_params['encoder_layers']
    decoder_layers = model_params['decoder_layers']
    output_projection = model_params['output_projection']

    embed_src = src_embedding[src_ids]
    src_l, d_model = embed_src.shape[1], embed_src.shape[2]
    embed_src = scale_embeddings_by_sqrt_d_model(embed_src, d_model)
    pe = build_sinusoidal_positional_encoding(src_l, d_model)
    embed_src += pe
    
    embed_tgt = tgt_embedding[tgt_ids]
    tgt_l, d_model = embed_tgt.shape[1], embed_tgt.shape[2]
    embed_tgt = scale_embeddings_by_sqrt_d_model(embed_tgt, d_model)
    pe = build_sinusoidal_positional_encoding(tgt_l, d_model)
    embed_tgt += pe

    src_mask = build_padding_mask(src_ids, pad_id)
    tgt_mask = build_padding_mask(tgt_ids, pad_id)
    causal_mask = build_causal_mask(tgt_l)
    tgt_mask = combine_padding_and_causal_masks(tgt_mask, causal_mask)

    encoder_output = stack_encoder_layers(embed_src, encoder_layers, num_heads, src_mask)
    decoder_output = stack_decoder_layers(embed_tgt, encoder_output, decoder_layers, num_heads, src_mask, tgt_mask)

    logits = apply_final_output_projection(decoder_output, output_projection)
    return apply_log_softmax_over_vocab(logits)

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    # TODO: allocate w_q, w_k, w_v, w_o, w1, b1, w2, b2, attn_gamma, attn_beta, ffn_gamma, ffn_beta.
    params = {}
    params['w_q'] =(torch.randn((d_model, d_model))* 0.1).requires_grad_(True)
    params['w_k'] = (torch.randn((d_model, d_model))* 0.1).requires_grad_(True)
    params['w_v'] = (torch.randn((d_model, d_model))* 0.1).requires_grad_(True)
    params['w_o'] = (torch.randn((d_model, d_model))* 0.1).requires_grad_(True)

    params['w1'] = (torch.randn((d_model, d_ff))* 0.1).requires_grad_(True)
    params['w2'] = (torch.randn((d_ff, d_model))* 0.1).requires_grad_(True)
    params['b1'] = torch.zeros(d_ff, requires_grad=True)
    params['b2'] = torch.zeros(d_model, requires_grad=True)

    params['attn_gamma'] = torch.ones(d_model, requires_grad=True)
    params['attn_beta'] = torch.zeros(d_model, requires_grad=True)
    params['ffn_gamma'] = torch.ones(d_model, requires_grad=True)
    params['ffn_beta'] = torch.zeros(d_model, requires_grad=True)

    return params

# Step 53 - init_decoder_layer_parameters
import torch

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    # TODO: return a dict of requires_grad tensors for one decoder layer
    params = {}
    params['w_q_self'] =(torch.randn((d_model, d_model))* 0.1).requires_grad_(True)
    params['w_k_self'] = (torch.randn((d_model, d_model))* 0.1).requires_grad_(True)
    params['w_v_self'] = (torch.randn((d_model, d_model))* 0.1).requires_grad_(True)
    params['w_o_self'] = (torch.randn((d_model, d_model))* 0.1).requires_grad_(True)

    params['w_q_cross'] =(torch.randn((d_model, d_model))* 0.1).requires_grad_(True)
    params['w_k_cross'] = (torch.randn((d_model, d_model))* 0.1).requires_grad_(True)
    params['w_v_cross'] = (torch.randn((d_model, d_model))* 0.1).requires_grad_(True)
    params['w_o_cross'] = (torch.randn((d_model, d_model))* 0.1).requires_grad_(True)

    params['w1'] = (torch.randn((d_model, d_ff))* 0.1).requires_grad_(True)
    params['w2'] = (torch.randn((d_ff, d_model))* 0.1).requires_grad_(True)
    params['b1'] = torch.zeros(d_ff, requires_grad=True)
    params['b2'] = torch.zeros(d_model, requires_grad=True)


    params['self_gamma'] = torch.ones(d_model, requires_grad=True)
    params['self_beta'] = torch.zeros(d_model, requires_grad=True)
    params['cross_gamma'] = torch.ones(d_model, requires_grad=True)
    params['cross_beta'] = torch.zeros(d_model, requires_grad=True)
    params['ffn_gamma'] = torch.ones(d_model, requires_grad=True)
    params['ffn_beta'] = torch.zeros(d_model, requires_grad=True)

    return params

# Step 54 - init_embedding_and_projection_parameters
import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # TODO: allocate three (vocab_size, d_model) tensors with requires_grad=True
    src_embedding = (torch.randn((vocab_size,d_model))*0.1).requires_grad_(True)
    tgt_embedding = (torch.randn((vocab_size,d_model))*0.1).requires_grad_(True)
    if tie_weights:
        return {'src_embedding':src_embedding, 
            'tgt_embedding':tgt_embedding,
            "output_projection":tgt_embedding}
    
    return  {'src_embedding':src_embedding, 
        'tgt_embedding':tgt_embedding, 
        "output_projection":(torch.randn((vocab_size,d_model))*0.1).requires_grad_(True)}

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    # TODO: walk the encoder, decoder, and embedding dicts and return a flat deduped list of tensors
    tensors = {}
    for params in encoder_layer_params:
        for obj in params.values():
            tensors[id(obj)] = obj
    for params in decoder_layer_params:
        for obj in params.values():
            tensors[id(obj)] = obj
    for obj in embedding_params.values():
         tensors[id(obj)] = obj
    
    return list(tensors.values())

# Step 56 - shift_targets_right_with_start_token
def shift_targets_right_with_start_token(target_ids, start_token_id):
    # TODO: prepend start_token_id and drop the last column so output shape matches target_ids
    new_targets = torch.empty(target_ids.shape, dtype=torch.int64)
    new_targets[:, 0] = start_token_id
    new_targets[:, 1:] = target_ids[:, :-1]
    return new_targets

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    # TODO: return the Noam warmup learning rate for the given step.
    return (d_model**(-.5))*min((step**(-0.5)), step* warmup_steps**(-1.5))

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # TODO: return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    return torch.full(shape,epsilon/(vocab_size-2))

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    # TODO: write the confidence value at each gold token id along the vocab axis
    output = smoothed_distribution.clone()
    index = gold_token_ids.unsqueeze(-1)
    output.scatter_(dim=-1, index=index, value=confidence)
    return output

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    # TODO: zero the pad column and the rows where the gold token equals pad_id
    smoothed_distribution[:,:,pad_id] = 0
    golden_mask = gold_token_ids == pad_id
    smoothed_distribution[golden_mask] = 0

    return smoothed_distribution

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    # TODO: combine log_probabilities with the smoothed target distribution into a scalar loss
    return (-log_probabilities * smoothed_distribution).sum()

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    # TODO: divide total_loss by the count of non-pad tokens in gold_token_ids
    total = (gold_token_ids != pad_id).sum()
    return total_loss / total if total else total_loss

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    # TODO: argmax over vocab, compare to gold, average over non-pad positions only
    predictions = torch.argmax(log_probabilities, axis=-1)
    mask = gold_token_ids != pad_id
    total = (gold_token_ids[mask] == predictions[mask]).sum()
    return total / mask.sum() if mask.sum() else torch.tensor(0.0,requires_grad=True)

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    # TODO: allocate zero buffers for first and second moments, plus step counter
    m = []
    v = []
    t = 0
    for param in parameter_list:
        m.append(torch.zeros(param.shape))
        v.append(torch.zeros(param.shape))
    return {'m':m, 'v':v, 't':t}

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    # TODO: apply the Adam first-moment EMA update and return the new tensor
    return m_prev * beta1 + (1-beta1) * grad

# Step 66 - update_adam_second_moment
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    # TODO: apply Adam's EMA update for the second moment of the gradient
    return beta2 * v_prev + (1-beta2) * grad**2

# Step 67 - apply_adam_bias_correction
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    # TODO: divide each moment by (1 - beta**step) using its respective beta
    return m_t/(1-beta1**step), v_t/(1-beta2**step)

# Step 69 - apply_adam_step_to_all_parameters
import torch

def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    # TODO: increment t, then for each param with a grad update m, v, bias-correct, and subtract delta in place.
    with torch.no_grad():
        optimizer_state['t'] += 1
        t = optimizer_state['t']
        m = optimizer_state['m']
        v = optimizer_state['v']
        for idx,param in enumerate(parameter_list):
            if param.grad is None:
                continue
            cur_m,cur_v = m[idx], v[idx]
            cur_m=update_adam_first_moment(cur_m,param.grad, beta1)
            cur_v=update_adam_second_moment(cur_v, param.grad, beta2)
            m[idx], v[idx] = cur_m, cur_v
            cur_m, cur_v = apply_adam_bias_correction(cur_m,cur_v,beta1,beta2,t)
            param -= learning_rate*cur_m/(torch.sqrt(cur_v)+epsilon)
        optimizer_state['t'] = t
        optimizer_state['m'] = m
        optimizer_state['v'] = v
    return optimizer_state

# Step 70 - zero_all_parameter_gradients
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    # TODO: clear the accumulated gradient on every parameter tensor in the list
    for param in parameter_list:
        param.grad=None

# Step 71 - compute_batch_training_loss
def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    # TODO: shift targets right, run the forward pass, build smoothed targets, and average the KL loss over non-pad tokens.
    if 'token_embedding' not in model_params:
        model_params['token_embedding'] = model_params.get(
            'tgt_embedding', model_params.get('src_embedding')
        )
    
    new_tgt = shift_targets_right_with_start_token(tgt_batch, config['start_id'])
    log_probabilities = run_transformer_forward(src_batch, new_tgt, model_params,config['num_heads'],config['pad_id'])
    
    confidence = 1 -config['smoothing']
    smoothed = build_uniform_smoothing_distribution(log_probabilities.shape, config['vocab_size'], config['smoothing'])
    smoothed = set_confidence_on_gold_tokens(smoothed, tgt_batch, confidence)
    smoothed = zero_pad_column_and_pad_token_rows(smoothed, tgt_batch,config['pad_id'])

    total_loss = compute_label_smoothed_kl_loss(log_probabilities, smoothed)
    return average_loss_over_non_pad_tokens(total_loss, tgt_batch, config['pad_id'])

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

