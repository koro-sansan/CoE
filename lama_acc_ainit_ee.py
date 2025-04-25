import os
import types
import tqdm
import torch
from datasets import load_dataset, load_from_disk
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers.models.bert.modeling_bert import BertIntermediate,BertSelfOutput,BertAttention,BertOutput,BertSelfAttention
from transformers.models.llama.modeling_llama import LlamaMLP,LlamaSdpaAttention
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import json
from typing import List, Optional, Tuple, Union
import math
from transformers.cache_utils import Cache
from transformers import AutoTokenizer, AutoModelForCausalLM,BloomForCausalLM, pipeline, LlamaConfig, LlamaForCausalLM
e_n = 4
e_k = 4
batch_size = 32
gradient_accumulation_steps = 4
mini_batch_size = batch_size // gradient_accumulation_steps

generation_args = dict(
    do_sample=False,
    max_length=2048,
    repetition_penalty=1.2,
    top_k=10,
    num_beams=2
)

folder = "./coe_070_acc_32_backwindow"
log = "./coe_070_acc_32_backwindow.txt"
ckpt_path = "./acc_best.bin"
train_layer_start = 0#
train_layer_end = 31#11 15 19 23 27 31
aim_acc = 0.70
learning_rate = 8e-5
keep = 8
set_wid = 1024
set_wid2 = 1024

wids = [1024, 1024, 1024, 1024,   1024, 1024, 1024, 1024,   512,480,256,128,   128, 32, 32, 32,   64, 16, 16, 16,   8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]#0.70

com = [0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1] #0.70

eval_gap = 14
train_mode = 'front_window'
#['front_first','back_first','back_window','front_window']
window_size = 16
load = 0
bosave = 1
save_layer = 0 #end
c_FFN = 1
c_att = 1
ainit = 0

tokenizer = AutoTokenizer.from_pretrained('./llama3', padding_side='left')
tokenizer.pad_token = tokenizer.bos_token
config = LlamaConfig.from_pretrained('./llama3')
config.output_hidden_states = True


class route_only(nn.Module):
    def __init__(self, hidden_size, export_num1, export_num2, export_num3):
        super().__init__()
        self.gates1 = nn.Linear(hidden_size, export_num1).cuda()
        self.gates2 = nn.Linear(hidden_size, export_num2).cuda()
        self.gates3 = nn.Linear(hidden_size, export_num3).cuda()

        self.export_num1 = export_num1
        self.export_num2 = export_num2
        self.export_num3 = export_num3
        
    def forward(self, x):
        bsz, seq_len, dim = x.size()
        batch_size, sequence_length, hidden_dim = x.shape
        x = x.view(-1, dim)

        router_logits = self.gates1(x)
        routing_weights_before = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights1, selected_experts = torch.topk(routing_weights_before, e_k, dim=-1)
        routing_weights1 /= routing_weights1.sum(dim=-1, keepdim=True)
        routing_weights1 = routing_weights1.to(x.dtype)
        expert_mask1 = torch.nn.functional.one_hot(selected_experts, num_classes=self.export_num1)
        expert_mask1 = expert_mask1.permute(2, 1, 0)
        
        router_logits = self.gates2(x)
        routing_weights_before = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights2, selected_experts = torch.topk(routing_weights_before, e_k, dim=-1)
        routing_weights2 /= routing_weights2.sum(dim=-1, keepdim=True)
        routing_weights2 = routing_weights2.to(x.dtype)
        expert_mask2 = torch.nn.functional.one_hot(selected_experts, num_classes=self.export_num1)
        expert_mask2 = expert_mask2.permute(2, 1, 0)

        router_logits = self.gates3(x)
        routing_weights_before = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights3, selected_experts = torch.topk(routing_weights_before, e_k, dim=-1)
        routing_weights3 /= routing_weights3.sum(dim=-1, keepdim=True)
        routing_weights3 = routing_weights3.to(x.dtype)
        expert_mask3 = torch.nn.functional.one_hot(selected_experts, num_classes=self.export_num1)
        expert_mask3 = expert_mask3.permute(2, 1, 0)


        return expert_mask1,expert_mask2,expert_mask3,routing_weights1,routing_weights2,routing_weights3

class improved_3part_route_noact_real_moe(nn.Module):
    def __init__(self, hidden_size, output_size, export_num1, export_num2, export_num3, rank0, rank1, rank2, rank3):
        super().__init__()
        self.factors1 = nn.ModuleList([nn.Linear(hidden_size, rank0).cuda() for _ in range(export_num1)])
        self.factors2 = nn.ModuleList([nn.Linear(rank0, rank1).cuda() for _ in range(export_num2)])
        self.factors3 = nn.ModuleList([nn.Linear(rank1, output_size).cuda() for _ in range(export_num2)])

        self.export_num1 = export_num1
        self.export_num2 = export_num2
        self.export_num3 = export_num3
        self.rank0 = rank0
        self.rank1 = rank1
        self.output_size = output_size

    def forward(self, x, expert_mask1, expert_mask2, expert_mask3, routing_weights1, routing_weights2, routing_weights3):
        bsz, seq_len, dim = x.size()
        batch_size, sequence_length, hidden_dim = x.shape
        x = x.view(-1, dim)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, self.output_size), dtype=x.dtype, device=x.device
        )
        current_state1 = torch.zeros(
            (batch_size * sequence_length, self.rank0), dtype=x.dtype, device=x.device
        )
        current_state2 = torch.zeros(
            (batch_size * sequence_length, self.rank1), dtype=x.dtype, device=x.device
        )

        for expert_idx in range(self.export_num1):
            idx, top_x = torch.where(expert_mask1[expert_idx])
            if top_x.shape[0] == 0:
                continue
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            current_state = x[None, top_x_list].reshape(-1, hidden_dim)
            expert_output = self.factors1[expert_idx](current_state)
            current_hidden_states = expert_output * routing_weights1[top_x_list, idx_list, None]
            current_state1.index_add_(0, top_x, current_hidden_states.to(x.dtype))

        for expert_idx in range(self.export_num2):
            idx, top_x = torch.where(expert_mask2[expert_idx])
            if top_x.shape[0] == 0:
                continue
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            current_state = current_state1[None, top_x_list].reshape(-1, self.rank0)
            expert_output = self.factors2[expert_idx](current_state)
            current_hidden_states = expert_output * routing_weights2[top_x_list, idx_list, None]
            current_state2.index_add_(0, top_x, current_hidden_states.to(x.dtype))

        for expert_idx in range(self.export_num3):
            idx, top_x = torch.where(expert_mask3[expert_idx])
            if top_x.shape[0] == 0:
                continue
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()
            current_state = current_state2[None, top_x_list].reshape(-1, self.rank1)
            expert_output = self.factors3[expert_idx](current_state)
            expert_output = torch.relu(expert_output)
            current_hidden_states = expert_output * routing_weights3[top_x_list, idx_list, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(x.dtype))


        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, -1)
        final_hidden_states = torch.relu(final_hidden_states)

        return final_hidden_states

if not os.path.exists(folder):
    os.makedirs(folder)

# t_model = LlamaForCausalLM.from_pretrained('./llama3', config=config)
t_model = LlamaForCausalLM(config=config)
# t_model.load_state_dict(torch.load(ckpt_path, map_location='cpu'), strict=False)
t_model.bfloat16()
t_model.cuda()

# model = LlamaForCausalLM.from_pretrained('./llama3', config=config)
model = LlamaForCausalLM(config=config)

class coedataset(Dataset):
    def __init__(self, json_file_paths):
        self.data = []
        for json_file_path in json_file_paths:
            with open(json_file_path, 'r') as f:
                da = json.load(f)
                self.data.extend(da)

    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        item = self.data[idx] 
        return item
    def remove(self, it):
        self.data.remove(it)

def dropout_add(x: torch.Tensor, residual: torch.Tensor, prob: float, training: bool) -> torch.Tensor:
    """
    Dropout add function

    Args:
        x (`torch.tensor`, *required*):
            input tensor
        residual (`torch.tensor`, *required*):
            residual tensor
        prob (`float`, *required*):
            dropout probability
        training (`bool`, *required*):
            training mode
    """
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def change_forward(model, e_n=4):
    def _forward(self, x):
        expert_mask1,expert_mask2,expert_mask3,routing_weights1,routing_weights2,routing_weights3 = self.router(x)
        if self.idx >= train_layer_start and self.idx <= train_layer_end and com[self.idx] == 1:
            down_proj = self.mlp2(self.act_fn(self.mlp3(x,expert_mask1, expert_mask2, expert_mask3, routing_weights1, routing_weights2, routing_weights3)) * self.mlp1(x, expert_mask1, expert_mask2, expert_mask3, routing_weights1, routing_weights2, routing_weights3), expert_mask1, expert_mask2, expert_mask3, routing_weights1, routing_weights2, routing_weights3)            
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj
    def _forward3(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        expert_mask1,expert_mask2,expert_mask3,routing_weights1,routing_weights2,routing_weights3 = self.router(hidden_states)
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()
        if self.idx >= train_layer_start and self.idx <= train_layer_end and com[self.idx] == 1:
            query_states = self.q(hidden_states, expert_mask1, expert_mask2, expert_mask3, routing_weights1, routing_weights2, routing_weights3)
            key_states = self.k(hidden_states, expert_mask1, expert_mask2, expert_mask3, routing_weights1, routing_weights2, routing_weights3)
            value_states = self.v(hidden_states, expert_mask1, expert_mask2, expert_mask3, routing_weights1, routing_weights2, routing_weights3)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        if self.idx >= train_layer_start and self.idx <= train_layer_end and com[self.idx] == 1:
            attn_output = self.o(attn_output, expert_mask1, expert_mask2, expert_mask3, routing_weights1, routing_weights2, routing_weights3)
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value
    def modify_ffn(ffn, idx):
        ffn.e_k = e_k
        ffn.router = route_only(4096,e_n,e_n,e_n)
        ffn.mlp3 = improved_3part_route_noact_real_moe(4096, 14336,e_n,e_n,e_n,wids[idx],wids[idx],wids[idx],wids[idx]).cuda()
        ffn.mlp1 = improved_3part_route_noact_real_moe(4096, 14336,e_n,e_n,e_n,wids[idx],wids[idx],wids[idx],wids[idx]).cuda()
        ffn.mlp2 = improved_3part_route_noact_real_moe(14336, 4096,e_n,e_n,e_n,wids[idx],wids[idx],wids[idx],wids[idx]).cuda()

        if ainit == 1:
            ffn.up_proj.cuda()
            U, S, V = torch.svd(ffn.up_proj.weight.data.t())
            ffn.mlp1.factors1.weight.data = U[:, :wids[idx]*e_n].t()
            for j in range(e_n):
                ffn.mlp1.factors2[j].weight.data = S[:wids[idx]*e_n].diag()
            ffn.mlp1.factors3.weight.data = V[:, :wids[idx]*e_n]
            ffn.up_proj.to('cpu')
            ffn.mlp1.to('cpu')

            ffn.down_proj.cuda()
            U, S, V = torch.svd(ffn.down_proj.weight.data.t())
            ffn.mlp2.factors1.weight.data = U[:, :wids[idx]*e_n].t()
            for j in range(e_n):
                ffn.mlp2.factors2[j].weight.data = S[:wids[idx]*e_n].diag()
            ffn.mlp2.factors3.weight.data = V[:, :wids[idx]*e_n]
            ffn.down_proj.to('cpu')
            ffn.mlp2.to('cpu')

            ffn.gate_proj.cuda()
            U, S, V = torch.svd(ffn.gate_proj.weight.data.t())
            ffn.mlp3.factors1.weight.data = U[:, :wids[idx]*e_n].t()
            for j in range(e_n):
                ffn.mlp3.factors2[j].weight.data = S[:wids[idx]*e_n].diag()
            ffn.mlp3.factors3.weight.data = V[:, :wids[idx]*e_n]
            ffn.gate_proj.to('cpu')
            ffn.mlp3.to('cpu')

        ffn.idx = idx
        ffn.forward = types.MethodType(_forward, ffn)
    def modify_att(ffn, idx):
        ffn.e_k = e_k
        ffn.router = route_only(4096,e_n,e_n,e_n)
        ffn.q = improved_3part_route_noact_real_moe(4096, 4096,e_n,e_n,e_n,wids[idx],wids[idx],wids[idx],wids[idx]).cuda()
        ffn.k = improved_3part_route_noact_real_moe(4096, 1024,e_n,e_n,e_n,int(wids[idx]/4),int(wids[idx]/4),int(wids[idx]/4),int(wids[idx]/4)).cuda()
        ffn.v = improved_3part_route_noact_real_moe(4096, 1024,e_n,e_n,e_n,int(wids[idx]/4),int(wids[idx]/4),int(wids[idx]/4),int(wids[idx]/4)).cuda()
        ffn.o = improved_3part_route_noact_real_moe(4096, 4096,e_n,e_n,e_n,wids[idx],wids[idx],wids[idx],wids[idx]).cuda()

        if ainit == 1:
            ffn.q_proj.cuda()
            U, S, V = torch.svd(ffn.q_proj.weight.data.t())
            ffn.q.factors1.weight.data = U[:, :wids[idx]*e_n].t()
            for j in range(e_n):
                ffn.q.factors2[j].weight.data = S[:wids[idx]*e_n].diag()
            ffn.q.factors3.weight.data = V[:, :wids[idx]*e_n]
            ffn.q_proj.to('cpu')
            ffn.q.to('cpu')

            ffn.k_proj.cuda()
            U, S, V = torch.svd(ffn.k_proj.weight.data.t())
            ffn.k.factors1.weight.data = U[:, :int(wids[idx]/4)*e_n].t()
            for j in range(e_n):
                ffn.k.factors2[j].weight.data = S[:int(wids[idx]/4)*e_n].diag()
            ffn.k.factors3.weight.data = V[:, :int(wids[idx]/4)*e_n]
            ffn.k_proj.to('cpu')
            ffn.k.to('cpu')

            ffn.v_proj.cuda()
            U, S, V = torch.svd(ffn.v_proj.weight.data.t())
            ffn.v.factors1.weight.data = U[:, :int(wids[idx]/4)*e_n].t()
            for j in range(e_n):
                ffn.v.factors2[j].weight.data = S[:int(wids[idx]/4)*e_n].diag()
            ffn.v.factors3.weight.data = V[:, :int(wids[idx]/4)*e_n]
            ffn.v_proj.to('cpu')
            ffn.v.to('cpu')

            ffn.o_proj.cuda()
            U, S, V = torch.svd(ffn.o_proj.weight.data.t())
            ffn.o.factors1.weight.data = U[:, :wids[idx]*e_n].t()
            for j in range(e_n):
                ffn.o.factors2[j].weight.data = S[:wids[idx]*e_n].diag()
            ffn.o.factors3.weight.data = V[:, :wids[idx]*e_n]
            ffn.o_proj.to('cpu')
            ffn.o.to('cpu')

        ffn.idx = idx
        ffn.forward = types.MethodType(_forward3, ffn)       


    for layer_idx, layer in enumerate(model.model.layers):
        print(layer_idx)
        if com[layer_idx] == 1:
            if c_FFN == 1:
                ffn = layer.mlp
                modify_ffn(ffn, layer_idx)
            if c_att == 1:
                att = layer.self_attn
                modify_att(att, layer_idx)
        

        
for p in model.parameters():
    p.requires_grad = False
change_forward(model, e_n)
if load == 1:
    model.load_state_dict(torch.load('./acc_coe_best.bin', map_location='cpu'), strict=False)

model.bfloat16()
model.cuda()

def set_train_range(model, start, end):
    for layer_idx, layer in enumerate(model.model.layers):
        if com[layer_idx] == 1:
            if layer_idx>=start and layer_idx<=end:
                for p in layer.mlp.mlp1.parameters():
                    p.requires_grad = True
                for p in layer.mlp.mlp2.parameters():
                    p.requires_grad = True
                for p in layer.mlp.mlp3.parameters():
                    p.requires_grad = True
                for p in layer.mlp.router.parameters():
                    p.requires_grad = True
                for p in layer.self_attn.q.parameters():
                    p.requires_grad = True
                for p in layer.self_attn.k.parameters():
                    p.requires_grad = True
                for p in layer.self_attn.v.parameters():
                    p.requires_grad = True
                for p in layer.self_attn.o.parameters():
                    p.requires_grad = True
                for p in layer.self_attn.router.parameters():
                    p.requires_grad = True
            else:
                for p in layer.mlp.mlp1.parameters():
                    p.requires_grad = False
                for p in layer.mlp.mlp2.parameters():
                    p.requires_grad = False
                for p in layer.mlp.mlp3.parameters():
                    p.requires_grad = False
                for p in layer.mlp.router.parameters():
                    p.requires_grad = False
                for p in layer.self_attn.q.parameters():
                    p.requires_grad = False
                for p in layer.self_attn.k.parameters():
                    p.requires_grad = False
                for p in layer.self_attn.v.parameters():
                    p.requires_grad = False
                for p in layer.self_attn.o.parameters():
                    p.requires_grad = False
                for p in layer.self_attn.router.parameters():
                    p.requires_grad = False

set_train_range(model, train_layer_start, train_layer_end)

train = coedataset(['./coe_data/acc/hellaswag/train.json','./coe_data/acc/piqa/train.json','./coe_data/acc/winogrande/train.json','./coe_data/acc/mmlu/train.json'])

hellaswag_eval = coedataset(['./coe_data/acc/hellaswag/test.json'])
piqa_eval = coedataset(['./coe_data/acc/piqa/test.json'])
winogrande_eval = coedataset(['./coe_data/acc/winogrande/test.json'])
mmlu_eval = coedataset(['./coe_data/acc/mmlu/test.json'])
evals = {"mmlu":mmlu_eval,"hellaswag":hellaswag_eval,"piqa":piqa_eval,"winogrande":winogrande_eval}

for key in evals:
    e = 0
    while e < len(evals[key]):
        len2 = tokenizer(evals[key][e]['text'], return_tensors='pt', padding=True, truncation=True).input_ids.shape[1]
        if len2 < 128:
            e+=1
        else:
            evals[key].remove(evals[key][e])

e = 0
while e < len(train):
    len2 = tokenizer(train[e]['text'], return_tensors='pt', padding=True, truncation=True).input_ids.shape[1]
    if len2 < 128 and len(train[e]['text']) != 0:
        e+=1
    else:
        train.remove(train[e])

dataloaders = torch.utils.data.DataLoader(train, batch_size=mini_batch_size, shuffle=True)


trick = {'0':'A', '1':'B','2':'C','3':'D','A':'0', 'B':'1','C':'2','D':'3'}

accu = 0
max_acc = 0
no_better = 0

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
all_step = len(train) // batch_size
warmup_step = all_step // 10
lr_lambda = lambda step: min(step / (warmup_step + 1e-8), 1.0)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
loss_func = torch.nn.MSELoss()

for epoch in range(3000):
    step = 0
    model.train()
    optimizer.zero_grad()

    for batch in tqdm.tqdm(dataloaders):
        print(log, folder, "          training layer: ", train_layer_start, " ~ ", train_layer_end)
        if len(batch['text'])!= mini_batch_size:
            break
        inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=128)
        input_ids = inputs.input_ids.cuda()
        labels = input_ids.clone()
        for t in range(mini_batch_size):
            labels[t][:-1] =-100
        outputs = model(input_ids = input_ids, labels = input_ids)
        
        with torch.no_grad():
            t_outputs = t_model(input_ids = input_ids)
            
        loss = outputs.loss * 10
        # logits loss
        prob_t = F.softmax(t_outputs.logits, dim=-1)
        log_prob_s = F.log_softmax(outputs.logits, dim=-1)
        loss += -(prob_t * log_prob_s).sum(dim=1).mean() * 10
        
        # # last_hidden loss
        h_t = t_outputs.hidden_states[-1].detach()
        h_s = outputs.hidden_states[-1]
        loss += F.mse_loss(h_s, h_t.bfloat16()) * 10

        # hidden_state loss
        mutiid = 0
        for h_s, h_t in zip(outputs.hidden_states, t_outputs.hidden_states):
            h_t = h_t.detach()
            loss += F.mse_loss(h_s, h_t.bfloat16()) * 0.5

        # print("all loss:  ", loss)

        loss.backward()

        step += 1
        if step % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % eval_gap == 0:
            on_eval = True
            accu += 1
            model.eval()
            accs=[]
            time_start = time.time()
            for key in evals:
                eval_dataloaders = torch.utils.data.DataLoader(evals[key], batch_size=mini_batch_size)
                correct = 0
                total = 0
                for batch in tqdm.tqdm(eval_dataloaders):
                    inputs = tokenizer(batch['text'], return_tensors='pt', padding=True, truncation=True, max_length=128)
                    input_ids = inputs.input_ids.cuda()
                    labels = batch['label']
                    outputs = model(input_ids = input_ids).logits[:,-1,:]
                    tmp_correct = 0
                    for ii in range(len(labels)):
                        probs = torch.tensor(
                                [
                                    outputs[ii][tokenizer("A").input_ids[-1]],
                                    outputs[ii][tokenizer("B").input_ids[-1]],
                                    outputs[ii][tokenizer("C").input_ids[-1]],
                                    outputs[ii][tokenizer("D").input_ids[-1]],
                                    outputs[ii][tokenizer("0").input_ids[-1]],
                                    outputs[ii][tokenizer("1").input_ids[-1]],
                                    outputs[ii][tokenizer("2").input_ids[-1]],
                                    outputs[ii][tokenizer("3").input_ids[-1]],
                                ]
                            ).float()
                        pred = {0: "A", 1: "B", 2: "C", 3: "D", 4:"0", 5:"1", 6:"2", 7:"3"}[torch.argmax(probs).item()]
                        if str(labels[ii]) == pred or (pred in trick and trick[pred] == str(labels[ii])):
                            tmp_correct+=1
                    correct += tmp_correct
                    total += len(labels)
                accs.append(correct * 1. / total)
                print(key,"_Acc", correct * 1. / total,'\n')
                with open(log, 'a') as d:
                    d.write(key+"_Acc:  " + str(correct * 1. / total)+'\n')
            time_use = time.time() - time_start
            print("Acc", sum(accs)/len(accs), "             ", train_layer_start, train_layer_end)
            with open(log, 'a') as d:
                d.write(str(train_layer_start)+' '+str(train_layer_end)+'  time:'+str(time_use)+"*****************"+str(sum(accs)/len(accs))+'\n')
            
            if max_acc < sum(accs)/len(accs):
                no_better = 0
                max_acc = sum(accs)/len(accs)
                record = model
                if bosave :
                    if train_layer_start == save_layer:
                        torch.save(model.state_dict(), '{}/acc_{}.bin'.format(folder, max_acc))
            else:
                model = record
                no_better += 1
            torch.save(model.state_dict(),  '{}/ckpt.bin'.format(folder))

            #从前向后
            if train_mode == 'front_first':
                if sum(accs)/len(accs) > aim_acc and train_layer_end < 31:
                    train_layer_end += window_size
                    set_train_range(model, train_layer_start, train_layer_end)
                    accu = 0
                    no_better = 0
                    max_acc = 0
                elif no_better > keep and train_layer_end < 31:
                    train_layer_end += window_size
                    set_train_range(model, train_layer_start, train_layer_end)
                    accu = 0
                    no_better = 0
                    max_acc = 0
                    model = record
            
            #从后向前
            elif train_mode == 'back_first':
                if sum(accs)/len(accs) > aim_acc and train_layer_start > 0:
                    train_layer_start -= window_size
                    set_train_range(model, train_layer_start, train_layer_end)
                    accu = 0
                    no_better = 0
                    max_acc = 0
                elif no_better > keep and train_layer_start > 0:
                    train_layer_start -= window_size
                    set_train_range(model, train_layer_start, train_layer_end)
                    accu = 0
                    no_better = 0
                    max_acc = 0
                    model = record
            
            #滑动窗口
            elif train_mode == 'back_window':
                if sum(accs)/len(accs) > aim_acc and train_layer_start > 0 :
                    train_layer_start -= window_size
                    train_layer_end -= window_size
                    set_train_range(model, train_layer_start, train_layer_end)
                    accu = 0
                    no_better = 0
                    max_acc = 0
                elif no_better > keep and train_layer_start > 0 :
                    train_layer_start -= window_size
                    train_layer_end -= window_size
                    set_train_range(model, train_layer_start, train_layer_end)
                    accu = 0
                    no_better = 0
                    max_acc = 0
                    model = record
            elif train_mode == 'front_window':
                if sum(accs)/len(accs) > aim_acc and train_layer_end <31 :
                    train_layer_start += window_size
                    train_layer_end += window_size
                    set_train_range(model, train_layer_start, train_layer_end)
                    accu = 0
                    no_better = 0
                    max_acc = 0
                elif no_better > keep and train_layer_end <31 :
                    train_layer_start += window_size
                    train_layer_end += window_size
                    set_train_range(model, train_layer_start, train_layer_end)
                    accu = 0
                    no_better = 0
                    max_acc = 0
                    model = record
            
            model.train()
            on_eval = False
