import torch
from torch import nn
from utils.nn_utils import (
    timestep_embedding,
    RotaryEmbedding,
    GemmaRMSNorm,
)
from tasks.domain_utils import DomainAwareLinear
from einops import rearrange

    
class DiTBlock(nn.Module):
    def __init__(self, llm_emb_dim, dim_head=64, heads=8, out_dim=512, gated=True, dropout=0.1, max_seq_len=10000):
        super().__init__()
        
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        
        # Projection layers for state keys, values. In case that the state from llm is different from the dim here,
        # this layer will project the state to the dim here.
        self.to_state_kv = nn.Linear(llm_emb_dim, dim_inner, bias=True)
        
        # Projection layers for action queries, keys, values
        self.to_action_qkv = nn.Linear(dim_inner, 3 * dim_inner, bias=True)
        
        # Gating projection for action tokens
        self.to_action_gate = nn.Linear(dim_inner, dim_inner, bias=True) if gated else None

        # Output projections
        self.to_out = nn.Linear(dim_inner, dim_inner, bias=True)

        # Rotary embedding
        self.rotary_emb = RotaryEmbedding(dim_head, max_seq_len=max_seq_len)
        
        # ffn
        self.ffn = nn.Sequential(
            nn.Linear(dim_inner, dim_inner),
            nn.GELU(),
            nn.Linear(dim_inner, out_dim),
        )
        
        # dropout
        self.dropout = nn.Dropout(dropout)
        
        # normalization
        self.norm1 = GemmaRMSNorm(dim_inner)
        self.norm2 = GemmaRMSNorm(dim_inner)

    def forward(self, vlm_keys, vlm_values, t_tokens, proprio_tokens, action_tokens):
        ''' 
            We only use the action queries for attention!
            vlm_keys: [b, seq_len(n), llm_emb_dim]
            action_tokens: [b, seq_len(m-2), action_token_dim]
        '''
        assert t_tokens.shape[1] == 1, "The time should be single timestep!"
        
        if proprio_tokens.ndim == 2:
            proprio_tokens = proprio_tokens.unsqueeze(1)  # (B, 1, D)

        proprio_token_len = proprio_tokens.shape[1]
        original_proprio_token_len = proprio_token_len
        original_all_tokens = torch.cat([t_tokens, proprio_tokens, action_tokens], dim=1)
        
        all_tokens = self.norm1(original_all_tokens)
        
        b, n, d = vlm_keys.shape
        _, m, _ = all_tokens.shape

        # Project state queries, keys, and values
        state_kv = self.to_state_kv(torch.cat([vlm_keys, vlm_values], dim=1))
        state_keys, state_values = state_kv.chunk(2, dim=1)
        
        # Project action queries, keys, and values
        action_qkv = self.to_action_qkv(all_tokens)
        action_queries, action_keys, action_values = action_qkv.chunk(3, dim=-1)
        
        # Reshape to multi-head format
        state_keys = rearrange(state_keys, 'b n (h d) -> b h n d', h=self.heads)
        state_values = rearrange(state_values, 'b n (h d) -> b h n d', h=self.heads)
        
        action_queries = rearrange(action_queries, 'b n (h d) -> b h n d', h=self.heads)
        action_keys = rearrange(action_keys, 'b n (h d) -> b h n d', h=self.heads)
        action_values = rearrange(action_values, 'b n (h d) -> b h n d', h=self.heads)

        keys = torch.cat([state_keys, action_keys], dim=-2)
        values = torch.cat([state_values, action_values], dim=-2)

        # Apply rotary embeddings
        action_queries, keys = self.rotary_emb(action_queries, keys)
        
        # Create attention mask
        causal_mask = torch.ones((n + m, n + m), dtype=torch.bool)
        causal_mask = causal_mask.triu(diagonal=1)

        state_bidirectional_mask = torch.zeros((n, n), dtype=torch.bool)
        action_bidirectional_mask = torch.ones((m, m), dtype=torch.bool)
        action_bidirectional_mask[proprio_token_len+1:, :] = False  # Allow action -> action
        action_bidirectional_mask[1:1+proprio_token_len, :1+proprio_token_len] = False  # Allow proprio -> time
        action_bidirectional_mask[0, 0] = False  # Allow time -> time

        full_mask = causal_mask.clone()

        full_mask[:n, :n] = state_bidirectional_mask
        full_mask[n:, n:] = action_bidirectional_mask
        full_mask[:n, n:] = True  # forbid state -> action

        full_mask = ~full_mask  # True means allow attention
        full_mask = rearrange(full_mask, 'i j -> 1 1 i j')  # Broadcast for multi-heads
        full_mask = full_mask.to(action_queries.device)

        # Compute scaled dot-product attention (only for action_queries)
        sim = torch.einsum('b h i d, b h j d -> b h i j', action_queries, keys) * self.scale
        sim = sim.masked_fill(~full_mask[:, :, -m:], float('-inf'))  # Apply attention mask
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, values)

        # Merge heads back
        all_out = rearrange(out, 'b h n d -> b n (h d)')

        # Apply gating to action output if gated
        if self.to_action_gate:
            gate = self.to_action_gate(all_out).sigmoid()
            all_out = all_out * gate

        # Final projection
        all_out = self.to_out(all_out)
  
        # dropout + residual
        all_out = original_all_tokens + self.dropout(all_out)
        
        # ffn
        ffn_output = self.ffn(self.norm2(all_out))
        all_out = all_out + self.dropout(ffn_output)

        timestep_token_output = all_out[:, 0:1, :]
        proprio_token_output = all_out[:, 1:original_proprio_token_len+1, :]
        action_tokens_output = all_out[:, proprio_token_len+1:, :]

        return timestep_token_output, proprio_token_output, action_tokens_output
    

class DiT(nn.Module):
    def __init__(self, cfg, num_layers, llm_emb_dim, domain_aware=False):
        super(DiT, self).__init__()
        self.cfg = cfg
        self.num_layers = num_layers
        self.hidden_size = cfg.hidden_size
        
        self.domain_aware = domain_aware
        if domain_aware:
            self.action_embed = DomainAwareLinear(cfg.action_dim, cfg.hidden_size)
            self.action_out_proj = DomainAwareLinear(cfg.hidden_size, cfg.action_dim)
            self.proprio_embed = DomainAwareLinear(cfg.proprio_dim, cfg.hidden_size)
        else:
            self.action_embed = nn.Linear(cfg.action_dim, cfg.hidden_size)
            self.action_out_proj = nn.Linear(cfg.hidden_size, cfg.action_dim)
            self.proprio_embed = nn.Linear(cfg.proprio_dim, cfg.hidden_size)
            
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
            nn.SiLU(),
            nn.Linear(cfg.hidden_size, cfg.hidden_size),
        )
        
        self.blocks = nn.ModuleList([
            DiTBlock(
                llm_emb_dim, 
                cfg.hidden_size//cfg.num_heads, 
                cfg.num_heads, 
                cfg.hidden_size, 
                gated=cfg.gated
            )
            for _ in range(num_layers)
        ])
    
    
    def forward(self, proprio, noisy_actions, t, llm_key, llm_value, bf16=False, domain_id=None):
        '''
            proprio: [b, proprio_len, proprio_dim]
            t: [b,]
            noisy_actions: [b, action_len, action_dim]
            llm_key: [num_layers, b, llm_len, dim]
        '''
        
        assert llm_key.shape[0] == llm_value.shape[0] == self.num_layers, "Invalid number of layers!"
        
        if self.domain_aware:
            action_feat = self.action_embed(noisy_actions, domain_id)
            proprio_feat = self.proprio_embed(proprio, domain_id)
        else:
            action_feat = self.action_embed(noisy_actions)
            proprio_feat = self.proprio_embed(proprio)
        time_emb = self.time_mlp(timestep_embedding(t, self.hidden_size, bf16=bf16)).unsqueeze(1)  # [b, 1, hidden_size]
        
        for layer_idx in range(self.num_layers):
            time_emb, proprio_feat, action_feat = self.blocks[layer_idx](
                llm_key[layer_idx], 
                llm_value[layer_idx], 
                time_emb, 
                proprio_feat, 
                action_feat, 
            )
            
        if self.domain_aware:
            action_feat = self.action_out_proj(action_feat, domain_id)
        else:
            action_feat = self.action_out_proj(action_feat)
        
        return action_feat
    
    