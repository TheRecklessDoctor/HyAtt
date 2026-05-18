import torch
from torch import nn
from torch import Tensor
from typing import Optional
# from hydra.utils import instantiate
# from timm.models.layers import trunc_normal_, DropPath
# import torch.nn.functional as F

class FFN(nn.Module):

        def __init__(
                self, 
                n_features_in:int, 
                n_features_out:int, 
                dropout:int, 
                act:nn.Module = None
        ):
                super().__init__()

                ops = []
                ops.append(nn.Linear(n_features_in, n_features_out))
                ops.append(nn.SiLU()) #prior version used Gelu
                ops.append(nn.Dropout(dropout))
                ops.append(nn.Linear(n_features_out, n_features_in))

                self.ffn = nn.Sequential(*ops)
        
        def forward(self, input):

                return self.ffn(input)

        

class Interactor(nn.Module):

        def __init__(
                self, 
                embed_dim: int, 
                feedforward_dim:int, 
                n_heads:int, 
                dropout:int
        ):
                super().__init__()
                self.self_attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
                self.cross_attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)

                self.ffn = FFN(embed_dim, feedforward_dim, dropout)

                self.norm1 = nn.LayerNorm(embed_dim)
                self.norm2 = nn.LayerNorm(embed_dim)
                self.norm3 = nn.LayerNorm(embed_dim)

                self.dropout1 = nn.Dropout(dropout)
                self.dropout2 = nn.Dropout(dropout)
                self.dropout3 = nn.Dropout(dropout)     
        
        def with_pos_embdedding(
                        self, 
                        tensor:Tensor, 
                        pos:Optional[Tensor]
        ):
                return tensor if pos is None else tensor + pos
        
        def forward(
                self, 
                tgt, 
                memory, 
                tgt_key_padding_mask: Optional[Tensor] = None,
                mem_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None
        ):
                
                q = k = self.with_pos_embdedding(tgt, query_pos)
                v = tgt

                tgt2 = self.self_attention(
                        query=q, 
                        key=k, 
                        value=v,
                        attn_mask=None,
                        key_padding_mask = tgt_key_padding_mask
                )[0]
                tgt = tgt + self.dropout1(tgt2)
                tgt = self.norm1(tgt)

                # cross-attention
                # Might have used wrong query. Test with self.with_pos_embedding(tgt, query_pos)
                tgt2 = self.cross_attention(
                        query = self.with_pos_embdedding(tgt, query_pos),
                        key = self.with_pos_embdedding(memory, pos),
                        value = memory, 
                        attn_mask = None,
                        key_padding_mask = mem_key_padding_mask
                )[0]   
                
                tgt = tgt + self.dropout2(tgt2)
                tgt = self.norm2(tgt)

                # this might be overwriting the pre-ffn representation. Need to change variable names.
                tgt = self.ffn(tgt)
                tgt = tgt + self.dropout3(tgt)
                tgt = self.norm3(tgt)

                return tgt



class Bridger(nn.Module):
        def __init__(
                self, 
                embed_dim: int, 
                feedforward_dim:int,
                n_heads:int, 
                dropout:int,
                fusion_weight: float = 0.5
        ):
                super().__init__()

                self.align_a = nn.Linear(embed_dim, embed_dim)

                self.interactor_a = Interactor(embed_dim, feedforward_dim, n_heads, dropout)
                self.interactor_b = Interactor(embed_dim, feedforward_dim, n_heads, dropout)

                # self.dropout_1 = nn.Dropout(dropout)
                # self.dropout_2 = nn.Dropout(dropout)

                # self.norm_a = nn.LayerNorm(embed_dim)
                # self.norm_b = nn.LayerNorm(embed_dim)
                # self.norm_a2 = nn.LayerNorm(embed_dim)
                # self.norm_b2 = nn.LayerNorm(embed_dim)

                # self.ffn_a = FFN(embed_dim, feedforward_dim, dropout, act)
                # self.ffn_b = FFN(embed_dim, feedforward_dim, dropout, act)

                # self.fusion_weight = nn.Parameter(torch.tensor(fusion_weight))


        def forward(
                self, 
                image_a, 
                image_b
        ):

                # projecting both paths to the same embedding space might cause an issue test with two
                # separate linear layers.
                image_a = self.align_a(image_a)
                image_b = self.align_a(image_b)
                
                cross_att_a = self.interactor_a(image_a, image_b)
                cross_att_b = self.interactor_b(image_b, image_a)

               # image_a = image_a + self.fusion_weight * cross_att_a
               # image_b = image_b + self.fusion_weight * cross_att_b

               # image_a = self.norm_a(image_a)
               # image_b = self.norm_b(image_b)

               # tgt_a = self.ffn_a(image_a)
               # tgt_b = self.ffn_b(image_b)
               #
               # image_a = orig_a + self.dropout_2(self.norm_a2(tgt_a))
               # image_b = orig_b + self.dropout_2(self.norm_b2(tgt_b))
                
               # return image_a, image_b
                return cross_att_a, cross_att_b

class SinglePathInteractor(nn.Module):

        def __init__(
                self, 
                embed_dim: int, 
                feedforward_dim:int, 
                n_heads:int, 
                dropout:int
        ):
                super().__init__()
                self.self_attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)
                self.cross_attention = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout)

                self.ffn = FFN(embed_dim, feedforward_dim, dropout)

                self.norm1 = nn.LayerNorm(embed_dim)
                self.norm2 = nn.LayerNorm(embed_dim)

                self.dropout1 = nn.Dropout(dropout)
                self.dropout2 = nn.Dropout(dropout)
        
        def with_pos_embdedding(
                        self, 
                        tensor:Tensor, 
                        pos:Optional[Tensor]
        ):
                return tensor if pos is None else tensor + pos
        
        def forward(
                self, 
                tgt, 
                memory, 
                tgt_key_padding_mask: Optional[Tensor] = None,
                mem_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None
        ):
                
                q = k = self.with_pos_embdedding(tgt, query_pos)
                v = tgt

                tgt2 = self.self_attention(
                        query=q, 
                        key=k, 
                        value=v,
                        attn_mask=None,
                        key_padding_mask = tgt_key_padding_mask
                )[0]
                tgt = tgt + self.dropout1(tgt2)
                tgt = self.norm1(tgt)

                tgt = self.ffn(tgt)
                tgt = tgt + self.dropout2(tgt)
                tgt = self.norm2(tgt)

                return tgt



class SinglePathBridger(nn.Module):
        def __init__(
                self, 
                embed_dim: int, 
                feedforward_dim:int,
                n_heads:int, 
                dropout:int
        ):
                super().__init__()

                self.align_a = nn.Linear(embed_dim, embed_dim)

                self.interactor_a = SinglePathInteractor(embed_dim, feedforward_dim, n_heads, dropout)
                # self.interactor_b = Interactor(embed_dim, feedforward_dim, n_heads, dropout, act)


        def forward(
                self, 
                image_a, 
                image_b
        ):

                # projecting both paths to the same embedding space might cause an issue test with two
                # separate linear layers.
                # image_a = self.align_a(image_a)
                # image_b = self.align_a(image_b)
                
                attention = self.interactor_a(image_a, image_b)

                
               # return image_a, image_b
                return attention
