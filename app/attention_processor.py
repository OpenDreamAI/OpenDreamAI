from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

xformers = None


class Attention(nn.Module):
    """
    A cross attention layer.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: bool = False,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        processor: Optional["AttnProcessor"] = None,
    ):
        """
        Initializes the Attention layer with the given parameters.

        Parameters:
            query_dim (int): The number of channels in the query.
            cross_attention_dim (Optional[int], optional): The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
            heads (int, optional): The number of heads to use for multi-head attention. Defaults to 8.
            dim_head (int, optional): The number of channels in each head. Defaults to 64.
            dropout (float, optional): The dropout probability to use. Defaults to 0.0.
            bias (bool, optional): Set to `True` for the query, key, and value linear layers to contain a bias parameter. Defaults to False.
            upcast_attention (bool, optional): Set to `True` to upcast attention scores to float before computing softmax. Defaults to False.
            upcast_softmax (bool, optional): Set to `True` to upcast attention scores to float before computing attention probabilities. Defaults to False.
            cross_attention_norm (bool, optional): Set to `True` to use LayerNorm in cross_attention computation. Defaults to False.
            added_kv_proj_dim (Optional[int], optional): If given, add key and value projection dimensions. Defaults to None.
            norm_num_groups (Optional[int], optional): If given, use GroupNorm with the specified number of groups. Defaults to None.
            out_bias (bool, optional): Set to `True` for the output linear layer to contain a bias parameter. Defaults to True.
            scale_qk (bool, optional): Set to `True` to scale the dot product of query and key by the inverse square root of the head dimension. Defaults to True.
            processor (Optional["AttnProcessor"], optional): If given, use the specified attention processor. Defaults to None.
        """
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.cross_attention_norm = cross_attention_norm

        self.scale = dim_head**-0.5 if scale_qk else 1.0

        self.heads = heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(
                num_channels=inner_dim,
                num_groups=norm_num_groups,
                eps=1e-5,
                affine=True,
            )
        else:
            self.group_norm = None

        if cross_attention_norm:
            self.norm_cross = nn.LayerNorm(cross_attention_dim)

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias)

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, cross_attention_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        if processor is None:
            processor = (
                AttnProcessor2_0()
                if hasattr(F, "scaled_dot_product_attention") and scale_qk
                else AttnProcessor()
            )
        self.set_processor(processor)

    def set_use_memory_efficient_attention_xformers(
        self,
        use_memory_efficient_attention_xformers: bool,
        attention_op: Optional[Callable] = None,
    ):
        """
        Sets whether to use memory-efficient attention with xformers library.

        Parameters:
            use_memory_efficient_attention_xformers (bool): Set to `True` to use memory-efficient attention implementation from the xformers library.
            attention_op (Optional[Callable], optional): Optional attention operation to be used with xformers' memory-efficient attention. Defaults to None.

        Raises:
            NotImplementedError: If `added_kv_proj_dim` is defined.
            ValueError: If torch.cuda.is_available() is False, as xformers' memory-efficient attention is only available for GPU.
        """

        is_lora = hasattr(self, "processor") and isinstance(
            self.processor, (LoRAAttnProcessor, LoRAXFormersAttnProcessor)
        )

        if use_memory_efficient_attention_xformers:
            if self.added_kv_proj_dim is not None:
                raise NotImplementedError(
                    "Memory efficient attention with `xformers` is currently not supported when"
                    " `self.added_kv_proj_dim` is defined."
                )
            elif not torch.cuda.is_available():
                raise ValueError(
                    "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                    " only available for GPU "
                )
            else:
                try:
                    # Make sure we can run the memory efficient attention
                    _ = xformers.ops.memory_efficient_attention(
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                    )
                except Exception as e:
                    raise e

            if is_lora:
                processor = LoRAXFormersAttnProcessor(
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    rank=self.processor.rank,
                    attention_op=attention_op,
                )
                processor.load_state_dict(self.processor.state_dict())
                processor.to(self.processor.to_q_lora.up.weight.device)
            else:
                processor = XFormersAttnProcessor(attention_op=attention_op)
        else:
            if is_lora:
                processor = LoRAAttnProcessor(
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    rank=self.processor.rank,
                )
                processor.load_state_dict(self.processor.state_dict())
                processor.to(self.processor.to_q_lora.up.weight.device)
            else:
                processor = AttnProcessor()

        self.set_processor(processor)

    def set_attention_slice(self, slice_size: int) -> None:
        """
        Sets the attention slice size for sliced attention.

        Parameters:
            slice_size (int): The size of the slice for the sliced attention.

        Raises:
            ValueError: If slice_size is greater than the number of heads in the Attention layer.
        """

        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(
                f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}."
            )

        if slice_size is not None and self.added_kv_proj_dim is not None:
            processor = SlicedAttnAddedKVProcessor(slice_size)
        elif slice_size is not None:
            processor = SlicedAttnProcessor(slice_size)
        elif self.added_kv_proj_dim is not None:
            processor = AttnAddedKVProcessor()
        else:
            processor = AttnProcessor()

        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor") -> None:
        """
        Sets the attention processor for the Attention layer.

        Parameters:
            processor (AttnProcessor): An instance of the AttnProcessor class or a subclass thereof.
        """

        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            self._modules.pop("processor")

        self.processor = processor

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor = None,
        attention_mask: Tensor = None,
        **cross_attention_kwargs,
    ):
        """
        Performs the forward pass of the Attention layer.

        Parameters:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).
            encoder_hidden_states (Optional[torch.Tensor], optional): Encoder hidden states tensor of shape (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask tensor of shape (batch_size, sequence_length). Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor: Tensor) -> Tensor:
        """
        Reshapes the input tensor from batch dimensions to head dimensions.

        Parameters:
            tensor (torch.Tensor): Input tensor of shape (batch_size, sequence_length, hidden_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size // head_size, sequence_length, hidden_size * head_size).
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size // head_size, seq_len, dim * head_size
        )
        return tensor

    def head_to_batch_dim(self, tensor: Tensor) -> Tensor:
        """
        Reshapes the input tensor from head dimensions to batch dimensions.

        Parameters:
            tensor (torch.Tensor): Input tensor of shape (batch_size // head_size, sequence_length, hidden_size * head_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, hidden_size).
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(
            batch_size * head_size, seq_len, dim // head_size
        )
        return tensor

    def get_attention_scores(
        self, query: Tensor, key: Tensor, attention_mask: Tensor = None
    ) -> Tensor:
        """
        Computes the attention scores between query and key tensors.

        Parameters:
            query (torch.Tensor): Query tensor of shape (batch_size, sequence_length, hidden_size).
            key (torch.Tensor): Key tensor of shape (batch_size, sequence_length, hidden_size).
            attention_mask (Optional[torch.Tensor], optional): Attention mask

        Returns:
            torch.Tensor: Attention scores tensor of shape (batch_size, sequence_length, sequence_length).
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0],
                query.shape[1],
                key.shape[1],
                dtype=query.dtype,
                device=query.device,
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(
        self, attention_mask: Tensor, target_length: int, batch_size: int = None
    ) -> Tensor:
        """
        Ensures that the attention mask tensor has the correct dimensions for the current batch size and target length.

        Parameters:
            attention_mask (torch.Tensor): Attention mask tensor of shape (batch_size, sequence_length).
            target_length (int): Target sequence length.
            batch_size (Optional[int], optional): Batch size. Defaults to None.

        Returns:
            torch.Tensor: Reshaped attention mask tensor with the correct dimensions for the current batch size and target length.
        """
        if batch_size is None:
            batch_size = 1

        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        if attention_mask.shape[-1] != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (
                    attention_mask.shape[0],
                    attention_mask.shape[1],
                    target_length,
                )
                padding = torch.zeros(
                    padding_shape,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if attention_mask.shape[0] < batch_size * head_size:
            attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        return attention_mask


class AttnProcessor:
    """
    An Attention Processor class that handles the core attention calculation for the Attention layer.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
    ):
        """
        Calls the attention processor to compute attention scores and hidden states.

        Parameters:
            attn (Attention): An instance of the Attention class.
            hidden_states (torch.Tensor): Tensor of input hidden states of shape (batch_size, sequence_length, hidden_size).
            encoder_hidden_states (Optional[torch.Tensor], optional): Tensor of encoder hidden states of shape (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask tensor of shape (batch_size, sequence_length). Defaults to None.

        Returns:
            torch.Tensor: The output hidden states of the attention layer with shape (batch_size, sequence_length, hidden_size).
        """
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class LoRALinearLayer(nn.Module):
    """
    A low-rank approximation (LoRA) linear layer that reduces the parameter size by using a lower-dimensional approximation.

    Parameters:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        rank (int, optional, defaults to 4): The rank of the low-rank approximation.
    """

    def __init__(self, in_features, out_features, rank=4):
        super().__init__()

        if rank > min(in_features, out_features):
            raise ValueError(
                f"LoRA rank {rank} must be less or equal than {min(in_features, out_features)}"
            )

        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)

        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Computes the output of the LoRALinearLayer given the input hidden states.

        Parameters:
            hidden_states (torch.Tensor): Input tensor of hidden states with shape (batch_size, sequence_length, in_features).

        Returns:
            torch.Tensor: The output tensor of the low-rank approximation with shape (batch_size, sequence_length, out_features).
        """
        orig_dtype = hidden_states.dtype
        dtype = self.down.weight.dtype

        down_hidden_states = self.down(hidden_states.to(dtype))
        up_hidden_states = self.up(down_hidden_states)

        return up_hidden_states.to(orig_dtype)


class LoRAAttnProcessor(nn.Module):
    """
    A Low-Rank Approximation (LoRA) Attention Processor class that handles the core attention calculation
    for the Attention layer with a low-rank approximation.
    """

    def __init__(self, hidden_size, cross_attention_dim=None, rank=4):
        """
        Initializes the LoRAAttnProcessor instance with given parameters.

        Parameters:
            hidden_size (int): The size of the hidden states.
            cross_attention_dim (int, optional): The size of the cross-attention hidden states. Defaults to None.
            rank (int, optional, defaults to 4): The rank of the low-rank approximation.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank)
        self.to_k_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank
        )
        self.to_v_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank
        )
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank)

    def __call__(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        scale=1.0,
    ) -> Tensor:
        """
        Calls the LoRA Attention Processor to compute attention scores and hidden states.

        Parameters:
            attn (Attention): An instance of the Attention class.
            hidden_states (torch.Tensor): Tensor of input hidden states of shape (batch_size, sequence_length, hidden_size).
            encoder_hidden_states (Optional[torch.Tensor], optional): Tensor of encoder hidden states of shape (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask tensor of shape (batch_size, sequence_length). Defaults to None.
            scale (float, optional, defaults to 1.0): The scaling factor for the low-rank approximation.

        Returns:
            torch.Tensor: The output hidden states of the attention layer with shape (batch_size, sequence_length, hidden_size).
        """
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query)

        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )

        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(
            encoder_hidden_states
        )
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(
            encoder_hidden_states
        )

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(
            hidden_states
        )
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class AttnAddedKVProcessor:
    """
    An Attention Processor class that computes attention scores and hidden states while adding keys and values
    from an external source, typically used in tasks such as machine translation or sequence-to-sequence modeling.

    This class is designed to be used with an instance of the Attention class to perform the attention calculation.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Calls the AttnAddedKVProcessor to compute attention scores and hidden states with added keys and values
        from an external source.

        Parameters:
            attn (Attention): An instance of the Attention class.
            hidden_states (torch.Tensor): Tensor of input hidden states of shape (batch_size, sequence_length, hidden_size).
            encoder_hidden_states (Optional[torch.Tensor], optional): Tensor of encoder hidden states of shape (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask tensor of shape (batch_size, sequence_length). Defaults to None.

        Returns:
            torch.Tensor: The output hidden states of the attention layer with shape (batch_size, sequence_length, hidden_size).
        """
        residual = hidden_states
        hidden_states = hidden_states.view(
            hidden_states.shape[0], hidden_states.shape[1], -1
        ).transpose(1, 2)
        batch_size, sequence_length, _ = hidden_states.shape
        encoder_hidden_states = encoder_hidden_states.transpose(1, 2)

        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(
            encoder_hidden_states_key_proj
        )
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(
            encoder_hidden_states_value_proj
        )

        key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states


class XFormersAttnProcessor:
    """
    An Attention Processor class that utilizes X-Formers for computing attention scores and hidden states.
    X-Formers provide a memory-efficient implementation of the attention mechanism.

    This class is designed to be used with an instance of the Attention class to perform the attention calculation.
    """

    def __init__(self, attention_op: Optional[Callable] = None):
        """
        Initializes an instance of the XFormersAttnProcessor class.

        Parameters:
            attention_op (Optional[Callable], optional): Optional custom attention operation to be used by X-Formers. Defaults to None.
        """
        self.attention_op = attention_op

    def __call__(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Calls the XFormersAttnProcessor to compute attention scores and hidden states using X-Formers.

        Parameters:
            attn (Attention): An instance of the Attention class.
            hidden_states (torch.Tensor): Tensor of input hidden states of shape (batch_size, sequence_length, hidden_size).
            encoder_hidden_states (Optional[torch.Tensor], optional): Tensor of encoder hidden states of shape (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask tensor of shape (batch_size, sequence_length). Defaults to None.

        Returns:
            torch.Tensor: The output hidden states of the attention layer with shape (batch_size, sequence_length, hidden_size).
        """
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )

        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query).contiguous()
        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=attention_mask,
            op=self.attention_op,
            scale=attn.scale,
        )
        hidden_states = hidden_states.to(query.dtype)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class AttnProcessor2_0:
    """
    An Attention Processor class that utilizes PyTorch 2.0's native scaled dot-product attention implementation.
    This class is designed to be used with an instance of the Attention class to perform the attention calculation.
    """

    def __init__(self):
        """
        Initializes an instance of the AttnProcessor2_0 class.

        Raises:
            ImportError: If PyTorch 2.0 is not installed.
        """
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )

    def __call__(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Calls the AttnProcessor2_0 to compute attention scores and hidden states using PyTorch 2.0's scaled dot-product attention.

        Parameters:
            attn (Attention): An instance of the Attention class.
            hidden_states (torch.Tensor): Tensor of input hidden states of shape (batch_size, sequence_length, hidden_size).
            encoder_hidden_states (Optional[torch.Tensor], optional): Tensor of encoder hidden states of shape (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask tensor of shape (batch_size, sequence_length). Defaults to None.

        Returns:
            torch.Tensor: The output hidden states of the attention layer with shape (batch_size, sequence_length, hidden_size).
        """
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        inner_dim = hidden_states.shape[-1]

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(
                attention_mask, sequence_length, batch_size
            )
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(
                batch_size, attn.heads, -1, attention_mask.shape[-1]
            )

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class LoRAXFormersAttnProcessor(nn.Module):
    """
    A memory-efficient attention processor class that combines the benefits of Low-Rank Adaptive (LoRA) attention
    and X-Formers attention mechanisms. This class is designed to be used with an instance of the Attention class
    to perform the attention calculation.
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        rank: Optional[int] = 4,
        attention_op: Optional[Callable] = None,
    ):
        """
        Initializes an instance of the LoRAXFormersAttnProcessor class.

        Parameters:
            hidden_size (int): The hidden size of the input tensor.
            cross_attention_dim (int): The dimension of the cross-attention layer.
            rank (int, optional): The rank of the low-rank approximation in LoRA. Defaults to 4.
            attention_op (Optional[Callable], optional): An optional attention operation to be used instead of the default attention mechanism. Defaults to None.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.rank = rank
        self.attention_op = attention_op

        self.to_q_lora = LoRALinearLayer(hidden_size, hidden_size, rank)
        self.to_k_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank
        )
        self.to_v_lora = LoRALinearLayer(
            cross_attention_dim or hidden_size, hidden_size, rank
        )
        self.to_out_lora = LoRALinearLayer(hidden_size, hidden_size, rank)

    def __call__(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        scale: Optional[float] = 1.0,
    ) -> Tensor:
        """
        Calls the LoRAXFormersAttnProcessor to compute attention scores and hidden states using memory-efficient attention
        with Low-Rank Adaptive (LoRA) layers.

        Parameters:
            attn (Attention): An instance of the Attention class.
            hidden_states (torch.Tensor): Tensor of input hidden states of shape (batch_size, sequence_length, hidden_size).
            encoder_hidden_states (Optional[torch.Tensor], optional): Tensor of encoder hidden states of shape (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask tensor of shape (batch_size, sequence_length). Defaults to None.
            scale (float, optional): Scaling factor for the LoRA layers. Defaults to 1.0.

        Returns:
            torch.Tensor: The output hidden states of the attention layer with shape (batch_size, sequence_length, hidden_size).
        """
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        query = attn.to_q(hidden_states) + scale * self.to_q_lora(hidden_states)
        query = attn.head_to_batch_dim(query).contiguous()

        encoder_hidden_states = (
            encoder_hidden_states
            if encoder_hidden_states is not None
            else hidden_states
        )

        key = attn.to_k(encoder_hidden_states) + scale * self.to_k_lora(
            encoder_hidden_states
        )
        value = attn.to_v(encoder_hidden_states) + scale * self.to_v_lora(
            encoder_hidden_states
        )

        key = attn.head_to_batch_dim(key).contiguous()
        value = attn.head_to_batch_dim(value).contiguous()

        hidden_states = xformers.ops.memory_efficient_attention(
            query,
            key,
            value,
            attn_bias=attention_mask,
            op=self.attention_op,
            scale=attn.scale,
        )
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states) + scale * self.to_out_lora(
            hidden_states
        )
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class SlicedAttnProcessor:
    """
    A class that implements a sliced attention mechanism, which computes attention in smaller batches to save memory.
    This class is designed to be used with an instance of the Attention class to perform the attention calculation.
    """

    def __init__(self, slice_size):
        """
        Initializes an instance of the SlicedAttnProcessor class.

        Parameters:
                slice_size (int): The size of each slice for batch processing of attention.
        """
        self.slice_size = slice_size

    def __call__(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Calls the SlicedAttnProcessor to compute attention scores and hidden states in a memory-efficient manner by
        dividing the input into smaller slices and processing them in batches.

        Parameters:
            attn (Attention): An instance of the Attention class.
            hidden_states (torch.Tensor): Tensor of input hidden states of shape (batch_size, sequence_length, hidden_size).
            encoder_hidden_states (Optional[torch.Tensor], optional): Tensor of encoder hidden states of shape (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask tensor of shape (batch_size, sequence_length). Defaults to None.

        Returns:
            torch.Tensor: The output hidden states of the attention layer with shape (batch_size, sequence_length, hidden_size).
        """
        batch_size, sequence_length, _ = (
            hidden_states.shape
            if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        query = attn.to_q(hidden_states)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        batch_size_attention, query_tokens, _ = query.shape
        hidden_states = torch.zeros(
            (batch_size_attention, query_tokens, dim // attn.heads),
            device=query.device,
            dtype=query.dtype,
        )

        for i in range(batch_size_attention // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = (
                attention_mask[start_idx:end_idx]
                if attention_mask is not None
                else None
            )

            attn_slice = attn.get_attention_scores(
                query_slice, key_slice, attn_mask_slice
            )

            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states


class SlicedAttnAddedKVProcessor:
    """
    A class that implements a sliced attention mechanism with added key and value projections, which computes attention
    in smaller batches to save memory. This class is designed to be used with an instance of the Attention class to
    perform the attention calculation.
    """

    def __init__(self, slice_size: int):
        """
        Initializes an instance of the SlicedAttnAddedKVProcessor class.

        Parameters:
            slice_size (int): The size of each slice for batch processing of attention.
        """
        self.slice_size = slice_size

    def __call__(
        self,
        attn: "Attention",
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Calls the SlicedAttnAddedKVProcessor to compute attention scores and hidden states in a memory-efficient manner
        by dividing the input into smaller slices and processing them in batches. It also incorporates additional key and
        value projections for the encoder_hidden_states.

        Parameters:
            attn (Attention): An instance of the Attention class.
            hidden_states (torch.Tensor): Tensor of input hidden states of shape (batch_size, sequence_length, hidden_size).
            encoder_hidden_states (Optional[torch.Tensor], optional): Tensor of encoder hidden states of shape (batch_size, sequence_length, hidden_size). Defaults to None.
            attention_mask (Optional[torch.Tensor], optional): Attention mask tensor of shape (batch_size, sequence_length). Defaults to None.

        Returns:
            torch.Tensor: The output hidden states of the attention layer with shape (batch_size, sequence_length, hidden_size).
        """
        residual = hidden_states
        hidden_states = hidden_states.view(
            hidden_states.shape[0], hidden_states.shape[1], -1
        ).transpose(1, 2)
        encoder_hidden_states = encoder_hidden_states.transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape

        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query)

        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        encoder_hidden_states_key_proj = attn.head_to_batch_dim(
            encoder_hidden_states_key_proj
        )
        encoder_hidden_states_value_proj = attn.head_to_batch_dim(
            encoder_hidden_states_value_proj
        )

        key = torch.cat([encoder_hidden_states_key_proj, key], dim=1)
        value = torch.cat([encoder_hidden_states_value_proj, value], dim=1)

        batch_size_attention, query_tokens, _ = query.shape
        hidden_states = torch.zeros(
            (batch_size_attention, query_tokens, dim // attn.heads),
            device=query.device,
            dtype=query.dtype,
        )

        for i in range(batch_size_attention // self.slice_size):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            query_slice = query[start_idx:end_idx]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = (
                attention_mask[start_idx:end_idx]
                if attention_mask is not None
                else None
            )

            attn_slice = attn.get_attention_scores(
                query_slice, key_slice, attn_mask_slice
            )

            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states.transpose(-1, -2).reshape(residual.shape)
        hidden_states = hidden_states + residual

        return hidden_states


AttentionProcessor = Union[
    AttnProcessor,
    XFormersAttnProcessor,
    SlicedAttnProcessor,
    AttnAddedKVProcessor,
    SlicedAttnAddedKVProcessor,
    LoRAAttnProcessor,
    LoRAXFormersAttnProcessor,
]
