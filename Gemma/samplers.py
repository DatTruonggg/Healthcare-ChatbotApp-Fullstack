from collections.abc import Sequence
import torch
from torch import nn
import numpy as np
import sentencepiece as spm
import dataclasses

@dataclasses.dataclass
class SamplerOutput:
  # Decoded samples from the model.
  text: list[str]

  # Per-step logits used during sampling.
  logits: list[list[float]]

  # Tokens corresponding to the generated samples.
  tokens: list[list[int]]

class SamplingState:
    def __init__(self, decoding_step, num_input_tokens, token_buffer, cache, done, total_sampling_steps, logits_buffer=None):
        self.decoding_step = decoding_step
        self.num_input_tokens = num_input_tokens
        self.token_buffer = token_buffer
        self.cache = cache
        self.done = done
        self.total_sampling_steps = total_sampling_steps
        self.logits_buffer = logits_buffer

def compute_attention_masks(time_step, seq_len, input_mask):
    bsz = input_mask.shape[0]
    batch_time_step = torch.full((bsz, 1), time_step, dtype=torch.uint32)
    causal_padding = torch.gt(
        torch.unsqueeze(torch.arange(seq_len), 0), batch_time_step
    )
    causal_padding = causal_padding * torch.unsqueeze(input_mask, dim=-1)
    attention_mask = causal_padding[:, None, None, :].bool()
    attention_mask = attention_mask.squeeze(1)
    return ~attention_mask

class Sampler:
    def __init__(
        self,
        transformer,
        vocab,
        params,
        dtype=torch.bfloat16,
    ):
        self.transformer = transformer
        self.vocab = vocab
        self.params = params
        self.dtype = dtype

    def sample_step(self, params, sampler_state):
        batch_size = sampler_state.token_buffer.shape[0]
        decoding_step = torch.tensor(sampler_state.decoding_step, dtype=torch.int32)
        last_token = sampler_state.token_buffer[:, decoding_step]
        input_mask = last_token != self.vocab.pad_id()
        attention_mask = compute_attention_masks(
            decoding_step, self.transformer.config.max_cache_length, input_mask
        )
        positions = torch.full((batch_size, 1), decoding_step, dtype=torch.int32)
        last_token = last_token.view(batch_size, 1)

        logits, cache = self.transformer(
            {'params': params},
            last_token,
            positions,
            sampler_state.cache,
            attention_mask,
        )

        next_token_candidate = torch.argmax(logits, dim=-1)  # [B, 1]
        next_token_candidate = next_token_candidate[:, 0]  # [B,]

        next_token_candidate = torch.where(
            decoding_step < sampler_state.num_input_tokens - 1,
            sampler_state.token_buffer[:, decoding_step + 1],
            next_token_candidate,
        )

        token_buffer = sampler_state.token_buffer.clone()
        token_buffer[:, decoding_step + 1] = next_token_candidate

        if sampler_state.logits_buffer is not None:
            next_logits = logits.squeeze(1)
            logits_buffer = sampler_state.logits_buffer.clone()
            logits_buffer[:, decoding_step + 1] = next_logits
        else:
            logits_buffer = None

        done = sampler_state.done | torch.eq(
            sampler_state.token_buffer[:, decoding_step + 1], self.vocab.eos_id()
        )

        return SamplingState(
            decoding_step=sampler_state.decoding_step + 1,
            num_input_tokens=sampler_state.num_input_tokens,
            token_buffer=token_buffer,
            logits_buffer=logits_buffer,
            cache=cache,
            done=done,
            total_sampling_steps=sampler_state.total_sampling_steps,
        )

    def init_cache(self, bsz):
        return self.transformer.init_cache(bsz)

    def init_sample_state(
        self, all_input_ids, total_sampling_steps, include_logits=False
    ):
        bsz = len(all_input_ids)
        num_input_tokens = [len(input_ids) for input_ids in all_input_ids]
        buffer_size = total_sampling_steps + 1

        token_buffer = torch.full(
            (bsz, buffer_size), self.vocab.pad_id(), dtype=torch.int32
        )
        for i, (input_ids, num_tokens) in enumerate(
            zip(all_input_ids, num_input_tokens)
        ):
            token_buffer[i, :num_tokens] = torch.tensor(input_ids)

        done = torch.zeros((bsz,), dtype=torch.bool)

        if include_logits:
            logits_buffer = torch.zeros(
                (bsz, buffer_size, self.transformer.config.num_embed),
                dtype=torch.float32,
            )
        else:
            logits_buffer = None

        return SamplingState(
            decoding_step=0,
            num_input_tokens=torch.tensor(num_input_tokens, dtype=torch.int32),
            token_buffer=token_buffer,
            logits_buffer=logits_buffer,
            cache=self.init_cache(bsz),
            done=done,
            total_sampling_steps=total_sampling_steps,
        )

    def tokenize(self, input_string):
        input_ids = self.vocab.EncodeAsIds(input_string)
        input_ids = [self.vocab.bos_id()] + input_ids
        input_ids = torch.tensor(input_ids, dtype=torch.int32)
        return input_ids

    def sample_fn(self, params, initial_sampling_state):
        def sample_with_params(sampler_state):
            return self.sample_step(params, sampler_state)

        def cond_fn(sampler_state):
            return (
                sampler_state.decoding_step < sampler_state.total_sampling_steps
            ) & torch.any(torch.logical_not(sampler_state.done))

        return torch.jit.script(torch.lax.while_loop)(
            cond_fn, sample_with_params, initial_sampling_state
        )

    def __call__(
        self,
        input_strings,
        total_generation_steps,
        echo=False,
        return_logits=True,
    ):
        all_input_ids = [self.tokenize(x) for x in input_strings]
        max_input_length = max(len(input_ids) for input_ids in all_input_ids)
        total_sampling_steps = max_input_length + total_generation_steps
        initial_sampling_state = self.init_sample_state(
            all_input_ids,
            include_logits=return_logits,
            total_sampling_steps=total_sampling_steps,
        )

        sampling_state = self.sample_fn(
            self.params, initial_sampling_state
        )

        out_tokens = []
        out_logits = []
        for i, (token_buffer, num_tokens) in enumerate(
            zip(
                sampling_state.token_buffer,
                sampling_state.num_input_tokens,
            )
        ):
            start_idx = 0 if echo else num_tokens.item()
            out_tokens.append(
                token_buffer[start_idx:total_sampling_steps].tolist()
            )
            if return_logits:
                logits_buffer = sampling_state.logits_buffer[i]
                out_logits.append(
                    logits_buffer[start_idx:total_sampling_steps].tolist()
                )

        decoded_outputs = [
            self.vocab.DecodeIds(tokens) for tokens in out_tokens
        ]

        result = SamplerOutput(
            text=decoded_outputs,
            logits=out_logits,
            tokens=out_tokens,
        )
        return result
