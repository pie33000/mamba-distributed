import inspect
import json

import tiktoken
import torch
import torch.nn.functional as F
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers.utils import CONFIG_NAME, WEIGHTS_NAME
from transformers.utils.hub import cached_file


class LMHeadModel(MambaLMHeadModel):
    def __init__(self, config, device):
        self.device = device
        self.enc = tiktoken.get_encoding("gpt2")
        super().__init__(config, device=device)

    def forward(
        self,
        input_ids,
        targets=None,
        position_ids=None,
        inference_params=None,
        num_last_tokens=0,
        **mixer_kwargs,
    ):
        # get logits from the model
        logits = super().forward(
            input_ids,
            position_ids=position_ids,
            inference_params=inference_params,
            num_last_tokens=num_last_tokens,
            **mixer_kwargs,
        )
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.logits.view(-1, logits.logits.size(-1)), targets.view(-1)
            )
        return logits.logits, loss

    @torch.no_grad()
    def generate(self, xgen, top_k=50, max_length=32, seed=42) -> torch.tensor:
        for _ in range(max_length):
            # Get model outputs and extract logits
            logits, _ = self(xgen)
            # Select the logits of the last generated token
            next_token_logits = logits[:, -1, :].squeeze()

            # Apply top-k sampling
            next_token_id = self.top_k_sampling(next_token_logits, k=top_k)

            # Append the sampled token to the generated sequence
            next_token_id_tensor = torch.tensor([[next_token_id]], device=self.device)
            xgen = torch.cat((xgen, next_token_id_tensor), dim=1)

            # Stop if the model generates the end-of-sequence token
            if next_token_id == self.enc.eot_token:
                break

        # Decode the generated sequence
        generated_text = self.enc.decode(xgen[0], skip_special_tokens=True)
        return generated_text

    def top_k_sampling(self, logits: torch.tensor, k: int = 50, seed: int = 42) -> int:
        """
        Apply top-k sampling to the logits and sample a token index.
        """
        # Get the top k logits and their indices
        top_k_logits, top_k_indices = torch.topk(logits, k)

        # Convert logits to probabilities
        probabilities = torch.softmax(top_k_logits, dim=-1)

        # Sample a token index from the top-k probabilities
        sample_rng = torch.Generator(device=self.device)
        sampled_index = torch.multinomial(
            probabilities, num_samples=1, generator=sample_rng
        )

        # Map the sampled index back to the full vocabulary
        next_token_id = top_k_indices[sampled_index]
        return next_token_id.item()

    @classmethod
    def load_from_hf(cls, model_name: str, device: str) -> "LMHeadModel":
        resolved_archive_file = cached_file(
            model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False
        )
        config_data = json.load(open(resolved_archive_file))

        resolved_archive_file = cached_file(
            model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False
        )
        state_dict = torch.load(
            resolved_archive_file, weights_only=True, map_location="cpu", mmap=True
        )

        config = MambaConfig(
            d_model=config_data["d_model"], n_layers=config_data["n_layers"]
        )
        model = cls(config, device=device)

        model.load_state_dict(state_dict)

        return model

    def configure_optimizers(
        self, weight_decay, learning_rate, device_type, master_process
    ):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(
                f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
            )
            print(
                f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
            )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused
        )
        return optimizer
