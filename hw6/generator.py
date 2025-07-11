import torch
import torch.nn as nn
from transformer_basics.layers import Decoder, DecoderLayer, MultiheadAttention, FeedForward, Embedding
from transformer_basics.transformer import get_pad_mask, get_subsequent_mask

class GeneratorTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        pad_token_id,
        bos_token_id,
        eos_token_id,
        max_len,
        tokenizer,
        device="cuda"
    ):
        super().__init__()
        self.device = device
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.max_len = max_len

        mha = MultiheadAttention(d_model, num_heads)
        ffn = FeedForward(d_model, d_ff)
        decoder_layer = DecoderLayer(mha, mha, ffn)
        self.decoder = Decoder(decoder_layer, num_layers)

        self.embedding = Embedding(d_model, vocab_size, pad_token_id)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.to(self.device)

    def forward(self, input_ids):
        input_ids = input_ids.to(self.device)

        pad_mask = get_pad_mask(input_ids, self.pad_token_id).to(self.device)
        tgt_mask = get_subsequent_mask(input_ids).to(self.device)
        mask = pad_mask & tgt_mask

        x = self.embedding(input_ids)
        x = self.decoder(x, encoder_memory=None, src_mask=None, tgt_mask=mask)
        x = self.norm(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, prompt, context_len=50, temperature=1.0, max_out_tokens=200):
        self.eval()

        input_ids = self.tokenizer.encode(prompt).ids
        input_ids = [self.bos_token_id] + input_ids
        input_ids = torch.tensor([input_ids], device=self.device)

        generated = input_ids.clone()

        for _ in range(max_out_tokens):
            input_context = generated[:, -context_len:].to(self.device)
            logits = self(input_context)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), 1)
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.item() == self.eos_token_id:
                break

        return self.tokenizer.decode(generated[0].tolist())
    

    @torch.no_grad()
    def generate_beam(
        self,
        prompt,
        context_len=50,
        max_out_tokens=100,
        beam_width=5,
        length_penalty=1.0
    ):
        self.eval()

        input_ids = self.tokenizer.encode(prompt).ids
        input_ids = [self.bos_token_id] + input_ids
        input_ids = torch.tensor(input_ids, device=self.device).unsqueeze(0)
        beams = [(input_ids, 0.0)]
        completed = []

        for _ in range(max_out_tokens):
            candidates = []

            for seq, score in beams:
                last_token = seq[0, -1].item()
                if last_token == self.eos_token_id:
                    completed.append((seq, score))
                    continue

                input_context = seq[:, -context_len:] 
                logits = self(input_context) 
                next_token_logits = logits[:, -1, :] 
                probs = torch.log_softmax(next_token_logits, dim=-1)

                topk_probs, topk_ids = probs.topk(beam_width, dim=-1)  

                for i in range(beam_width):
                    next_id = topk_ids[0, i].unsqueeze(0).unsqueeze(0) 
                    new_seq = torch.cat([seq, next_id], dim=1) 
                    new_score = score + topk_probs[0, i].item()
                    candidates.append((new_seq, new_score))

            beams = sorted(
                candidates,
                key=lambda x: x[1] / (len(x[0][0]) ** length_penalty),
                reverse=True
            )[:beam_width]

            if all(seq[0, -1].item() == self.eos_token_id for seq, _ in beams):
                break

        completed += [b for b in beams if b[0][0, -1].item() == self.eos_token_id]
        best = max(
            completed if completed else beams,
            key=lambda x: x[1] / (len(x[0][0]) ** length_penalty)
        )

        token_ids = best[0][0].tolist() 
        return self.tokenizer.decode(token_ids)






