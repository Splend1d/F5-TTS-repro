"""
Frozen text-encoder token vectors placed on F5-TTS text positions (semantic conditioning, "option B").

Two kinds of encoder, chosen by the name in the config (`model.sem_encoder`):

- a PyLate ColBERT model id (e.g. lightonai/GTE-ModernColBERT-v1): exactly PyLate's document encoding,
  `encode(texts, is_query=False)`: same "[D] " prefix, tokenizer, length limit, fp32 forward and L2 normalization.
  `output_value=None` returns every position plus the mask the default call filters with (skiplist punctuation and
  padding removed), so nothing is recomputed here.
- "hf:<model id>" (e.g. hf:answerdotai/ModernBERT-base): a plain transformers encoder, its last hidden layer, with
  the same text, the same punctuation skiplist and the same placement, for comparing against an encoder that was
  not trained for retrieval.

Placement: the F5 text (a list of characters, or pinyin syllables) is joined into the string the encoder reads, and
each kept token's vector is copied onto every F5 position whose characters the token covers. A BPE word token
includes its leading space, so the space before a word carries that word's vector. Positions with no vector
(dropped punctuation, characters of no kept token, padding) get mask False and add nothing. Special tokens ([CLS],
[D], [SEP]) cover no characters and are not placed. When several tokens cover one F5 position, the position keeps
the first: the byte-level tokenizer splits a rare character (e.g. a math italic letter) into byte pieces that all
cover that character, and a pinyin syllable ("wo3" -> "wo", "3") is a single F5 position. Texts longer than the
encoder's length limit are truncated the way the encoder itself truncates; positions past the cut get no vector.

Encoders are looked up by name in a per-process cache instead of living inside the model, so they are never trained,
saved in checkpoints, cast to the model dtype, or copied into the EMA model.
"""

from __future__ import annotations

import string

import torch


_ENCODERS: dict[str, TokenSemantics] = {}


def get_semantic_encoder(name: str) -> TokenSemantics:
    if name not in _ENCODERS:
        _ENCODERS[name] = HFTokenSemantics(name[3:]) if name.startswith("hf:") else ColBERTSemantics(name)
    return _ENCODERS[name]


class TokenSemantics:
    """Places per-token vectors on F5 positions; subclasses encode the strings."""

    tokenizer = None
    dim: int
    special_ids: set[int]

    def encode(
        self, strings: list[str], device
    ) -> list[tuple[list[int], list[tuple[int, int]], list[bool], torch.Tensor]]:
        """Per string: token ids, character spans in the string, keep flags, token vectors [len(ids), dim]."""
        raise NotImplementedError

    @torch.no_grad()
    def __call__(self, text: list[list[str]], device) -> tuple[torch.Tensor, torch.Tensor]:
        """text: F5 text as lists of characters -> sem [b, nt, dim] float32, sem_mask [b, nt] bool."""
        device = torch.device(device)
        strings = ["".join(t) for t in text]
        owners = [[k for k, e in enumerate(t) for _ in e] for t in text]  # string char index -> F5 position

        # the frozen encoder stays fp32 even inside the trainer's bf16 autocast
        with torch.autocast(device_type=device.type, enabled=False):
            encoded = self.encode(strings, device)

        nt = max(len(t) for t in text)
        sem = torch.zeros(len(text), nt, self.dim, device=device)
        sem_mask = torch.zeros(len(text), nt, dtype=torch.bool, device=device)
        for i, (s, (ids, spans, keep, emb)) in enumerate(zip(strings, encoded)):
            pos_token = [-1] * len(text[i])
            for t, (tok_id, (a, b)) in enumerate(zip(ids, spans)):
                if not keep[t] or tok_id in self.special_ids:
                    continue
                if a == b:
                    raise RuntimeError(
                        f"kept token {self.tokenizer.convert_ids_to_tokens(tok_id)!r} covers no text in {s!r}"
                    )
                for c in range(a, b):  # a position several tokens cover keeps the first one
                    if pos_token[owners[i][c]] < 0:
                        pos_token[owners[i][c]] = t

            idx = torch.tensor(pos_token, device=device)
            has = idx >= 0
            sem[i, : len(pos_token)] = torch.where(has.unsqueeze(-1), emb.to(device)[idx.clamp(min=0)], 0.0)
            sem_mask[i, : len(pos_token)] = has
        return sem, sem_mask


class ColBERTSemantics(TokenSemantics):
    def __init__(self, name: str):
        from pylate import models

        self.name = name
        self.model = models.ColBERT(name, device="cpu")
        self.model.eval().requires_grad_(False)
        self.tokenizer = self.model.tokenizer
        self.prefix_id = self.model.document_prefix_id
        self.max_length = self.model.document_length
        self.dim = self.model[-1].linear.out_features
        self.special_ids = set(self.tokenizer.all_special_ids) | {self.prefix_id}

    def encode(self, strings, device):
        out = self.model.encode(
            strings,
            is_query=False,
            output_value=None,
            convert_to_tensor=True,
            batch_size=len(strings),
            show_progress_bar=False,
            device=str(device),
        )
        # sentence-transformers tokenizes str(s).strip(), truncated to document_length - 1 (room for the prefix);
        # offsets of the stripped text shift by the leading whitespace
        offsets = self.tokenizer(
            [s.strip() for s in strings],
            add_special_tokens=True,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length - 1,
        )
        encoded = []
        for i, s in enumerate(strings):
            lead = len(s) - len(s.lstrip())
            # PyLate inserts the document prefix right after [CLS]
            ids = offsets["input_ids"][i]
            ids = ids[:1] + [self.prefix_id] + ids[1:]
            spans = [(a + lead, b + lead) if b > a else (a, b) for a, b in offsets["offset_mapping"][i]]
            spans = spans[:1] + [(0, 0)] + spans[1:]
            enc_ids = out["input_ids"][i].tolist()
            if enc_ids[: len(ids)] != ids or any(out["attention_mask"][i][len(ids) :].tolist()):
                raise RuntimeError(f"tokenizer offsets do not match the ids PyLate encoded for {s!r}")
            encoded.append((ids, spans, out["masks"][i].tolist(), out["token_embeddings"][i]))
        return encoded


class HFTokenSemantics(TokenSemantics):
    """Last hidden layer of a plain transformers encoder, read like the ColBERT documents: the stripped text with the
    tokenizer's special tokens, PyLate's default punctuation skiplist dropped, no projection or normalization."""

    def __init__(self, name: str):
        from transformers import AutoModel, AutoTokenizer

        self.name = name
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.model.eval().requires_grad_(False)
        self.max_length = self.model.config.max_position_embeddings
        self.dim = self.model.config.hidden_size
        self.special_ids = set(self.tokenizer.all_special_ids)
        self.skiplist = {self.tokenizer.convert_tokens_to_ids(w) for w in string.punctuation}

    def encode(self, strings, device):
        self.model.to(device)
        batch = self.tokenizer(
            [s.strip() for s in strings],
            add_special_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        offsets = batch.pop("offset_mapping").tolist()
        hidden = self.model(**batch.to(device)).last_hidden_state
        encoded = []
        for i, s in enumerate(strings):
            lead = len(s) - len(s.lstrip())
            n = int(batch["attention_mask"][i].sum())  # right padding: the first n positions are the text
            ids = batch["input_ids"][i, :n].tolist()
            spans = [(a + lead, b + lead) if b > a else (a, b) for a, b in offsets[i][:n]]
            keep = [tok_id not in self.skiplist for tok_id in ids]
            encoded.append((ids, spans, keep, hidden[i, :n]))
        return encoded
