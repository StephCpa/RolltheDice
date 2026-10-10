"""转储训练起始状态的关键配置与首批损失（用于跨环境 diff，采纳评审建议）。"""
import json, sys, torch
import torch.nn.functional as F
sys.path.insert(0, '.')
from train import encode_raw, pad_batch, mtp_seq, ntp_seq, MAX_LEN
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
out = {"torch": torch.__version__, "cuda": torch.version.cuda}
import transformers; out["transformers"] = transformers.__version__
tok = GPT2TokenizerFast.from_pretrained('gpt2'); tok.pad_token = tok.eos_token
vocab = json.load(open('data/circle.10.9.10.10000/vocab.json', encoding='utf-8'))
tok.add_tokens(vocab)
model = GPT2LMHeadModel.from_pretrained('gpt2'); model.resize_token_embeddings(len(tok))
cfg = model.config
out["cfg"] = {k: getattr(cfg, k, None) for k in
              ["resid_pdrop","embd_pdrop","attn_pdrop","summary_first_dropout",
               "layer_norm_epsilon","n_layer","n_head","n_embd"]}
out["attn_impl"] = getattr(model.config, "_attn_implementation", "n/a")
dev = "cuda" if torch.cuda.is_available() else "cpu"
model.to(dev)
torch.manual_seed(42)
raw = json.load(open('data/circle.10.9.10.10000/train.json', encoding='utf-8'))[:64]
exs = encode_raw(raw, tok)
seqs = [ntp_seq(p, a) for p, a in exs]
inp, lab = pad_batch(seqs, tok.eos_token_id)
inp, lab = inp.to(dev), lab.to(dev)
model.eval()
with torch.no_grad():
    with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev=="cuda")):
        loss = model(inp, labels=lab).loss.item()
out["first64_npt_loss"] = round(loss, 6)
# 一步 AdamW 更新后的损失变化（捕获优化器/调度差异）
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev=="cuda")):
    l2 = model(inp, labels=lab).loss
opt.zero_grad(); l2.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()
with torch.no_grad():
    with torch.autocast(device_type=dev, dtype=torch.float16, enabled=(dev=="cuda")):
        l3 = model(inp, labels=lab).loss.item()
out["loss_after_1_step"] = round(l3, 6)
out["device"] = torch.cuda.get_device_name(0) if dev=="cuda" else "cpu"
print(json.dumps(out, indent=1))
