"""Compare LiteRT decode-walk logits (parity_logits_bigmodel.py lt -> npz key 'lt') with MLX reference logits (npz key 'logits')."""
import sys, numpy as np
ref = np.load(sys.argv[1])["logits"]; lt = np.load(sys.argv[2])["lt"]; m = min(len(ref), len(lt)); ref, lt = ref[:m], lt[:m]
def softmax(x): x = x - x.max(-1, keepdims=True); e = np.exp(x); return e / e.sum(-1, keepdims=True)
top1 = (ref.argmax(-1) == lt.argmax(-1)).mean(); P, Q = softmax(ref), softmax(lt)
kl = np.mean(np.sum(P * (np.log(P + 1e-9) - np.log(Q + 1e-9)), -1)); r = np.mean([np.corrcoef(ref[i], lt[i])[0, 1] for i in range(m)])
print(f"positions {m} | top-1 agreement {100*top1:.1f}% | mean KL(mlx||lt) {kl:.4f} | mean Pearson {r:.5f} | max|diff| {np.abs(ref-lt).max():.3f}")
for i in range(m): print(f"  pos {i:2d} mlx top1 {int(ref[i].argmax()):6d} lt top1 {int(lt[i].argmax()):6d} corr {np.corrcoef(ref[i], lt[i])[0,1]:.5f}")
