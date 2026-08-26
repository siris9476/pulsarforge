import numpy as np
for name, oracle, dump in [("tiny", "tools/glm_tiny_hf_logits.npy", "tools/glm_tiny_logits_stream.bin"),
                           ("sparse", "tools/glm_sparse_hf_logits.npy", "tools/glm_sparse_logits_stream.bin")]:
    hf = np.load(oracle)
    ours = np.fromfile(dump, dtype=np.float32).reshape(hf.shape)
    am_ok = int((hf.argmax(axis=1) == ours.argmax(axis=1)).sum())
    cos = np.array([np.dot(hf[i], ours[i]) / (np.linalg.norm(hf[i]) * np.linalg.norm(ours[i])) for i in range(hf.shape[0])])
    print(f"{name}: argmax {am_ok}/{hf.shape[0]}  cos min={cos.min():.6f} mean={cos.mean():.6f}")
