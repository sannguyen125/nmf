"""Vài kiểm tra nhỏ cho hàm NMF tự viết trong src/nmf.py."""
import warnings

import numpy as np
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning

from src.nmf import init_wh, nmf, normalize

# chạy đủ 200 vòng nên sklearn sẽ báo chưa hội tụ, bỏ qua cảnh báo này
warnings.filterwarnings("ignore", category=ConvergenceWarning)

V = np.random.default_rng(1).random((12, 8))
k = 3
n_iter = 200

# tol=0 để chạy đủ số vòng, cho dễ so với sklearn
res = nmf(V, k, max_iter=n_iter, seed=0, tol=0)
W, H, losses = res["W"], res["H"], res["losses"]

print("1. W, H không âm:", bool((W >= 0).all() and (H >= 0).all()))

diffs = np.diff(losses)
print("2. Loss không tăng qua các vòng:", bool((diffs <= 1e-12).all()))
print(f"   loss ban đầu {losses[0]:.4f}, sau {len(losses) - 1} vòng {losses[-1]:.4f}")

# sklearn phân rã X = V.T, nên W của sklearn ứng với H.T của mình và ngược lại
W0, H0 = init_wh(V, k, seed=0)
sk = NMF(n_components=k, init="custom", solver="mu", max_iter=n_iter, tol=1e-12)
sk.fit_transform(V.T, W=H0.T.copy(), H=W0.T.copy())
diff = abs(losses[-1] - sk.reconstruction_err_) / sk.reconstruction_err_
print(f"3. Loss của mình {losses[-1]:.4f}, sklearn {sk.reconstruction_err_:.4f} "
      f"(sau {sk.n_iter_} vòng), lệch {diff:.2e}:", bool(diff < 1e-3))

W2, H2 = normalize(W, H)
print("4. Sau normalize tích WH không đổi:", bool(np.allclose(W @ H, W2 @ H2)))
print("   tổng mỗi cột W:", np.round(W2.sum(axis=0), 4))
