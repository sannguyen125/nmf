"""NMF bằng luật cập nhật nhân (Lee & Seung 2001): V (m x n) ~ W (m x k) . H (k x n)."""
import numpy as np

EPS = 1e-10
MAX_ITER = 200
TOL = 1e-4


def init_wh(V, k, seed):
    rng = np.random.default_rng(seed)
    m, n = V.shape
    scale = np.sqrt(V.mean() / k)
    W = rng.random((m, k)) * scale
    H = rng.random((k, n)) * scale
    return W, H


def frob_loss(V, W, H):
    return np.linalg.norm(V - W @ H, "fro")


def nmf(V, k, max_iter=MAX_ITER, seed=0, save_history=False, tol=TOL):
    """Chạy NMF một lần. Dừng khi hết max_iter hoặc loss giảm chưa tới tol (tương đối).
    Nếu save_history=True thì lưu W, H ở mọi vòng (vòng 0 là lúc khởi tạo)."""
    W, H = init_wh(V, k, seed)
    losses = [frob_loss(V, W, H)]
    history = [(W.copy(), H.copy())] if save_history else None

    for it in range(max_iter):
        # cập nhật H trước rồi mới tới W
        H = H * (W.T @ V) / (W.T @ W @ H + EPS)
        W = W * (V @ H.T) / (W @ H @ H.T + EPS)

        losses.append(frob_loss(V, W, H))
        if save_history:
            history.append((W.copy(), H.copy()))
        if losses[-2] - losses[-1] < tol * losses[-2]:
            break

    return {"W": W, "H": H, "losses": losses, "history": history, "seed": seed}


def fit_nmf(V, k, n_restarts=5, max_iter=MAX_ITER, seed=0):
    """NMF phụ thuộc điểm khởi tạo nên chạy n_restarts lần với các seed khác nhau,
    lấy lần có loss cuối nhỏ nhất."""
    best = None
    for s in range(seed, seed + n_restarts):
        res = nmf(V, k, max_iter=max_iter, seed=s)
        if best is None or res["losses"][-1] < best["losses"][-1]:
            best = res
    return best


def normalize(W, H):
    """Chia mỗi cột W cho tổng cột đó và nhân hàng tương ứng của H lên, tích WH không đổi.
    Sau đó xếp chủ đề theo tổng hàng của H, giảm dần."""
    s = W.sum(axis=0)
    W = W / s
    H = H * s[:, None]
    order = np.argsort(-H.sum(axis=1))
    return W[:, order], H[order]


def top_words(W, terms, n=5):
    """Với mỗi chủ đề (cột W), lấy n từ có trọng số lớn nhất."""
    result = []
    for j in range(W.shape[1]):
        idx = np.argsort(-W[:, j])[:n]
        result.append([terms[i] for i in idx])
    return result
