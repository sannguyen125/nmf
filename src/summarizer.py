"""Tóm tắt trích xuất dựa trên NMF, có 2 cách chọn câu: topic và grs."""
import numpy as np

from src.nmf import fit_nmf, normalize, top_words
from src.preprocess import MIN_SYLLABLES, preprocess
from src.vectorize import build_matrix


def select_topic(H, n):
    """Duyệt chủ đề từ quan trọng nhất (hàng 0 của H sau normalize), mỗi chủ đề lấy
    câu có H lớn nhất chưa được chọn. Nếu n > k thì quay lại chủ đề đầu."""
    chosen = []
    t = 0
    while len(chosen) < n:
        for j in np.argsort(-H[t % H.shape[0]]):
            if j not in chosen:
                chosen.append(j)
                break
        t += 1
    return chosen


def grs_scores(H):
    """Theo Lee et al. (2009): trọng số chủ đề w_k = tổng hàng k / tổng H,
    điểm câu score_j = tổng w_k * H[k, j]."""
    w = H.sum(axis=1) / H.sum()
    return w, w @ H


def select_grs(H, n):
    """Lấy n câu có điểm GRS cao nhất."""
    w, score = grs_scores(H)
    return list(np.argsort(-score)[:n])


def nmf_select(mat, n, k, method, seed, n_restarts=5):
    """Chạy NMF trên mat["V"] rồi chọn câu. Trả về chỉ số câu đã chọn, W và H."""
    V = mat["V"]
    # văn bản ngắn thì không lấy được nhiều hơn số câu đang có
    n = min(n, V.shape[1])
    k = min(k, V.shape[0], V.shape[1])

    res = fit_nmf(V, k, n_restarts=n_restarts, seed=seed)
    W, H = normalize(res["W"], res["H"])
    if method == "topic":
        cols = select_topic(H, n)
    elif method == "grs":
        cols = select_grs(H, n)
    else:
        raise ValueError("method phải là 'topic' hoặc 'grs'")

    # cột của V chỉ gồm các câu đủ từ, đổi về vị trí câu trong bài
    selected = sorted(mat["sent_idx"][c] for c in cols)
    return selected, W, H


# hai hàm dưới cùng kiểu với các hàm trong baselines.py để dễ chạy thử nghiệm
def nmf_topic(sentences, mat, n=3, k=3, seed=42):
    return nmf_select(mat, n, k, "topic", seed)[0]


def nmf_grs(sentences, mat, n=3, k=3, seed=42):
    return nmf_select(mat, n, k, "grs", seed)[0]


def summarize(text, n_sentences=3, k=3, method="topic", seed=42, min_syllables=MIN_SYLLABLES):
    """Tóm tắt một văn bản. Câu được chọn xếp theo thứ tự trong bài."""
    sentences = preprocess(text, min_syllables=min_syllables)
    if not sentences:
        raise ValueError("Văn bản rỗng")
    mat = build_matrix(sentences)
    selected, W, H = nmf_select(mat, n_sentences, k, method, seed)
    return {
        "sentences": sentences,
        "selected": selected,
        "summary": " ".join(sentences[i]["original"] for i in selected),
        "V": mat["V"],
        "W": W,
        "H": H,
        "terms": mat["terms"],
        "sent_idx": mat["sent_idx"],
        "topics": top_words(W, mat["terms"], 5),
    }
