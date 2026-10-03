"""Các phương pháp để so sánh với NMF: Lead-3, Random-3, LSA, TextRank.
Hàm nào cũng nhận (sentences, mat, n) và trả về chỉ số câu, xếp theo thứ tự trong bài."""
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def lead(sentences, mat, n=3):
    return list(range(min(n, len(sentences))))


def random_pick(sentences, mat, n=3, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(sentences), size=min(n, len(sentences)), replace=False)
    return sorted(int(i) for i in idx)


def lsa(sentences, mat, n=3):
    """Gong & Liu (2001): SVD của V, với mỗi vector kỳ dị lấy câu có trị tuyệt đối lớn nhất."""
    V = mat["V"]
    n = min(n, V.shape[1])
    U, S, Vt = np.linalg.svd(V, full_matrices=False)

    chosen = []
    for i in range(n):
        # hàng i của Vt cho biết mỗi câu gắn với vector kỳ dị thứ i bao nhiêu.
        # SVD có giá trị âm và dấu của vector là tùy ý nên phải xét trị tuyệt đối
        row = np.abs(Vt[i % Vt.shape[0]])
        for j in np.argsort(-row):
            if j not in chosen:
                chosen.append(j)
                break
    return sorted(mat["sent_idx"][j] for j in chosen)


def textrank(sentences, mat, n=3):
    """Đồ thị câu, trọng số cạnh là cosine giữa các cột của V, xếp hạng bằng PageRank."""
    V = mat["V"]
    n = min(n, V.shape[1])
    sim = cosine_similarity(V.T)
    np.fill_diagonal(sim, 0)

    scores = nx.pagerank(nx.from_numpy_array(sim))
    best = sorted(scores, key=scores.get, reverse=True)[:n]
    return sorted(mat["sent_idx"][j] for j in best)
