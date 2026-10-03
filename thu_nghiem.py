"""Phần 3: thử nghiệm trên 300 bài của bộ VietNews, ghi kết quả ra thư mục results/.
Chạy: python thu_nghiem.py (mất vài phút, lần đầu phải tải dữ liệu từ HuggingFace)."""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_dataset

from src.baselines import lead, lsa, random_pick, textrank
from src.evaluate import rouge
from src.nmf import nmf, top_words
from src.preprocess import count_syllables, detokenize, preprocess, split_sentences
from src.summarizer import nmf_grs, nmf_select, nmf_topic
from src.vectorize import build_matrix

N_DOCS = 300
SEED = 42
MIN_SENT, MAX_SENT = 5, 40
N = 3  # số câu tóm tắt
RESULTS = Path(__file__).parent / "results"
# vị trí trong split test của 2 bài làm ví dụ, chọn tay để tránh tin bạo lực
EXAMPLES = [18425, 16001]
METHODS = {
    "NMF-topic": nmf_topic,
    "NMF-grs": nmf_grs,
    "LSA": lsa,
    "TextRank": textrank,
    "Lead-3": lead,
    "Random-3": random_pick,
}


def load_docs():
    """Xáo trộn split test theo seed 42, lấy lần lượt các bài có 5-40 câu cho đủ 300 bài."""
    ds = load_dataset("harouzie/vietnews", split="test", cache_dir="data/raw")
    docs = []
    for i in np.random.default_rng(SEED).permutation(len(ds)):
        row = ds[int(i)]
        text = detokenize(row["article"])
        if MIN_SENT <= len(split_sentences(text)) <= MAX_SENT:
            docs.append({
                "idx": int(i),
                "guid": row["guid"],
                "title": detokenize(row["title"]),
                "text": text,
                "abstract": detokenize(row["abstract"]),
            })
        if len(docs) == N_DOCS:
            break
    return docs


def summary_text(sentences, idx):
    return " ".join(sentences[i]["original"] for i in idx)


def mean_rouge(docs, get_idx):
    """ROUGE trung bình trên các bài, get_idx(d) trả về các câu được chọn của bài d."""
    scores = [rouge(summary_text(d["sentences"], get_idx(d)), d["abstract"]) for d in docs]
    return pd.DataFrame(scores).mean()


def fig_rouge(table):
    ax = table[["rouge1", "rouge2", "rougeL"]].plot.bar(figsize=(8, 4.5), rot=0)
    ax.set_xlabel("Phương pháp")
    ax.set_ylabel("F1")
    ax.set_title(f"ROUGE của 6 phương pháp ({N_DOCS} bài VietNews)")
    ax.set_ylim(0, table["rouge1"].max() * 1.25)
    ax.legend(["ROUGE-1", "ROUGE-2", "ROUGE-L"], ncol=3, loc="upper center")
    plt.tight_layout()
    plt.savefig(RESULTS / "rouge_6_phuong_phap.png", dpi=120)
    plt.close()


def fig_k(df_k, lead_r1):
    plt.figure(figsize=(6, 4))
    for method in ["NMF-topic", "NMF-grs"]:
        plt.plot(df_k["k"], df_k[method], marker="o", label=method)
    plt.axhline(lead_r1, color="gray", linestyle="--", label="Lead-3")
    plt.xticks(df_k["k"])
    plt.xlabel("Số chủ đề k")
    plt.ylabel("ROUGE-1 (F1)")
    plt.title("ROUGE-1 của NMF theo số chủ đề k")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.savefig(RESULTS / "nmf_theo_k.png", dpi=120)
    plt.close()


def fig_position(per_doc):
    """Vị trí tương đối của câu được chọn: chia bài thành 3 phần đầu, giữa, cuối."""
    parts = ["Đầu bài", "Giữa bài", "Cuối bài"]
    data = {}
    for method in ["Lead-3", "NMF-topic", "NMF-grs"]:
        count = np.zeros(3)
        for _, row in per_doc[per_doc["phuong_phap"] == method].iterrows():
            for i in map(int, row["cau_chon"].split()):
                count[min(int(3 * i / row["so_cau"]), 2)] += 1
        data[method] = count / count.sum()
    df_pos = pd.DataFrame(data, index=parts)
    ax = df_pos.plot.bar(figsize=(6, 4), rot=0)
    ax.set_xlabel("Vị trí trong bài")
    ax.set_ylabel("Tỉ lệ câu được chọn")
    ax.set_title("Vị trí các câu được chọn")
    plt.tight_layout()
    plt.savefig(RESULTS / "vi_tri_cau.png", dpi=120)
    plt.close()
    return df_pos


def fig_loss(d):
    losses = nmf(d["mat"]["V"], 3, seed=SEED)["losses"]
    plt.figure(figsize=(6, 4))
    plt.plot(losses, marker=".")
    plt.xlabel("Vòng lặp")
    plt.ylabel("Loss ‖V − WH‖")
    plt.title(f"Loss của NMF trên bài {d['idx']} (k = 3)")
    plt.tight_layout()
    plt.savefig(RESULTS / "loss_mot_bai.png", dpi=120)
    plt.close()
    return losses


def write_examples(docs_by_idx):
    lines = ["# Ví dụ tóm tắt trên 2 bài thử nghiệm", ""]
    for idx in EXAMPLES:
        d = docs_by_idx[idx]
        sents = d["sentences"]
        chosen, W, H = nmf_select(d["mat"], N, 3, "topic", SEED)
        lines += [f"## Bài {idx}: {d['title']}", "",
                  f"Số câu: {len(sents)}, số câu dùng được: {len(d['mat']['sent_idx'])}.", "",
                  "Chủ đề NMF tìm được (k = 3):", ""]
        for j, ws in enumerate(top_words(W, d["mat"]["terms"], 5)):
            lines.append(f"- Chủ đề {j + 1}: {', '.join(ws)}")
        lines.append("")
        for name, idx_sel in [("NMF-topic", chosen),
                              ("NMF-grs", nmf_grs(sents, d["mat"], N)),
                              ("Lead-3", lead(sents, d["mat"], N))]:
            text = summary_text(sents, idx_sel)
            r = rouge(text, d["abstract"])
            lines += [f"**{name}** (câu {', '.join(str(i + 1) for i in idx_sel)}; "
                      f"ROUGE-1 = {r['rouge1']:.4f}, ROUGE-L = {r['rougeL']:.4f}):", "", text, ""]
        lines += ["**Tóm tắt mẫu (abstract):**", "", d["abstract"], ""]
    (RESULTS / "vi_du.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    # để in được tiếng Việt khi chạy trên Windows
    sys.stdout.reconfigure(encoding="utf-8")
    RESULTS.mkdir(exist_ok=True)
    stats = []

    print("Đang lấy dữ liệu và tiền xử lý...")
    docs = load_docs()
    t0 = time.perf_counter()
    for d in docs:
        d["sentences"] = preprocess(d["text"])
        d["mat"] = build_matrix(d["sentences"])
    t_pre = (time.perf_counter() - t0) / len(docs) * 1000
    pd.DataFrame([{
        "idx": d["idx"], "guid": d["guid"], "tieu_de": d["title"],
        "so_cau": len(d["sentences"]), "so_cau_dung_duoc": len(d["mat"]["sent_idx"]),
    } for d in docs]).to_csv(RESULTS / "bai_thu_nghiem.csv", index=False, encoding="utf-8-sig")

    stats.append(f"Số bài: {len(docs)}")
    stats.append(f"Số câu trung bình mỗi bài: {np.mean([len(d['sentences']) for d in docs]):.2f}")
    stats.append("Số câu dùng được trung bình mỗi bài: "
                 f"{np.mean([len(d['mat']['sent_idx']) for d in docs]):.2f}")
    stats.append(f"Số âm tiết trung bình mỗi bài: {np.mean([count_syllables(d['text']) for d in docs]):.2f}")
    stats.append("Số câu trung bình của abstract: "
                 f"{np.mean([len(split_sentences(d['abstract'])) for d in docs]):.2f}")
    stats.append(f"Số âm tiết trung bình của abstract: {np.mean([count_syllables(d['abstract']) for d in docs]):.2f}")
    stats.append(f"Thời gian tiền xử lý trung bình mỗi bài (tách câu, tách từ, tạo V): {t_pre:.1f} ms")
    print("\n".join(stats))

    # kiểm tra nhanh: Lead-3 trên VietNews thường có ROUGE-1 khoảng 0.25
    r1 = mean_rouge(docs[:30], lambda d: lead(d["sentences"], d["mat"], N))["rouge1"]
    print(f"\nKiểm tra nhanh: ROUGE-1 của Lead-3 trên 30 bài đầu = {r1:.4f}")
    if r1 < 0.15:
        print("ROUGE-1 của Lead-3 thấp bất thường, có thể ROUGE đang tính sai tiếng Việt. Dừng lại.")
        sys.exit(1)

    print("\nChạy 6 phương pháp...")
    rows = []
    for d in docs:
        for name, f in METHODS.items():
            t0 = time.perf_counter()
            idx = f(d["sentences"], d["mat"], N)
            t = (time.perf_counter() - t0) * 1000
            rows.append({
                "idx": d["idx"], "phuong_phap": name,
                **rouge(summary_text(d["sentences"], idx), d["abstract"]),
                "thoi_gian_ms": t, "cau_chon": " ".join(map(str, idx)), "so_cau": len(d["sentences"]),
            })
    per_doc = pd.DataFrame(rows)
    per_doc.to_csv(RESULTS / "diem_tung_bai.csv", index=False, encoding="utf-8-sig")
    table = per_doc.groupby("phuong_phap", sort=False)[["rouge1", "rouge2", "rougeL", "thoi_gian_ms"]].mean()
    table = table.round(4)
    table.index.name = "phuong_phap"
    table.to_csv(RESULTS / "bang_ket_qua.csv", encoding="utf-8-sig")
    print(table)
    fig_rouge(table)

    print("\nNMF với k = 2, 3, 4, 5...")
    rows = []
    for k in [2, 3, 4, 5]:
        row = {"k": k}
        for method in ["topic", "grs"]:
            row[f"NMF-{method}"] = mean_rouge(
                docs, lambda d: nmf_select(d["mat"], N, k, method, SEED)[0])["rouge1"]
        rows.append(row)
    df_k = pd.DataFrame(rows).round(4)
    df_k.to_csv(RESULTS / "nmf_theo_k.csv", index=False, encoding="utf-8-sig")
    print(df_k)
    fig_k(df_k, table.loc["Lead-3", "rouge1"])

    df_pos = fig_position(per_doc).round(4)
    df_pos.to_csv(RESULTS / "vi_tri_cau.csv", encoding="utf-8-sig")
    print("\nVị trí câu được chọn:")
    print(df_pos)

    # NMF phụ thuộc điểm khởi tạo: mỗi seed chỉ chạy 1 lần, không lấy loss nhỏ nhất
    print("\nNMF với 5 seed trên 50 bài...")
    rows = []
    chosen = {}
    for s in range(5):
        row = {"seed": s}
        for method in ["topic", "grs"]:
            sel = {d["idx"]: nmf_select(d["mat"], N, 3, method, s, n_restarts=1)[0] for d in docs[:50]}
            chosen[s, method] = sel
            r = mean_rouge(docs[:50], lambda d: sel[d["idx"]])
            row[f"NMF-{method} R1"] = r["rouge1"]
            row[f"NMF-{method} R2"] = r["rouge2"]
            row[f"NMF-{method} RL"] = r["rougeL"]
        rows.append(row)
    df_seed = pd.DataFrame(rows).set_index("seed")
    df_seed.loc["độ lệch chuẩn"] = df_seed.std()
    df_seed = df_seed.round(4)
    df_seed.to_csv(RESULTS / "nmf_theo_seed.csv", encoding="utf-8-sig")
    print(df_seed)
    for method in ["topic", "grs"]:
        n_diff = sum(len({tuple(chosen[s, method][d["idx"]]) for s in range(5)}) > 1 for d in docs[:50])
        line = (f"NMF-{method}: {n_diff}/50 bài có tập câu được chọn thay đổi giữa 5 seed "
                f"({n_diff / 50:.0%})")
        stats.append(line)
        print(line)

    # 3 câu được chọn thuộc bao nhiêu chủ đề khác nhau (chủ đề có H lớn nhất ở mỗi câu)
    for method in ["topic", "grs"]:
        n_topics = []
        for d in docs:
            sel, W, H = nmf_select(d["mat"], N, 3, method, SEED)
            cols = [d["mat"]["sent_idx"].index(i) for i in sel]
            n_topics.append(len(set(H[:, cols].argmax(axis=0))))
        n_topics = np.array(n_topics)
        line = (f"NMF-{method}: 3 câu được chọn thuộc trung bình {n_topics.mean():.2f} chủ đề khác nhau, "
                f"{(n_topics == 1).mean():.2%} số bài có cả 3 câu cùng một chủ đề")
        stats.append(line)
        print(line)

    docs_by_idx = {d["idx"]: d for d in docs}
    losses = fig_loss(docs_by_idx[EXAMPLES[0]])
    stats.append(f"Hình loss: bài {EXAMPLES[0]}, dừng sau {len(losses) - 1} vòng, "
                 f"loss từ {losses[0]:.4f} xuống {losses[-1]:.4f}")
    write_examples(docs_by_idx)

    (RESULTS / "thong_ke.txt").write_text("\n".join(stats) + "\n", encoding="utf-8")
    print("\nĐã ghi kết quả vào thư mục results/")


if __name__ == "__main__":
    main()
