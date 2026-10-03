"""Ứng dụng demo: mô phỏng NMF và tóm tắt văn bản dựa trên chủ đề.
Chạy bằng lệnh: streamlit run app.py"""
import html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.baselines import lead, lsa, textrank
from src.nmf import EPS, TOL, nmf, normalize, top_words
from src.preprocess import preprocess
from src.summarizer import grs_scores, select_grs, select_topic, summarize
from src.vectorize import build_matrix

DATA = Path(__file__).parent / "data"
RESULTS = Path(__file__).parent / "results"
SAMPLES = {
    "Tin mẫu 1: HLV Park Hang-seo và bánh gạo cay": "tin_mau_1",
    "Tin mẫu 2: Doanh nghiệp Nhật chuyển nhà máy sang Việt Nam": "tin_mau_2",
    "Tin mẫu 3: Công nghệ lót ống không đào đường": "tin_mau_3",
}
HIGHLIGHT = "background-color: #ffe08a; color: black"


@st.cache_data
def cached_preprocess(text, min_syllables):
    return preprocess(text, min_syllables=min_syllables)


@st.cache_data
def cached_nmf(V, k, max_iter, seed):
    return nmf(V, k, max_iter=max_iter, seed=seed, save_history=True)


@st.cache_data
def cached_summarize(text, n_sentences, k, method):
    return summarize(text, n_sentences=n_sentences, k=k, method=method)


def show_matrix(M, rows, cols, digits=2):
    st.dataframe(pd.DataFrame(np.round(M, digits), index=rows, columns=cols))


def show_text(sentences, selected):
    """In văn bản gốc, câu được chọn tô vàng, câu bị loại để màu xám."""
    parts = []
    for i, s in enumerate(sentences):
        t = html.escape(s["original"])
        if i in selected:
            parts.append(f"<mark style='{HIGHLIGHT}'>{t}</mark>")
        elif not s["used"]:
            parts.append(f"<span style='color: gray'>{t}</span>")
        else:
            parts.append(t)
    st.html("<div style='line-height: 1.8'>" + " ".join(parts) + "</div>")
    st.caption("Câu tô vàng: được chọn vào bản tóm tắt. Câu màu xám: bị loại "
               "(trùng câu trước, quá ngắn hoặc còn quá ít từ sau khi bỏ stopword).")


def tab_mo_phong():
    st.header("Mô phỏng NMF trên dữ liệu nhỏ")
    default = (DATA / "mo_phong.txt").read_text(encoding="utf-8")
    text = st.text_area("Văn bản (mặc định là 9 câu review điện thoại)", default, height=230)

    c1, c2, c3, c4, c5 = st.columns(5)
    k = c1.selectbox("Số chủ đề k", [2, 3, 4], index=1)
    seed = c2.number_input("Seed", 0, 1000, 0,
                           help="Thử seed 49 để thấy trường hợp NMF gộp chủ đề pin với giá.")
    max_iter = c3.number_input("Số vòng lặp tối đa", 10, 500, 200, step=10)
    method = c4.radio("Cách chọn câu", ["topic", "grs"], horizontal=True)
    n_sent = c5.number_input("Số câu tóm tắt", 1, 5, 3)

    # trang mô phỏng không loại câu ngắn vì câu mẫu đều ngắn
    sentences = cached_preprocess(text, 0)
    if not sentences:
        st.warning("Hãy nhập văn bản.")
        return
    names = [f"S{i + 1}" for i in range(len(sentences))]

    st.subheader("Bước 1. Tách câu")
    st.write("Văn bản được tách thành các câu bằng `underthesea.sent_tokenize`. "
             "Mỗi câu được xem như một tài liệu, ký hiệu S1, S2, ...")
    st.dataframe(pd.DataFrame({"Câu": names, "Nội dung": [s["original"] for s in sentences]}),
                 hide_index=True)

    st.subheader("Bước 2. Tiền xử lý")
    st.write("Tách từ bằng `underthesea.word_tokenize` (từ ghép nối bằng dấu _), đưa về chữ thường, "
             "bỏ dấu câu, số và stopword. Câu còn dưới 2 từ sẽ không đưa vào ma trận.")
    st.dataframe(pd.DataFrame({
        "Câu": names,
        "Sau khi tách từ": [s["tokenized"] for s in sentences],
        "Sau khi bỏ stopword": [" ".join(s["words"]) for s in sentences],
        "Ghi chú": [s["reason"] and f"bị loại ({s['reason']})" for s in sentences],
    }), hide_index=True)

    try:
        mat = build_matrix(sentences)
    except ValueError as e:
        st.error(str(e))
        return
    V, terms = mat["V"], mat["terms"]
    cols = [names[i] for i in mat["sent_idx"]]
    if k > min(V.shape):
        st.error(f"k = {k} lớn hơn số câu hoặc số từ, hãy giảm k.")
        return

    st.subheader("Bước 3. Ma trận đếm và IDF")
    st.write("Hàng là từ, cột là câu. df là số câu chứa từ đó, "
             "idf = ln((1 + n) / (1 + df)) + 1 với n là số câu (công thức IDF làm mịn).")
    c1, c2 = st.columns([3, 1])
    with c1:
        show_matrix(mat["counts"], terms, cols, 0)
    with c2:
        df_count = (mat["counts"] > 0).sum(axis=1)
        st.dataframe(pd.DataFrame({"df": df_count, "idf": np.round(mat["idf"], 3)}, index=terms))

    st.subheader("Bước 4. Ma trận V (TF-IDF)")
    st.write("Bên trái là tf × idf. Bên phải là sau khi chuẩn hóa L2 từng cột (mỗi câu có độ dài "
             "vector bằng 1), để câu dài không lấn át câu ngắn. NMF chạy trên ma trận bên phải.")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("tf × idf")
        show_matrix(mat["V_raw"], terms, cols, 3)
    with c2:
        st.caption("V sau khi chuẩn hóa L2")
        show_matrix(V, terms, cols, 3)

    res = cached_nmf(V, k, int(max_iter), int(seed))
    history, losses = res["history"], res["losses"]
    n_iter = len(losses) - 1
    topics = [f"Chủ đề {j + 1}" for j in range(k)]

    st.subheader("Bước 5. Khởi tạo W, H ngẫu nhiên")
    st.write(f"W ({V.shape[0]} × {k}) và H ({k} × {V.shape[1]}) lấy ngẫu nhiên theo seed {seed}, "
             f"nhân với √(mean(V) / k) để tích WH cùng cỡ với V. Loss ban đầu ‖V − WH‖ = {losses[0]:.4f}.")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("W ban đầu")
        show_matrix(history[0][0], terms, topics, 3)
    with c2:
        st.caption("H ban đầu")
        show_matrix(history[0][1], topics, cols, 3)

    st.subheader("Bước 6. Lặp cập nhật W và H")
    st.write("Mỗi vòng cập nhật H trước rồi tới W theo luật nhân của Lee và Seung: "
             "H ← H × (WᵀV) / (WᵀWH), W ← W × (VHᵀ) / (WHHᵀ). Vì chỉ nhân và chia các số dương "
             "nên W, H luôn không âm.")
    if n_iter < max_iter:
        st.info(f"Dừng ở vòng {n_iter} vì loss gần như không giảm nữa "
                f"(giảm chưa tới {TOL:g} lần so với vòng trước).")
    else:
        st.info(f"Dừng ở vòng {n_iter} vì đã chạy đủ số vòng lặp tối đa.")

    t = st.slider("Xem vòng lặp t", 1, n_iter, 1) if n_iter > 1 else 1
    W_t, H_t = history[t]

    c1, c2 = st.columns([1, 2])
    with c1:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(range(len(losses)), losses, marker=".")
        ax.plot(t, losses[t], "ro")
        ax.axvline(t, color="r", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Vòng lặp")
        ax.set_ylabel("Loss ‖V − WH‖")
        ax.set_title("Loss qua các vòng lặp")
        st.pyplot(fig)
        plt.close(fig)
        st.write(f"Loss ở vòng {t}: **{losses[t]:.4f}**")
    with c2:
        st.caption(f"H ở vòng {t}")
        show_matrix(H_t, topics, cols, 3)
        st.caption(f"W ở vòng {t}")
        show_matrix(W_t, terms, topics, 3)

    # tính tay một phần tử của H để thấy luật cập nhật
    st.write("**Tính tay một phần tử của H** (dùng W, H của vòng trước):")
    c1, c2 = st.columns(2)
    i = c1.selectbox("Chủ đề", range(k), format_func=lambda x: topics[x])
    j = c2.selectbox("Câu", range(len(cols)), format_func=lambda x: cols[x])
    W_old, H_old = history[t - 1]
    num = (W_old.T @ V)[i, j]
    den = (W_old.T @ W_old @ H_old)[i, j] + EPS
    st.code(f"H[{i},{j}] mới = H[{i},{j}] cũ × (WᵀV)[{i},{j}] / (WᵀWH)[{i},{j}]\n"
            f"          = {H_old[i, j]:.4f} × {num:.4f} / {den:.4f} = {H_old[i, j] * num / den:.4f}\n"
            f"(giá trị trong bảng H ở vòng {t}: {H_t[i, j]:.4f})", language=None)

    st.subheader("Bước 7. Kết quả phân rã")
    st.write("Chia mỗi cột W cho tổng của nó (và nhân hàng tương ứng của H lên) để mỗi cột W "
             "có tổng bằng 1, tích WH không đổi. Chủ đề được xếp theo tổng hàng của H, giảm dần.")
    W, H = normalize(history[-1][0], history[-1][1])
    words = top_words(W, terms, 5)
    st.dataframe(pd.DataFrame(
        [[f"{w} ({W[terms.index(w), j]:.2f})" for w in ws] for j, ws in enumerate(words)],
        index=topics, columns=[f"Từ {x + 1}" for x in range(5)]))

    fig, ax = plt.subplots(figsize=(1 + 0.7 * len(cols), 1 + 0.6 * k))
    im = ax.imshow(H, cmap="YlOrRd", aspect="auto")
    for a in range(k):
        for b in range(len(cols)):
            ax.text(b, a, f"{H[a, b]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(k), topics)
    ax.set_xlabel("Câu")
    ax.set_title("Ma trận H (chủ đề × câu)")
    fig.colorbar(im)
    st.pyplot(fig, width="content")
    plt.close(fig)

    st.subheader(f"Bước 8. Chọn câu theo cách {method}")
    n_sent = min(int(n_sent), len(cols))
    w, score = grs_scores(H)
    if method == "topic":
        st.write("Duyệt chủ đề từ quan trọng nhất, mỗi chủ đề lấy câu có giá trị H lớn nhất "
                 "chưa được chọn.")
        chosen = select_topic(H, n_sent)
        st.dataframe(pd.DataFrame({
            "Lượt": range(1, n_sent + 1),
            "Chủ đề": [topics[x % k] for x in range(n_sent)],
            "Câu được chọn": [cols[c] for c in chosen],
            "Giá trị H": [round(H[x % k, c], 3) for x, c in enumerate(chosen)],
        }), hide_index=True)
    else:
        st.write("Trọng số chủ đề w_k = tổng hàng k của H / tổng H. "
                 "Điểm câu score_j = Σ w_k · H[k, j]. Lấy các câu có điểm cao nhất.")
        st.write("Trọng số chủ đề: " + ", ".join(f"{topics[x]}: {w[x]:.3f}" for x in range(k)))
        chosen = select_grs(H, n_sent)

    labels = [cols[c] for c in chosen]
    table = pd.DataFrame({
        "Nội dung": [sentences[s]["original"] for s in mat["sent_idx"]],
        "Chủ đề chính": [topics[x] for x in H.argmax(axis=0)],
        "Điểm GRS": np.round(score, 3),
    }, index=cols)
    st.dataframe(table.style.apply(
        lambda row: [HIGHLIGHT if row.name in labels else ""] * len(row), axis=1
    ).format({"Điểm GRS": "{:.3f}"}))

    selected = sorted(mat["sent_idx"][c] for c in chosen)
    st.write("**Bản tóm tắt:**")
    st.success(" ".join(sentences[s]["original"] for s in selected))


def tab_tom_tat():
    st.header("Tóm tắt văn bản")
    source = st.radio("Nguồn văn bản", ["Bài mẫu", "Dán văn bản", "Tải file .txt"], horizontal=True)
    reference = None
    if source == "Bài mẫu":
        name = st.selectbox("Chọn bài", list(SAMPLES))
        text = (DATA / f"{SAMPLES[name]}.txt").read_text(encoding="utf-8")
        reference = (DATA / f"{SAMPLES[name]}_tomtat.txt").read_text(encoding="utf-8")
    elif source == "Dán văn bản":
        text = st.text_area("Văn bản", height=250)
    else:
        f = st.file_uploader("Chọn file .txt (mã UTF-8)", type="txt")
        text = f.read().decode("utf-8-sig") if f else ""

    c1, c2, c3 = st.columns(3)
    n_sent = c1.slider("Số câu tóm tắt", 1, 10, 3)
    k = c2.slider("Số chủ đề k", 1, 10, 3)
    method = c3.radio("Cách chọn câu", ["topic", "grs"], horizontal=True, key="method_tt")

    if not text.strip():
        st.info("Chưa có văn bản.")
        return
    try:
        res = cached_summarize(text, n_sent, k, method)
    except ValueError as e:
        st.error(str(e))
        return
    sentences = res["sentences"]
    n_used = len(res["sent_idx"])
    if n_used <= n_sent:
        st.warning(f"Văn bản chỉ có {n_used} câu dùng được nên bản tóm tắt lấy hết các câu này.")

    st.subheader("Bản tóm tắt")
    st.success(res["summary"])
    if reference:
        st.write("**Tóm tắt mẫu** (sapo của bài báo, do người viết):")
        st.write(reference)

    st.subheader("Văn bản gốc")
    show_text(sentences, res["selected"])

    st.subheader("Các chủ đề tìm được")
    W, terms = res["W"], res["terms"]
    st.dataframe(pd.DataFrame(
        [[f"{w} ({W[terms.index(w), j]:.2f})" for w in ws] for j, ws in enumerate(res["topics"])],
        index=[f"Chủ đề {j + 1}" for j in range(len(res["topics"]))],
        columns=[f"Từ {x + 1}" for x in range(len(res["topics"][0]))]))

    with st.expander("So sánh với các phương pháp khác"):
        mat = build_matrix(sentences)
        rows = [(f"NMF ({method})", res["selected"])]
        rows += [(f"Lead-{n_sent}", lead(sentences, mat, n_sent)),
                 ("LSA", lsa(sentences, mat, n_sent)),
                 ("TextRank", textrank(sentences, mat, n_sent))]
        for name, idx in rows:
            st.markdown(f"**{name}** (câu {', '.join(str(i + 1) for i in idx)})")
            st.write(" ".join(sentences[i]["original"] for i in idx))
        st.caption("Số thứ tự câu tính từ 1 theo thứ tự trong văn bản gốc.")


def tab_thu_nghiem():
    """Chỉ đọc kết quả đã lưu trong results/, không chạy lại thí nghiệm."""
    st.header("Thử nghiệm trên bộ VietNews")
    if not (RESULTS / "bang_ket_qua.csv").exists():
        st.info("Chưa có kết quả. Hãy chạy `python thu_nghiem.py` trước.")
        return

    st.write("300 bài lấy ngẫu nhiên từ split test của VietNews (seed 42, chỉ giữ bài có 5 đến 40 câu). "
             "Mỗi phương pháp lấy 3 câu, so với sapo của bài báo bằng ROUGE (F1, tính theo âm tiết).")
    with st.expander("Thống kê dữ liệu"):
        st.text((RESULTS / "thong_ke.txt").read_text(encoding="utf-8"))

    st.subheader("Bảng kết quả")
    table = pd.read_csv(RESULTS / "bang_ket_qua.csv", encoding="utf-8-sig")
    table.columns = ["Phương pháp", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Thời gian (ms/bài)"]
    st.dataframe(table, hide_index=True)
    st.caption("Thời gian chỉ tính bước chọn câu, chưa tính tiền xử lý (xem phần thống kê dữ liệu).")
    st.image(str(RESULTS / "rouge_6_phuong_phap.png"))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("NMF theo số chủ đề k")
        st.image(str(RESULTS / "nmf_theo_k.png"))
    with c2:
        st.subheader("Vị trí câu được chọn")
        st.image(str(RESULTS / "vi_tri_cau.png"))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Loss trên một bài")
        st.image(str(RESULTS / "loss_mot_bai.png"))
    with c2:
        st.subheader("NMF với 5 seed (50 bài)")
        st.write("Mỗi seed chỉ chạy NMF một lần, không chọn lần có loss nhỏ nhất.")
        st.dataframe(pd.read_csv(RESULTS / "nmf_theo_seed.csv", encoding="utf-8-sig", index_col=0))

    st.subheader("Nhận xét")
    # bỏ dòng tiêu đề của file
    st.markdown((RESULTS / "nhan_xet.md").read_text(encoding="utf-8").split("\n", 1)[1])
    with st.expander("Ví dụ trên 2 bài"):
        st.markdown((RESULTS / "vi_du.md").read_text(encoding="utf-8"))


st.set_page_config(page_title="NMF và tóm tắt văn bản", layout="wide")
st.title("NMF và tóm tắt văn bản dựa trên chủ đề")
tab1, tab2, tab3 = st.tabs(["Mô phỏng NMF", "Tóm tắt văn bản", "Thử nghiệm"])
with tab1:
    tab_mo_phong()
with tab2:
    tab_tom_tat()
with tab3:
    tab_thu_nghiem()
