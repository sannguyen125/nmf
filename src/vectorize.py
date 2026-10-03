"""Tạo ma trận V (từ x câu) từ các câu đã tiền xử lý."""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_matrix(sentences):
    """Mỗi câu là một "tài liệu". Trả về dict gồm V (TF-IDF), ma trận đếm, idf, terms
    và sent_idx là vị trí trong danh sách câu ứng với từng cột của V."""
    sent_idx = [i for i, s in enumerate(sentences) if s["used"]]
    if not sent_idx:
        raise ValueError("Không có câu nào đủ từ để tạo ma trận")
    docs = [" ".join(sentences[i]["words"]) for i in sent_idx]

    # văn bản đã tách từ rồi nên chỉ cần split theo khoảng trắng
    count_vec = CountVectorizer(analyzer=str.split)
    counts = count_vec.fit_transform(docs).toarray()

    # norm=None để V đúng bằng tf * idf như công thức trong slide, dễ kiểm tra bằng tay
    tfidf_vec = TfidfVectorizer(analyzer=str.split, norm=None)
    V = tfidf_vec.fit_transform(docs).toarray()

    # sklearn để câu ở hàng, chuyển vị cho giống slide: hàng là từ, cột là câu
    return {
        "V": V.T,
        "counts": counts.T,
        "idf": tfidf_vec.idf_,
        "terms": list(tfidf_vec.get_feature_names_out()),
        "sent_idx": sent_idx,
    }
