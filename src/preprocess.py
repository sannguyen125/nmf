"""Tiền xử lý văn bản tiếng Việt: tách câu, tách từ, bỏ dấu câu, số và stopword."""
import re
from pathlib import Path

from underthesea import sent_tokenize, word_tokenize

STOPWORD_FILE = Path(__file__).resolve().parent.parent / "data" / "stopwords_vi.txt"
MIN_WORDS = 2
MIN_SYLLABLES = 8  # trang mô phỏng đặt 0 vì câu mẫu ngắn


def load_stopwords(path=STOPWORD_FILE):
    with open(path, encoding="utf-8") as f:
        # file gốc viết từ ghép bằng dấu cách, đổi thành _ cho khớp với word_tokenize
        return {line.strip().replace(" ", "_") for line in f if line.strip()}


STOPWORDS = load_stopwords()


def detokenize(text):
    """Văn bản VietNews đã tách từ sẵn (Hưng_Yên, dấu câu cách rời).
    Bỏ _ và ghép lại dấu câu để đưa về dạng bình thường."""
    text = text.replace("_", " ")
    text = re.sub(r" ([,.;:!?)%…])", r"\1", text)
    text = text.replace("( ", "(")
    # ngoặc kép: " abc " -> "abc"
    text = re.sub(r"([\"']) (.*?) \1", r"\1\2\1", text)
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text):
    # tách theo dòng trước, vì có văn bản xuống dòng mà không có dấu chấm
    sentences = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            sentences.extend(sent_tokenize(line))
    return sentences


def clean_words(tokenized, stopwords=STOPWORDS):
    """Từ câu đã tách từ, giữ lại các từ có nghĩa (chữ thường, không dấu câu, không số)."""
    words = []
    for w in tokenized.lower().split():
        w = re.sub(r"[^\w]", "", w).strip("_")
        # \w vẫn giữ chữ số nên phải kiểm tra thêm
        if not w or not w.replace("_", "").isalpha():
            continue
        if w in stopwords:
            continue
        words.append(w)
    return words


def count_syllables(sent):
    # đếm các cụm cách nhau bởi dấu cách có chứa chữ hoặc số, bỏ qua dấu câu đứng riêng
    return sum(1 for w in sent.split() if re.search(r"\w", w))


def dedup_key(sent):
    # khóa để so câu trùng: chữ thường, bỏ dấu câu
    return " ".join(re.sub(r"[^\w\s]", " ", sent.lower()).split())


def preprocess(text, stopwords=STOPWORDS, min_syllables=MIN_SYLLABLES):
    """Trả về danh sách câu, mỗi câu là dict gồm câu gốc, câu đã tách từ và các từ còn lại.
    Câu bị loại (trùng câu trước, ngắn hơn min_syllables âm tiết, hoặc còn dưới MIN_WORDS từ)
    vẫn giữ để hiển thị nhưng used=False, lý do ghi ở reason."""
    result = []
    seen = set()
    for sent in split_sentences(text):
        tokenized = word_tokenize(sent, format="text")
        words = clean_words(tokenized, stopwords)

        key = dedup_key(sent)
        if key in seen:
            reason = "trùng"
        elif count_syllables(sent) < min_syllables:
            reason = "ngắn"
        elif len(words) < MIN_WORDS:
            reason = "ít từ"
        else:
            reason = ""
        seen.add(key)

        result.append({
            "original": sent,
            "tokenized": tokenized,
            "words": words,
            "used": reason == "",
            "reason": reason,
        })
    return result
