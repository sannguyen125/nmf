"""Tiền xử lý văn bản tiếng Việt: tách câu, tách từ, bỏ dấu câu, số và stopword."""
import re
from pathlib import Path

from underthesea import sent_tokenize, word_tokenize

STOPWORD_FILE = Path(__file__).resolve().parent.parent / "data" / "stopwords_vi.txt"
MIN_WORDS = 2


def load_stopwords(path=STOPWORD_FILE):
    with open(path, encoding="utf-8") as f:
        # file gốc viết từ ghép bằng dấu cách, đổi thành _ cho khớp với word_tokenize
        return {line.strip().replace(" ", "_") for line in f if line.strip()}


STOPWORDS = load_stopwords()


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


def preprocess(text, stopwords=STOPWORDS):
    """Trả về danh sách câu, mỗi câu là dict gồm câu gốc, câu đã tách từ và các từ còn lại.
    Câu còn dưới MIN_WORDS từ thì used=False, không đưa vào ma trận."""
    result = []
    for sent in split_sentences(text):
        tokenized = word_tokenize(sent, format="text")
        words = clean_words(tokenized, stopwords)
        result.append({
            "original": sent,
            "tokenized": tokenized,
            "words": words,
            "used": len(words) >= MIN_WORDS,
        })
    return result
