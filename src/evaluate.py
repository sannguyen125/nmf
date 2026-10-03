"""Đánh giá bản tóm tắt bằng ROUGE-1, ROUGE-2, ROUGE-L (F1), tính theo âm tiết."""
import re

from rouge_score import rouge_scorer


class SyllableTokenizer:
    """Tokenizer mặc định của rouge-score xóa mọi ký tự ngoài a-z0-9 nên làm hỏng tiếng Việt.
    Ở đây chỉ đưa về chữ thường, bỏ dấu câu và tách theo khoảng trắng (mỗi token là một âm tiết)."""

    def tokenize(self, text):
        text = text.lower().replace("_", " ")
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()


scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=SyllableTokenizer())


def rouge(summary, reference):
    """Trả về dict F1 của rouge1, rouge2, rougeL."""
    scores = scorer.score(reference, summary)
    return {name: s.fmeasure for name, s in scores.items()}
