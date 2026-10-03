# NMF và tóm tắt văn bản dựa trên chủ đề

Bài tập lớn môn Phân tích và khai phá dữ liệu văn bản (PTIT). Chương trình gồm 3 phần theo đề bài:

1. Mô phỏng thuật toán NMF từng bước trên 9 câu review điện thoại.
2. Dùng NMF để tóm tắt trích xuất văn bản tiếng Việt (chọn câu theo chủ đề).
3. Thử nghiệm trên 300 bài của bộ VietNews, so sánh với Lead-3, Random-3, LSA, TextRank bằng ROUGE.

Cả ba phần dùng chung một hàm NMF tự cài bằng NumPy trong `src/nmf.py` (luật cập nhật nhân của Lee và Seung).

## Cài đặt

Cần Python 3.11 trở lên. Trên Windows nên để project ở đường dẫn ngắn (ví dụ `D:\nmf`), vì gói jupyter có file nằm rất sâu, đường dẫn dài quá 260 ký tự thì pip sẽ báo lỗi `No such file or directory`.

```
python -m venv .venv
.venv\Scripts\activate          (Windows)
source .venv/bin/activate       (Linux, macOS)
pip install -r requirements.txt
```

## Chạy chương trình

```
streamlit run app.py      # mở app ở http://localhost:8501, có 3 tab ứng với 3 phần
python kiem_tra.py        # kiểm tra NMF: W, H không âm, loss giảm dần, so với sklearn
python thu_nghiem.py      # chạy lại phần thử nghiệm, ghi kết quả ra results/ (khoảng 1 phút)
```

Lần đầu chạy `thu_nghiem.py` sẽ tải bộ `harouzie/vietnews` từ HuggingFace về `data/raw/` (thư mục này không đưa lên git). App không cần tải gì, tab Thử nghiệm chỉ đọc kết quả có sẵn trong `results/`.

`mo_phong.ipynb` chạy lại phần mô phỏng bằng notebook, dùng khi demo app bị lỗi.

## Cấu trúc thư mục

```
app.py            app Streamlit, 3 tab
src/
  preprocess.py   tách câu, tách từ (underthesea), bỏ stopword, lọc câu trùng và câu ngắn
  vectorize.py    ma trận V (từ x câu) theo TF-IDF
  nmf.py          NMF, chạy nhiều seed, chuẩn hóa W và H
  summarizer.py   chọn câu theo NMF: cách topic và cách grs
  baselines.py    Lead-3, Random-3, LSA, TextRank
  evaluate.py     ROUGE-1, ROUGE-2, ROUGE-L tính theo âm tiết
data/             stopword, dữ liệu mô phỏng, 3 bài tin mẫu kèm tóm tắt
results/          bảng kết quả, hình vẽ, nhận xét của phần thử nghiệm
docs/             ghi chú kết quả chạy thử để viết báo cáo
kiem_tra.py, thu_nghiem.py, mo_phong.ipynb
```

## Đưa app lên Streamlit Community Cloud

1. Đăng nhập https://share.streamlit.io bằng tài khoản GitHub.
2. Chọn "Create app", chọn repo này, nhánh `main`, file chính `app.py`.
3. Trong "Advanced settings" chọn Python 3.11 trở lên (nhóm đã thử 3.11 và 3.13), rồi bấm Deploy.

Lần đầu cài thư viện mất vài phút.

## Dữ liệu và tài liệu tham khảo

- Bộ dữ liệu VietNews (VNDS): Nguyen, V.-H. et al. (2019). VNDS: A Vietnamese dataset for summarization. NICS. Bản dùng trong bài: `harouzie/vietnews` trên HuggingFace.
- Danh sách stopword: https://github.com/stopwords/vietnamese-stopwords
