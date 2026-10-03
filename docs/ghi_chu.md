# Ghi chú kết quả chạy thử (để đưa vào báo cáo)

## 1. NMF trên dữ liệu mô phỏng, thử nhiều seed

Dữ liệu: `data/mo_phong.txt`, 9 câu thuộc 3 chủ đề. Câu 0, 3, 6 nói về pin; câu 1, 4, 7 nói về camera; câu 2, 5, 8 nói về giá. Sau tiền xử lý còn 30 từ, ma trận V có kích thước 30 × 9.

Thiết lập: k = 3, max_iter = 200, TOL = 1e-4. Mỗi seed chạy `nmf()` một lần, sau đó `normalize()`. Một câu được xếp vào chủ đề có giá trị H lớn nhất ở cột của câu đó. Nhóm em coi là "tách đúng" khi 3 nhóm câu thu được trùng với 3 chủ đề ở trên.

### 1.1. V là tf × idf, có chuẩn hóa L2 từng cột (cách đang dùng)

| Seed | Loss cuối | Số vòng | Tách đúng | Top 5 từ mỗi chủ đề |
|---|---|---|---|---|
| 0 | 2.1079 | 12 | Có | chụp, camera, đẹp, chế_độ, chân_dung / giá, máy, hợp_lý, phân_khúc, đối_thủ / pin, điện_thoại, thời_lượng, phút, mươi |
| 1 | 2.1091 | 14 | Có | chụp, camera, đẹp, chế_độ, xóa_phông / pin, thời_lượng, điện_thoại, mươi, phút / giá, máy, hợp_lý, đối_thủ, phân_khúc |
| 2 | 2.1080 | 19 | Có | camera, chụp, đẹp, chế_độ, xóa_phông / giá, máy, hợp_lý, phân_khúc, đối_thủ / pin, thời_lượng, điện_thoại, mươi, sạc |
| 3 | 2.1079 | 14 | Có | camera, chụp, đẹp, chân_dung, chế_độ / giá, máy, hợp_lý, mua, rẻ / pin, thời_lượng, điện_thoại, mươi, sạc |
| 4 | 2.1080 | 22 | Có | camera, chụp, đẹp, chế_độ, xóa_phông / giá, máy, hợp_lý, phân_khúc, đối_thủ / pin, thời_lượng, điện_thoại, mươi, sạc |

Cả 5 seed đều tách đúng, loss chênh nhau rất ít. `fit_nmf()` với 5 seed này chọn seed 3, là seed có loss nhỏ nhất.

Nhóm em chạy thêm seed 0 đến 99 thì thấy 6 seed bị sai:

| Seed | Loss cuối | Số vòng | Chuyện gì xảy ra |
|---|---|---|---|
| 31 | 2.1504 | 38 | 3 chủ đề vẫn đúng, nhưng câu 6 ("pin vẫn trâu và máy không bị nóng") bị xếp vào chủ đề giá, vì từ "máy" có trọng số cao trong chủ đề giá |
| 39 | 2.2341 | 52 | Một chủ đề gộp "máy, pin, giá", camera đứng riêng, chủ đề còn lại chỉ có câu 5 |
| 49 | 2.2688 | 18 | Gộp pin và giá thành một chủ đề (giá, máy, pin, hợp_lý); camera bị tách thành chụp đêm và chân dung |
| 65 | 2.2696 | 27 | Giống seed 49 |
| 91 | 2.2390 | 23 | Gộp pin và giá; camera bị tách thành chụp ngày/chân dung và chụp đêm (hơi, nhiễu, mờ, đêm) |
| 96 | 2.2388 | 17 | Giống seed 91 |

Các seed sai đều có loss lớn hơn các seed đúng (khoảng 2.15 đến 2.27, so với 2.108). Như vậy NMF dừng ở một cực tiểu địa phương kém hơn. Đây là lý do nhóm em chạy nhiều lần rồi lấy lần có loss nhỏ nhất (`n_restarts=5`).

### 1.2. So sánh: V là tf × idf, không chuẩn hóa

Trước đó nhóm em dùng V = tf × idf, không chuẩn hóa (`norm=None`), và chạy cùng 5 seed:

| Seed | Loss cuối | Tách đúng |
|---|---|---|
| 0 | 10.7554 | Không |
| 1 | 10.7455 | Có |
| 2 | 11.0324 | Không |
| 3 | 10.7288 | Có |
| 4 | 10.7517 | Không |

Không chuẩn hóa thì chỉ 2/5 seed tách đúng, các seed còn lại cũng gộp pin với giá và tách camera làm đôi. Khi chuẩn hóa L2, mỗi câu có độ dài vector bằng 1. Câu dài (ví dụ câu 1 có 7 từ) không còn lấn át câu ngắn (câu 5 chỉ có 2 từ), nên các chủ đề cân bằng hơn. Loss ở 1.2 lớn hơn nhiều so với 1.1 chỉ vì V chưa chuẩn hóa có giá trị lớn hơn, không so sánh trực tiếp được.

Ví dụ câu 0, các giá trị khác 0 của cột V trước và sau chuẩn hóa: (3.833, 2.609, 2.609) thành (0.72, 0.49, 0.49).

## 2. Các bài tin mẫu

Ba bài lấy từ split test của bộ `harouzie/vietnews`. Văn bản gốc đã tách từ sẵn nên nhóm em bỏ dấu `_` và ghép lại dấu câu (hàm `detokenize` trong `src/preprocess.py`), ngoài ra giữ nguyên.

| File | Vị trí trong split test | guid | Tiêu đề |
|---|---|---|---|
| tin_mau_1.txt | 11720 | 11735 | Báo Hàn nói HLV Park Hang - seo giúp tăng doanh thu bánh gạo cay |
| tin_mau_2.txt | 2591 | 2592 | Nhiều doanh nghiệp Nhật Bản chuyển nhà máy sang Việt Nam |
| tin_mau_3.txt | 3173 | 3174 | Xem người Nhật trình diễn công nghệ lót ống không đào đường |

Tóm tắt mẫu (abstract) nằm trong các file `tin_mau_*_tomtat.txt`. Cuối bài thường có chú thích ảnh (ví dụ "Huấn luyện viên Park Hang - seo.", "... - Ảnh: QUANG KHẢI"), và các chú thích này bị tách thành câu riêng.

Nhận xét khi chạy 6 phương pháp trên tin_mau_1 (n = 3, k = 3): câu chú thích ảnh "Huấn luyện viên Park Hang - seo." xuất hiện 2 lần (câu 0 và câu 14). TextRank chọn cả hai câu này vì hai câu giống hệt nhau nên cosine bằng 1. NMF-topic và LSA cũng chọn một trong hai câu đó. Lý do là sau khi chuẩn hóa L2, câu rất ngắn chỉ gồm từ khóa của một chủ đề sẽ có giá trị H lớn ở chủ đề đó.

## 3. Lọc câu trùng và câu quá ngắn

Từ nhận xét ở mục 2, nhóm em thêm hai bước lọc câu vào tiền xử lý. Hai bước này áp dụng như nhau cho mọi phương pháp, kể cả Lead-3 và Random-3:

- Bỏ câu trùng: hai câu được coi là trùng nếu giống nhau sau khi đưa về chữ thường và bỏ dấu câu. Chỉ giữ lần xuất hiện đầu tiên.
- Bỏ câu ngắn: câu gốc có ít hơn `min_syllables` âm tiết thì bị loại. Mặc định là 8 cho ứng dụng và thử nghiệm. Trang mô phỏng đặt 0, vì câu "Camera chụp đêm hơi mờ và nhiễu." chỉ có 7 âm tiết.

Câu bị loại vẫn được hiển thị nhưng không đưa vào ma trận V và không được chọn. Lead-3 lấy 3 câu đầu tiên trong số các câu còn lại. Ngoài ra vẫn giữ điều kiện cũ: câu còn dưới 2 từ sau khi bỏ stopword thì cũng bị loại (cột "ít từ" ở bảng dưới).

### 3.1. Thống kê trên 300 bài thử nghiệm

Cách lấy 300 bài: xáo trộn chỉ số của split test bằng `np.random.default_rng(42).permutation`, rồi lấy lần lượt các bài có 5 đến 40 câu (đếm sau `detokenize` và tách câu) cho đến khi đủ 300 bài. Ở lần chạy này, 300 bài đầu tiên sau khi xáo trộn đều thỏa điều kiện.

| | Số câu | Tỉ lệ |
|---|---|---|
| Giữ lại | 4803 | 94.55% |
| Loại vì trùng | 34 | 0.67% |
| Loại vì ngắn (< 8 âm tiết) | 214 | 4.21% |
| Loại vì còn dưới 2 từ sau khi bỏ stopword | 29 | 0.57% |
| Tổng | 5080 | 100% |

Sau khi lọc, không có bài nào còn dưới 3 câu dùng được.

Nhóm em in ngẫu nhiên 20 câu bị loại để kiểm tra. Phần lớn là chú thích ảnh hoặc tên người kèm chức vụ ("Thủ tướng Canada Justin Trudeau.", "Xe tăng M1A2 Abrams của Mỹ.", "(Ảnh minh hoạ).", "Video: Guardian."), mảnh câu do tách câu sai hoặc dữ liệu bị cắt ("ngã gục ngay bục giảng.", "Và hiển nhiên, trái ng"), và câu nói ngắn trong trích dẫn ("Vẫn chưa ngã ngũ.", "Thì không nói được."). Các câu bị loại vì trùng đều là chú thích ảnh lặp lại ở cuối bài, ví dụ "Cố vấn an ninh quốc gia Mỹ John Bolton.". Trong 20 câu này nhóm em không thấy câu nào mang nội dung chính của bài. Có vài câu như "Đêm trời lạnh." có nghĩa nhưng chỉ là câu miêu tả phụ.

### 3.2. Chạy lại tin_mau_1

Sau khi lọc, tin_mau_1 còn 13/16 câu dùng được. Câu 0 bị loại vì ngắn, câu 14 vì trùng câu 0, câu 15 ("Món bánh gạo xào cay Hàn Quốc.") vì ngắn. Không phương pháp nào còn chọn câu "Huấn luyện viên Park Hang - seo.":

| Phương pháp | Câu được chọn |
|---|---|
| Lead-3 | 1, 2, 3 |
| Random-3 | 1, 9, 10 |
| LSA | 5, 7, 11 |
| TextRank | 1, 4, 5 |
| NMF-topic | 1, 5, 13 |
| NMF-grs | 1, 2, 7 |
