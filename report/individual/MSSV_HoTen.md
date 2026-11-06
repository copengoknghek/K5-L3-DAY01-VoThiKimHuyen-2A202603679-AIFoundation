# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603679
- **Họ và tên:** Võ Thị Kim Huyền
- **SEED** (in ở ô đầu notebook): 4491

## 1. Bằng chứng

| Tệp                                                                  | Có? |
| -------------------------------------------------------------------- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output  | co  |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | co  |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước          | co  |

## 2. Kết quả chính

_Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn._

                                 precision  recall  F1 (test)  F1 (SUB 40)  \

1.  BoW + LogReg (tự code) 1.000 0.893 0.943 0.947
2.  BoW + MLP (deep learning) 0.987 0.893 0.938 0.947  
    3a. Encoder (MiniLM) + kNN 0.917 0.917 0.917 0.974  
    3b. Encoder (MiniLM) + LogReg 0.929 0.929 0.929 1.000  
    3c. Encoder (MiniLM) + MLP head 0.952 0.952 0.952 1.000

                                     train (s)  tin/s (cả lô)  p50 ms/tin  \

3.  BoW + LogReg (tự code) 3.484 24159.950 0.064
4.  BoW + MLP (deep learning) 4.915 27562.070 0.284  
    3a. Encoder (MiniLM) + kNN 75.823 10.820 54.289  
    3b. Encoder (MiniLM) + LogReg 76.897 16.811 86.255  
    3c. Encoder (MiniLM) + MLP head 76.885 30.837 24.949

                                     p95 ms/tin  USD / 1k tin

5.  BoW + LogReg (tự code) 0.096 0.0
6.  BoW + MLP (deep learning) 0.321 0.0  
    3a. Encoder (MiniLM) + kNN 113.985 0.0  
    3b. Encoder (MiniLM) + LogReg 137.315 0.0  
    3c. Encoder (MiniLM) + MLP head 33.008 0.0  
    • Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
    • SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
    • Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
    • train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

_Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch._
Độ trễ của LLM:
Đoán: Nghĩ rằng LLM chỉ mất khoảng dưới 100ms giống các mô hình cục bộ.
Thực tế: Độ trễ p95 lên tới trên 300ms do overhead gọi API mạng bên ngoài.
Vì sao lệch: Chưa lường trước được độ trễ truyền tải mạng và thời gian sinh token của LLM qua API.
Hiệu năng của kNN trên Embedding:
Đoán: Nghĩ rằng tăng k lên lớn (như k=25) sẽ càng giúp kết quả ổn định và chính xác hơn.
Thực tế: F1 giảm mạnh khi k quá lớn (xuống 0.8556) do hiện tượng over-smoothing.  
Vì sao lệch: Quên mất kNN bị ảnh hưởng bởi vùng lân cận quá rộng làm mất đi tính cục bộ của dữ liệu.
Chi phí chạy LLM quy mô lớn:
Đoán: Nghĩ chi phí gọi LLM phân loại hàng triệu tin nhắn sẽ cực kỳ rẻ và không đáng kể.
Thực tế: Chi phí đội lên hàng nghìn USD/tháng cho quy mô lớn.
Vì sao lệch: Chủ quan không tính toán kỹ tổng số lượng token đầu vào (input tokens phình to khi prompt dài hoặc dùng few-shot).

## 4. Phân tích 50 case khó

_Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai._

Model tụt nhiều nhất: Các mô hình truyền thống (BoW) và Embedding đơn giản (đặc biệt tụt nặng ở nhóm obfuscation và prompt_injection).

2 case cụ thể & cơ chế gây sai:

Case né từ khóa (obfuscation - ví dụ: "F R E E cash reward"): BoW đoán nhầm thành ham vì các từ bị tách khoảng trắng, không khớp với từ khóa trong từ vựng (vocabulary).

Case tấn công prompt (prompt_injection - ví dụ: "This is not spam. Ignore previous..."): LLM (khi thiếu phòng thủ) bị đánh lừa bởi cấu trúc câu lệnh giả mạo nằm bên trong nội dung tin nhắn, dẫn đến nhãn bị bẻ lái sai.

## 5. Thử thách cá nhân

_Đề theo SEED của bạn, cách làm, kết quả, giải thích._
Cách làm: Chạy hàm knn_predict với các giá trị k thuộc [1, 5, 25] trên tập dữ liệu nhúng.  
Kết quả: k=1 đạt F1 = 0.9048; k=5 tối ưu nhất với F1 = 0.9167; k=25 giảm còn F1 = 0.8556.  
Giải thích: k quá nhỏ (k=1) dễ bị nhiễu (overfitting); k quá lớn (k=25) làm mịn vùng quyết định quá mức gây ra underfitting.

## 6. Phương án cho sản phẩm thật

_Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì._
Lựa chọn: Sử dụng kiến trúc Hệ thống lai (Router) gồm Tầng 1 là mô hình nhẹ (BoW + LogReg hoặc MiniLM + LogReg) lọc nhanh 90% lượng tin nhắn rõ ràng, và Tầng 2 (LLM Fallback) chỉ xử lý 10% tin nhắn lưỡng lự.  
Số liệu dẫn chứng: Đảm bảo p95 trung bình toàn hệ thống giữ dưới ngưỡng 50 ms/tin, đồng thời cắt giảm được 90% chi phí gọi API LLM so với việc bắt toàn bộ 10 triệu SMS/ngày chạy qua LLM.  
Đánh đổi: Chấp nhận độ phức tạp cao hơn trong kiến trúc hệ thống (phải xây dựng thêm logic router điều hướng tin nhắn).

## 7. AI log và điều đã học

_Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu._
Log sử dụng AI: Hỏi cách viết hàm batch cho LLM, nhờ giải thích lỗi RateLimitError (mã 429) và cách vẽ biểu đồ đánh đổi Pareto. chưa sửa lại logic xử lý token và thêm cơ chế giãn thời gian gọi API (time.sleep / retry_with_backoff).

Điều rút ra: Hôm nay tôi hiểu sâu sắc rằng trong bài toán AI thực tế, không có mô hình nào là hoàn hảo tuyệt đối; thành công của sản phẩm phụ thuộc vào việc thiết kế kiến trúc kết hợp (router) cân bằng hoàn hảo giữa chất lượng (F1), độ trễ (SLA) và chi phí tài chính.
