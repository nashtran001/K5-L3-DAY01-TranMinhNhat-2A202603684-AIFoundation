# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202603684
- **Họ và tên:** Trần Minh Nhật
- **SEED:** 4776

## 1. Bằng chứng

| Tệp | Tình trạng |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) | Đã chạy, có output |
| [`lab1_report.json`](lab1_report.json) | Đủ kết quả, 15/15 kiểm tra đạt và 10 dự đoán |
| [`hand_calc.jpg`](hand_calc.jpg) | Có đủ 5 phép tính tay và các bước |

Sau khi bỏ trùng, dữ liệu có 5.169 tin, chia thành 3.619 tin train, 775 tin validation và 775 tin test. Tập SUB lấy từ test, gồm 20 spam và 20 ham.

## 2. Kết quả chính

| Phương án | F1 test | F1 SUB 40 | p95 ms/tin | USD/1.000 tin |
| --- | ---: | ---: | ---: | ---: |
| BoW + LogReg | 0,904 | 0,857 | 0,117 | 0 |
| BoW + MLP | 0,951 | 1,000 | 0,297 | 0 |
| MiniLM + kNN | 0,926 | 0,923 | 44,105 | 0 |
| MiniLM + LogReg | 0,941 | 0,974 | 12,326 | 0 |
| MiniLM + MLP head | 0,956 | 0,974 | 37,661 | 0 |
| LLM nhỏ zero-shot | Chưa đo | 1,000 | 2.318,395 | 0,027973 |
| LLM nhỏ few-shot | Chưa đo | 1,000 | 2.018,991 | 0,072373 |

Số liệu lấy từ JSON và output notebook. LLM dùng `gemini-3.5-flash-lite`, chỉ chạy trên SUB 40 nên chưa có F1 trên toàn bộ test. Vì SUB nhỏ, điểm tuyệt đối chưa đủ để kết luận LLM luôn tốt hơn. Chi phí 0 của các mô hình cục bộ là không mất phí API, chưa tính hạ tầng; chi phí LLM được tính theo cấu hình giá trong notebook.

## 3. Dự đoán và thực tế

1. **Bỏ ReLU:** Tôi đoán F1 sẽ gần LogReg, nhưng mạng không ReLU đạt 0,951, cao hơn LogReg tự code 0,904. Hai lớp Linear vẫn gộp được thành một phép biến đổi affine, nhưng cách tối ưu khác nhau nên kết quả không nhất thiết giống nhau.
2. **Tin nhắn ngụy trang:** Tôi đoán Embedding + LogReg đúng nhiều hơn BoW, nhưng cả hai đều đúng 5/6. Cả hai bỏ sót câu “Last chance! Your account rewards expire tonight, confirm details to keep them”. Nội dung này giống thông báo tài khoản hợp lệ, cho thấy embedding vẫn có thể nhầm ý đồ của tin nhắn.
3. **50 case khó:** Tôi đoán accuracy BoW khoảng 60%, embedding 75% và LLM 85%. Thực tế, BoW đạt 52–54%, embedding đạt 60–72%, còn LLM đạt 100%. Tôi đã đánh giá quá cao khả năng xử lý tin mới của các mô hình học từ bộ SMS ban đầu, đồng thời đánh giá thấp LLM trên bộ case này.

## 4. Phân tích 50 case khó

| Phương án | F1 test thường | F1 case khó | Accuracy case khó |
| --- | ---: | ---: | ---: |
| BoW + LogReg | 0,904 | 0,333 | 52% |
| BoW + MLP | 0,951 | 0,378 | 54% |
| MiniLM + kNN | 0,926 | 0,750 | 72% |
| MiniLM + LogReg | 0,941 | 0,612 | 62% |
| MiniLM + MLP head | 0,956 | 0,583 | 60% |
| LLM zero-shot | 1,000 (SUB) | 1,000 | 100% |

**BoW + MLP tụt nhiều nhất**, từ F1 0,951 xuống 0,378. Mô hình sai cả 6 case né từ khóa, chỉ đúng 1/10 spam giọng tự nhiên và 2/8 case thao túng. Hai ví dụ cụ thể:

- **Case #29:** “C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w…” là spam nhưng MLP đoán ham. Việc thay chữ bằng số khiến các từ như `w0n`, `g1ft`, `Cl4im` không khớp từ quen thuộc trong vocab.
- **Case #31:** “U have been s.e.l.e.c.t.e.d for a c.a.s.h r.e.w.a.r.d…” cũng bị đoán thành ham. Dấu chấm làm tokenizer tách từng chữ, mất các từ mang tín hiệu spam. MLP dùng đầu vào BoW nên không khắc phục được việc mất thông tin này.

MiniLM + LogReg đúng toàn bộ nhóm né từ khóa nhưng chỉ đúng 10% nhóm spam giọng tự nhiên. LLM zero-shot và prompt phòng thủ đều đúng 50/50, kể cả nhóm prompt injection; lần chạy này chưa cho thấy prompt phòng thủ giúp tăng điểm.

Test thường cùng nguồn với train, có thể còn những mẫu gần trùng nên điểm khá lạc quan. Bộ case khó giúp tìm điểm yếu nhưng không đại diện đầy đủ cho SMS thực tế. Năm case tự thêm chưa có output đánh giá lại, nên kết quả ở đây chỉ tính 50 case gốc.

## 5. Thử thách cá nhân

Với SEED 4776, tôi nhận **đề #1: đổi ngưỡng của BoW + LogReg sang 0,2 và 0,8**. Tôi giữ nguyên mô hình đã train, chỉ thay ngưỡng để gán nhãn spam.

| Ngưỡng | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| 0,2 | 0,916 | 0,956 | 0,936 |
| 0,5 | 1,000 | 0,825 | 0,904 |
| 0,8 | 1,000 | 0,588 | 0,740 |

Hạ ngưỡng giúp bắt nhiều spam hơn nhưng tăng chặn nhầm; nâng ngưỡng khiến mô hình thận trọng hơn nhưng bỏ sót nhiều spam. Với ngân hàng sợ chặn nhầm OTP, tôi chọn 0,8 theo hướng ưu tiên thận trọng. Tuy nhiên, trong test này cả 0,5 và 0,8 đều không có chặn nhầm, còn 0,5 có recall tốt hơn. Vì vậy cần kiểm tra thêm trên tin OTP thực tế trước khi chốt ngưỡng.

## 6. Phương án cho sản phẩm thật

Với 10 triệu SMS/ngày, tải trung bình khoảng **116 tin/giây** và yêu cầu p95 dưới 50 ms, tôi chọn **BoW + MLP cho bước xử lý ban đầu**. Mô hình đạt F1 test 0,951, p95 0,297 ms và throughput theo lô khoảng 37.216 tin/giây. Đánh đổi là khả năng xử lý spam mới còn yếu, thể hiện qua accuracy 54% trên case khó.

Tôi đề xuất chuyển các tin lưỡng lự sang MiniLM + LogReg, có p95 12,326 ms và throughput theo lô khoảng 1.588 tin/giây. Những trường hợp còn bất đồng có thể dùng LLM kiểm tra bổ sung ngoài luồng tức thời. Cần đo lại độ trễ toàn tuyến và tải đỉnh trước khi triển khai, vì benchmark theo lô chưa phản ánh đầy đủ tải thực tế.

Cả năm mô hình cục bộ đều đạt p95 dưới 50 ms trong lần đo, nhưng kNN chỉ xử lý khoảng 54 tin/giây theo lô, thấp hơn tải trung bình. LLM có p95 trên 2 giây nên không phù hợp để gọi trực tiếp cho mọi tin. Theo chi phí trong notebook, nếu xử lý toàn bộ 10 triệu tin/ngày trong 30 ngày, zero-shot tốn khoảng **8.392 USD/tháng**, few-shot khoảng **21.712 USD/tháng**, chưa tính hạ tầng. Few-shot không tăng F1 trên SUB nên tôi chưa chọn cách này.

## 7. AI log và điều đã học

| Lần dùng AI | Hỏi gì | Nhận gì | Đối chiếu / chỉnh sửa |
| --- | --- | --- | --- |
| Hỗ trợ notebook (theo log đã lưu) | Giải thích và hỗ trợ câu trả lời còn thiếu | Giải thích MLP, embedding, LLM và đánh đổi triển khai | Log ghi đã đối chiếu kết quả phần 3.B; số liệu còn thiếu cần kiểm chứng thêm |
| Hoàn thiện báo cáo | Tổng hợp nội dung từ các tệp trong `report` | Bảng kết quả, phân tích dự đoán, case khó và phương án triển khai | AI đối chiếu JSON với output, phân biệt SUB 40 với toàn test và làm rõ kết quả chọn ngưỡng |
| Biên tập báo cáo | Rút gọn, viết súc tích và tự nhiên hơn | Bản báo cáo ngắn gọn hơn | Giữ các số liệu chính và giới hạn của kết quả; chỉ sửa Markdown |

Tôi hiểu rõ hơn rằng điểm test cao chưa chắc đồng nghĩa với xử lý tốt tin mới, nhất là khi cách viết khác dữ liệu train. Embedding cũng không tự giải quyết mọi trường hợp khó; đầu phân loại và dữ liệu học vẫn ảnh hưởng lớn. Khi chọn mô hình, tôi cần cân nhắc cả chất lượng, chặn nhầm, độ trễ và chi phí thay vì chỉ nhìn F1.
