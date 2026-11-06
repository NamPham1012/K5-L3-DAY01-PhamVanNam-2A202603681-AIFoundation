# \# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

# 

# \- \*\*MSSV:\*\* 2A202603681

# \- \*\*Họ và tên:\*\* Phạm Văn Nam

# \- \*\*SEED:\*\* 2404

# 

# \## 1. Bằng chứng

# 

# Các dấu chọn dưới đây được giữ theo báo cáo gốc; chưa đối chiếu được các tệp đính kèm trong phiên chỉnh sửa này.

# 

# | Tệp | Có? |

# | --- | --- |

# | \[`day01\_lab.ipynb`](day01\_lab.ipynb) — notebook đã chạy, còn output | ☑ |

# | \[`lab1\_report.json`](lab1\_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |

# | \[`hand\_calc.jpg`](hand\_calc.jpg) — ảnh tính tay có các bước | ☑ |

# 

# \## 2. Kết quả chính

# 

# | Cách | F1 (test) | F1 (SUB 40) | p95 ms/tin | USD / 1k tin |

# | --- | --- | --- | --- | --- |

# | 1. BoW + LogReg (tự code) | 0.893 | 0.919 | 0.089 | 0.0 |

# | 2. BoW + MLP | 0.934 | 0.947 | 0.250 | 0.0 |

# | 3a. MiniLM + kNN | 0.878 | 0.947 | 67.07 | 0.0 |

# | 3b. MiniLM + LogReg | 0.943 | 0.947 | 36.43 | 0.0 |

# | 3c. MiniLM + MLP head | 0.948 | 0.923 | 41.18 | 0.0 |

# | 4a. LLM nhỏ zero-shot (groq/gpt-oss-20b) | — | 0.864 | 619.42 | 0.0\* |

# | 4c. LLM nhỏ few-shot (groq/gpt-oss-20b) | — | 0.857 | 963.58 | 0.0\* |

# | 4b. LLM lớn zero-shot (groq/gpt-oss-120b) | — | 0.976 | 387.29 | 0.0\* |

# 

# \\\* Bảng `PRICE\_PER\_1M` chưa có giá model Groq nên chi phí hiện 0, không phải miễn phí thật. Tôi dùng Groq vì gói miễn phí Gemini chỉ cho 20 lần gọi/ngày với model lớn (4A cần 40 lần).

# 

# \## 3. Dự đoán và thực tế

# 

# 1\. \*\*`dl\_relu`:\*\* Tôi đoán bỏ ReLU thì mạng chỉ còn tuyến tính nên F1 tụt về mức LogReg (khoảng 0.89). Thực tế vẫn 0.934, bằng MLP có ReLU. Lệch vì phần hơn của MLP so với LogReg ở bài này đến từ cách tối ưu (Adam, mini-batch, hơn 1700 lần cập nhật) chứ không phải từ lớp ẩn phi tuyến; với bag-of-words, mô hình tuyến tính đã đủ.

# 2\. \*\*`emb\_tricky`:\*\* Tôi đoán Encoder + LogReg đúng nhiều hơn BoW. Thực tế hoà 5/6. BoW không bị các tin ham có "free/won/prize" lừa vì từ trò chuyện như "my", "me", "lol" kéo mạnh về ham. Cả hai cùng sai tin phishing "Last chance! Your account rewards expire tonight…" vì dữ liệu train gần như không có kiểu spam này.

# 3\. \*\*`llm\_models`:\*\* Tôi đoán model lớn chỉ hơn model nhỏ khoảng 0.02–0.05 F1. Thực tế hơn 0.11 (0.976 so với 0.864): model nhỏ quá nhạy, chặn nhầm 4 tin ham bình thường. Bất ngờ thêm là model lớn lại nhanh hơn (p50 261 ms so với 275 ms) trong lần đo này; khác biệt có thể chịu ảnh hưởng của hạ tầng và điều kiện đo, chưa đủ dữ liệu để xác định nguyên nhân.

# 

# \## 4. Phân tích 50 case khó

# 

# BoW tụt nhiều nhất: BoW + MLP từ F1 0.947 (SUB) xuống 0.389, accuracy chỉ 56%. Nhóm "Spam giọng tự nhiên" và "Né từ khoá" là nặng nhất (BoW + MLP đúng 0%). Ngược lại LLM nhỏ tăng lên 0.903, và 0.935 khi có prompt phòng thủ.

# 

# \- \*\*Case #13 (spam giọng tự nhiên):\*\* "Hope you're well! Quick reminder that your photos from the event are ready, view \& order prints here: snap-prints.co/r/88". BoW chỉ nhìn từng từ rời rạc; tin này không có từ khoá spam nào (free, call, txt, claim) mà toàn từ lịch sự đời thường, nên tổng trọng số nghiêng về ham và bị lọt.

# \- \*\*Case #2 (tưởng spam nhưng không phải):\*\* "FREE pizza in the office kitchen right now, come quick before it's gone". LLM + prompt phòng thủ đoán spam. LLM đọc như người nhưng quá nhạy với "FREE" và giọng thúc giục "come quick", nên coi một tin nội bộ bình thường là quảng cáo.

# 

# Bài học: test cùng phân phối với train quá lạc quan; điểm cao ở đó chưa chứng minh model hiểu spam.

# 

# \## 5. Thử thách cá nhân

# 

# \### Đề 5 — So sánh BoW và Embedding + LogReg

# 

# Tôi thử bốn tin nhắn: hai tin được gán nhãn spam và hai tin ham chứa từ `free` hoặc `win`. Kết quả thực tế trong notebook:

# 

# | Tin nhắn | Nhãn thật | BoW + LogReg | Embedding + LogReg |

# | --- | --- | --- | --- |

# | Y0ur r.e.w.a.r.d is waiting. Send your details to collect it. | Spam | Ham ✗ | Ham ✗ |

# | Hi! Earn extra income from home. Reply to learn how to join today. | Spam | Ham ✗ | Spam ✓ |

# | Are you free after class? Let's study together at the library. | Ham | Ham ✓ | Ham ✓ |

# | I hope we win the football match on Saturday. See you there! | Ham | Ham ✓ | Ham ✓ |

# | \*\*Số tin đúng\*\* | | \*\*2/4\*\* | \*\*3/4\*\* |

# 

# 1\. \*\*Tin 1:\*\* Cả hai đều dự đoán sai. Cách viết biến dạng `Y0ur` và `r.e.w.a.r.d` có thể khiến dấu hiệu spam khó nhận ra, nhưng kết quả này không chứng minh đó là nguyên nhân.

# 2\. \*\*Tin 2:\*\* BoW dự đoán sai, còn Embedding + LogReg dự đoán đúng. Tin là lời mời kiếm thu nhập, tránh những từ khóa rõ ràng như `free`, `win` và `prize`. Embedding có thể giúp biểu diễn nội dung của lời mời, nhưng chưa xác định được chính xác đặc trưng nào quyết định dự đoán.

# 3\. \*\*Tin 3:\*\* Cả hai dự đoán đúng là ham. `free` mang nghĩa “rảnh” trong lời rủ bạn đi học.

# 4\. \*\*Tin 4:\*\* Cả hai dự đoán đúng là ham. `win` nói về thắng một trận bóng, không phải trúng thưởng.

# 

# Trên bốn tin này, Embedding + LogReg đúng 3/4 và BoW đúng 2/4; cả hai đều bỏ sót tin spam thứ nhất. Mẫu quá nhỏ để kết luận một mô hình luôn tốt hơn. AI đã đề xuất các tin nhắn và code; tôi đưa vào Colab, chạy, xử lý lỗi theo hướng dẫn và xem kết quả.

# 

# \## 6. Phương án cho sản phẩm thật

# 

# 10 triệu SMS/ngày tương đương khoảng 116 tin/giây, yêu cầu p95 < 50 ms. Các cách đạt SLA: BoW + LogReg (0.09 ms), BoW + MLP (0.25 ms), MiniLM + LogReg (36 ms), MiniLM + MLP head (41 ms). LLM bị loại vì p95 tới 387–964 ms; nếu dùng LLM cho mọi tin thì theo phép tính tay (10.5 USD cho 1 triệu tin) mất khoảng 105 USD/ngày, khoảng 3.150 USD/tháng.

# 

# Tôi chọn \*\*router 2 tầng\*\*: mọi tin qua BoW + MLP (F1 0.934, hơn 50.000 tin/giây), tin chắc chắn (p < 0.3 hoặc > 0.7) trả kết quả ngay, chỉ tin lưỡng lự (0.3–0.7) mới gửi LLM lớn có prompt phòng thủ để xử lý ngoại tuyến, cộng bước chuẩn hoá đầu vào chống né từ khoá. Đánh đổi: các tin lưỡng lự có độ trễ cao hơn, và hệ thống phức tạp hơn. Nếu chỉ khoảng 5% tin vào LLM và đơn giá không đổi, chi phí ước tính khoảng 157,5 USD/tháng. Đây là giả định, chưa phải chi phí đo thực tế; các tin chuyển sang LLM có thể không đáp ứng p95 < 50 ms nếu phải chờ kết quả.

# 

# \## 7. AI log và điều đã học

# 

# | Phần | Hỏi AI gì | AI trả lời / viết gì | Tôi đã làm gì |

# | --- | --- | --- | --- |

# | Phần 0 | Cách tạo repo, mở Colab; giải thích data leakage | Hướng dẫn thao tác, giải thích, soạn câu trả lời mẫu | Đọc hiểu, viết lại câu trả lời |

# | TODO 1.1 `vectorize` | Nhờ hướng dẫn, sau đó nhờ bản hoàn chỉnh | Giải thích ý tưởng, đưa khung có chỗ trống, chỉ lỗi (thụt lề, thiếu `in`, `text`/`texts`), rồi đưa code hoàn chỉnh | Tự gõ, sửa lỗi nhiều lần, chạy test ra  |

# | TODO 1.2 | Nhờ viết code | Đưa khung 2 chỗ trống | Tự điền, sửa lỗi thiếu dấu trừ trong `np.exp(-z)` |

# | TODO 1.3, 1.4 | Nhờ giải thích gradient descent, precision/recall/F1, rồi nhờ viết hoàn chỉnh | Giải thích bằng ví dụ và viết code `train\_logreg`, `metrics` | Đọc hiểu từng dòng, chạy test ra  |

# | TODO 2.1, 2.2 | Nhờ viết MLP và vòng lặp huấn luyện | Viết code, giải thích từng dòng | Đọc hiểu, chạy test ra  |

# | TODO 3.1–3.3 | Nhờ viết cosine, top\_k\_similar, knn\_predict | Viết code, giải thích chuẩn hoá, argsort, bỏ phiếu | Đọc hiểu, chạy test ra  |

# | TODO 4.1–4.4 | Nhờ viết call\_openai, prompt, parse\_label, llm\_classify\_batch | Viết code | Chạy test giả lập ra  |

# | Lỗi API Phần 4 | Hỏi lỗi 429, 400, 503 của Gemini | Gợi ý thêm `time.sleep`, tăng retry, đổi reasoning\_effort sang "low", rồi chuyển sang Groq | Tự tạo key Groq, lưu vào Colab Secrets, chạy lại 4A/4B |

# | Câu hỏi, Bắt lỗi, Checkpoint, Phản ánh | Nhờ phân tích kết quả và soạn câu trả lời | Phân tích số liệu, soạn câu trả lời mẫu | Đối chiếu với kết quả của mình, chỉnh lại |

# | Tính tay | Nhờ trình bày lời giải | Viết các bước tính | Tự chép tay, tự kiểm tra lại |

# | Thử thách cá nhân — đề 5 | Nhờ AI phân tích yêu cầu, viết code so sánh và hỗ trợ giải thích | AI đề xuất bốn tin nhắn, viết code, hướng dẫn sửa lỗi và giải thích dựa trên kết quả thực tế | Đưa code vào Colab, chạy các ô cần thiết, kiểm tra kết quả và đưa phần giải thích vào báo cáo |

# 

# \*\*Điều đã học:\*\* Accuracy cao có thể vô nghĩa khi dữ liệu lệch; một mô hình luôn đoán ham vẫn được 87%. Mô hình phức tạp hơn chưa chắc tốt hơn: bỏ ReLU không làm giảm F1, và trên 50 case khó thì xếp hạng giữa BoW và LLM đảo ngược hoàn toàn. Chọn mô hình cho sản phẩm thật phải cân cả chất lượng, độ trễ và chi phí, không chỉ nhìn một con số F1.

\-

