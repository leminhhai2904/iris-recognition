# Nhận diện mống mắt (Iris Recognition) bằng Deep Learning

Dự án đồ án môn học nhận dạng mẫu về chủ đề **Nhận dạng mống mắt (Iris Recognition)** dựa trên hướng tiếp cận học sâu (Deep Learning) sử dụng kiến trúc ResNet-50 và cơ chế chuyển vị học (Transfer Learning) từ bài báo nghiên cứu **DeepIris**.

---

## Liên kết tải trọng số mô hình (Checkpoints)
Nếu bạn không muốn tự huấn luyện lại mô hình từ đầu, bạn có thể tải các tệp trọng số đã được huấn luyện sẵn tại đây và đặt vào thư mục `checkpoints/`:
[Google Drive Checkpoints Link](https://drive.google.com/drive/folders/11D7otQcxl9pQbbhH4NukXlX-QTJLxtra?usp=sharing)

---

## Cài đặt môi trường

1. **Khởi tạo môi trường ảo (Python Virtual Environment):**
   ```bash
   python -m venv venv
   ```

2. **Kích hoạt môi trường ảo:**
   * **Trên Windows (Command Prompt / PowerShell):**
     ```powershell
     .\venv\Scripts\activate
     ```
   * **Trên macOS / Linux:**
     ```bash
     source venv/bin/activate
     ```

3. **Cài đặt các thư viện cần thiết:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Chuẩn bị dữ liệu (Dataset)

Dự án sử dụng bộ dữ liệu mống mắt **IIT Delhi (IITD) Iris Database**.
1. Tải bộ dữ liệu tại: [Kaggle IITD Iris Dataset](https://www.kaggle.com/datasets/cminhhuymai/iitd-iris) hoặc trang chủ [IIT Delhi](https://www4.comp.polyu.edu.hk/~csajaykr/IITD/Database_Iris.htm).
2. Giải nén và đặt thư mục dữ liệu vào thư mục gốc của dự án với tên thư mục là `iitd-iris`. Cấu trúc thư mục chuẩn:
   ```text
    iris-recognition/
    ├── Source/
    │   ├── dataloader.py
    │   ├── models.py
    │   ├── training.py
    │   └── webcam_demo.py
    ├── iitd-iris/
    │   ├── 001/
    │   │   ├── 01_L.bmp
    │   │   └── 02_R.bmp
    │   ├── 002/
    │   └── ...
    ├── README.md
    └── requirements.txt
   ```

---

## Hướng dẫn chạy Huấn luyện (Training)

Chạy lệnh dưới đây để bắt đầu huấn luyện mạng nơ-ron:
```bash
python Source/training.py
```
* **Cách thức hoạt động:** Chương trình sẽ chia tập dữ liệu thành 5 ảnh để train, 1 ảnh để validate và 4 ảnh để test ngẫu nhiên cho mỗi đối tượng.
* **Đầu ra:** Kết thúc quá trình huấn luyện, tệp trọng số tốt nhất sẽ được lưu tại `checkpoints/best_model.pth` cùng file nhật ký `checkpoints/metrics.csv`.

---

## Hướng dẫn chạy Live Webcam Demo

Tệp demo webcam thời gian thực được lưu tại `Source/webcam_demo.py`. Chạy lệnh sau để khởi động demo:
```bash
python Source/webcam_demo.py
```

> [!NOTE]
> Trong lần đầu tiên chạy, PyTorch sẽ tự động tải các trọng số ResNet-50 ImageNet mặc định từ máy chủ (`resnet50-0676ba61.pth` khoảng 100MB). Vui lòng kết nối mạng và chờ đợi cho quá trình tải hoàn tất (chỉ diễn ra một lần duy nhất).
> Nếu chưa huấn luyện mô hình, chương trình sẽ chạy ở chế độ mô phỏng sinh kết quả ngẫu nhiên để test giao diện (`[Demo Mode]`).

### Các chế độ chạy trong Webcam Demo:

#### 1. Chế độ Thủ công (Manual ROI Mode) - Mặc định
* **Mô tả:** Hiển thị một ô vuông màu xanh lá cây cố định ở tâm màn hình webcam.
* **Cách dùng:** Người dùng căn chỉnh mắt của mình (hoặc hình ảnh mống mắt hiển thị trên điện thoại) nằm trọn bên trong ô vuông này. Hệ thống sẽ tự động cắt ảnh, xử lý và dự đoán danh tính.
* **Ứng dụng quan trọng:** Bắt buộc sử dụng chế độ này khi bạn muốn thực hiện **demo bằng ảnh mống mắt trên điện thoại** (vì ảnh mống mắt trong bộ dữ liệu là ảnh cắt siêu cận cảnh, bộ lọc tự động sẽ không phát hiện được cấu trúc mắt đầy đủ).

#### 2. Chế độ Tự động (Auto-Detection Mode)
* **Mô tả:** Sử dụng thuật toán Haar Cascade của OpenCV để tự động nhận dạng vị trí mắt của bạn trên khuôn mặt và bám theo (tracking) trong thời gian thực.
* **Cách dùng:** Khi webcam đang chạy, nhấn phím **`m`** trên bàn phím để kích hoạt chế độ tự động. Bạn không cần phải để mắt vào ô vuông cố định nữa; một khung vuông sẽ tự động bám theo mắt bạn và hiển thị kết quả nhận diện.
* **Lưu ý:** Chỉ hoạt động hiệu quả khi đưa **mắt thật của người dùng trước webcam** (có đầy đủ lông mày, mí mắt, cấu trúc hốc mặt).

### Các phím tương tác nhanh (khi đang mở cửa sổ webcam):
* **Phím `m`**: Chuyển đổi qua lại giữa chế độ **Tự động (Auto-Detection)** và **Thủ công (Manual ROI)**.
* **Phím `q`**: Thoát chương trình và đóng camera an toàn.
