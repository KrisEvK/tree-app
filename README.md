# Phân loại cây xanh phường Bến Nghé sử dụng Deep Learning

Ứng dụng Streamlit dùng Deep Learning để nhận diện 19 loài cây xanh đường phố tại phường Bến Nghé, TP.HCM từ ảnh lá cây do người dùng tải lên hoặc chụp trực tiếp.

## 1. Yêu cầu môi trường

- Python 3.10 trở lên
- pip
- RAM tối thiểu khoảng 8 GB, khuyến nghị 16 GB khi chạy đầy đủ các mô hình
- Có thể chạy bằng CPU; nếu máy có GPU/CUDA thì PyTorch có thể tận dụng tùy môi trường cài đặt

Các thư viện Python được khai báo trong `requirements.txt`:

```txt
streamlit
torch
torchvision
timm
Pillow
numpy
```

## 2. Cấu trúc thư mục cần có

Để ứng dụng chạy đúng, các file sau cần nằm trong cùng một thư mục với `app.py`:

```text
tree_app/
├── app.py
├── fusion_model.py
├── ensemble_model.py
├── info19.json
├── requirements.txt
├── Dockerfile
├── README.md
├── ConvNeXtV2Tiny_S1_CrossEntropy.pth
├── FastViT_S1_CrossEntropy.pth
├── best_model.pth
├── DINOv2_ViTS14_LPFT_Final_ema.pth
└── fusion_best.pth
```

Vai trò của các file chính:

| File | Mục đích |
| --- | --- |
| `app.py` | File chính chạy giao diện Streamlit |
| `fusion_model.py` | Hàm load và dự đoán cho mô hình Fusion |
| `ensemble_model.py` | Hàm load và dự đoán cho mô hình Cascaded Ensemble |
| `info19.json` | Dữ liệu mô tả 19 loài cây |
| `requirements.txt` | Danh sách thư viện cần cài |
| `ConvNeXtV2Tiny_S1_CrossEntropy.pth` | Weight cho ConvNeXtV2-Tiny |
| `FastViT_S1_CrossEntropy.pth` | Weight cho FastViT |
| `best_model.pth` | Weight cho MobileNetV4 |
| `DINOv2_ViTS14_LPFT_Final_ema.pth` | Weight cho DINOv2 |
| `fusion_best.pth` | Weight cho Fusion head |

Lưu ý: Không đổi tên các file `.pth` nếu chưa sửa lại đường dẫn tương ứng trong `app.py`.

## 3. Cách chạy ứng dụng trên máy cá nhân

Mở Terminal/PowerShell tại thư mục `tree_app`, sau đó chạy:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo.

Trên Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Trên macOS/Linux:

```bash
source .venv/bin/activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

Chạy ứng dụng:

```bash
streamlit run app.py
```

Sau khi chạy thành công, Streamlit sẽ hiển thị địa chỉ truy cập, thường là:

```text
http://localhost:8501
```

Mở địa chỉ này trên trình duyệt để sử dụng ứng dụng.

## 4. Cách chạy bằng Docker

Nếu máy đã cài Docker, có thể build và chạy container:

```bash
docker build -t tree-app .
docker run --rm -p 7860:7860 tree-app
```

Sau đó mở:

```text
http://localhost:7860
```

## 5. Cách sử dụng

1. Mở ứng dụng trên trình duyệt.
2. Chọn phương pháp nhận diện trong giao diện.
3. Tải ảnh cây/lá lên hoặc chụp ảnh trực tiếp.
4. Xem kết quả dự đoán, độ tin cậy và thông tin mô tả loài cây.

Ứng dụng hiện hỗ trợ các hướng nhận diện:

| Phương pháp | Mô hình sử dụng |
| --- | --- |
| ConvNeXtV2-Tiny | `ConvNeXtV2Tiny_S1_CrossEntropy.pth` |
| Fusion | ConvNeXtV2 + FastViT + `fusion_best.pth` |
| Cascaded Ensemble | ConvNeXtV2 + FastViT + MobileNetV4 + DINOv2 |

## 6. Lỗi thường gặp

### Không tìm thấy file `.pth`

Kiểm tra lại các file weight đã nằm cùng cấp với `app.py` và đúng tên file như trong phần cấu trúc thư mục.

### Không tìm thấy `info19.json`

Đặt lại file `info19.json` vào thư mục gốc của ứng dụng, cùng cấp với `app.py`.

### Cài `torch` quá lâu hoặc lỗi

PyTorch có thể cần bản cài phù hợp với hệ điều hành và CUDA. Nếu cài qua `requirements.txt` lỗi, hãy cài PyTorch theo hướng dẫn chính thức tại trang chủ PyTorch, sau đó chạy lại:

```bash
pip install -r requirements.txt
```

### Ứng dụng chạy chậm

Các mô hình Deep Learning có dung lượng lớn. Khi chạy bằng CPU, lần dự đoán đầu tiên có thể mất nhiều thời gian vì ứng dụng cần load model vào bộ nhớ.

