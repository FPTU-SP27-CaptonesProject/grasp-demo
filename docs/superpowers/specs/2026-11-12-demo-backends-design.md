# Demo: backend đổi được, dữ liệu đúng với HF, preflight

- **Ngày:** 2026-11-12
- **Bổ sung cho:** `2026-11-11-grasp-demo-design.md`. Những gì không nhắc ở đây vẫn giữ như spec gốc.
- **Nhánh:** `feat/demo-backends` (tạo từ `main` của bạn cùng nhóm). Mở PR để review, không push thẳng lên `main`.
- **Phụ thuộc:** `grasp-grpo-trainer` nhánh `feat/grpo-training-protocol`, commit `9ff7150` trở lên
  (`run_example(..., emit=)`; `baselines` đã có `emit` từ trước).

## 0. Hiện trạng và lý do

Bản `main` (commit 1bae3b6) đã có khung server, runner, lưu và phát lại, nhưng chưa chạy được:

1. Trainer chưa có `emit` trong `run_example`. Đã sửa ở trainer (9ff7150).
2. `fetch_data.py` tìm `triples.sqlite` riêng trong từng corpus, nhưng HF chỉ có cache dùng chung.
3. Không tải được `eval_ids.json` hay `dev.jsonl`, nên danh sách câu eval trống.
4. `fetch_checkpoint.py` không tải được một thư mục con của repo nhiều checkpoint.
5. Nhãn checkpoint đọc `step` và `f1` từ `grpo_config.json`. `step` ở đó tính trong từng lần chạy, còn khóa `f1` không tồn tại.
6. Health check chỉ tạo client, không kiểm tra thành phần có chạy thật không.
7. Bedrock chậm: kiểm tra ngày 2026-11-12, 24/24 lời gọi thành công nhưng mất 4–24 giây cho 125–300 token. Ngày 2026-11-12 trước đó còn có lúc treo hẳn.
8. Giao diện chỉ in JSON thô, thiếu các yếu tố spec mục 5 yêu cầu.
9. Hai tab cùng mở stream của một lần chạy thì tranh nhau một hàng đợi.

Thêm vào đó:
- Train đã dừng hẳn. Ứng viên GRPO chỉ còn `s100`, `s250`, `s314`, và sẽ không có `best/` của run full.
- Chưa biết máy local (RTX 5060 Ti 16 GB) có chạy nổi không. Vì vậy demo phải chạy được cả trên máy local lẫn máy thuê, và có bước đo trước để chọn.

## 1. Hai chế độ triển khai, cùng một code

| | `models.local.yaml` | `models.vast.yaml` |
|---|---|---|
| Reader 7B | llama-server GGUF Q6_K, `http://localhost:8080/v1` | vLLM FP8 như lúc eval, `http://localhost:8000/v1` |
| Proxy 3B | `backend: hf`: transformers bf16, base + 2 LoRA, có lock | `backend: vllm`: policy server vLLM, `http://localhost:8001/v1` |
| E5 | CPU | GPU (`cuda:1`), như `RETRIEVER_GPU` |
| Server demo | máy local | máy Vast. Ở máy local mở `ssh -L 7860:localhost:7860`, rồi vào `http://localhost:7860` |

**Backend `vllm`:**
- Lúc khởi động, server nạp từng adapter lên policy server bằng `POST /load_lora_adapter`, với tên `demo-sft` và `demo-grpo`. Cách này giống `eval_adapter_vast.sh`, dùng `READER_API_KEY` làm Bearer.
- Server gỡ các adapter này khi tắt.
- Câu được sinh bằng `grasp_grpo.policy_server.VLLMPolicy(url, key, load_tokenizer(adapter), model=<tên>, timeout=60)` (mặc định của nó là 900 giây, quá dài cho demo), bọc qua `role_generate_fn(..., temperature=0, top_p=1)`. Hai cột GRASP không cần lock, vì vLLM tự gom batch.
- Nếu tên adapter đã tồn tại (server cũ chưa gỡ), gỡ rồi nạp lại.

**Backend `hf`:** giữ như `ProxyPolicy` hiện tại (base và LoRA SFT là `default`, GRPO nạp bằng `load_adapter`, có lock).

Cả hai backend trả về cùng kiểu `generate_fn(role, prompt, samples)`. Runner không cần biết đang dùng backend nào.

## 2. Cấu hình `models.*.yaml`

```yaml
reader:   {url: http://localhost:8080/v1, model: qwen2.5-7b-instruct, tokenizer: Qwen/Qwen2.5-7B-Instruct, api_key_env: READER_API_KEY}
strong:   {url: https://bedrock-mantle.ap-southeast-2.api.aws/v1, model: qwen.qwen3-235b-a22b-2507, timeout: 60, retries: 1}
proxy:
  backend: hf                     # hf | vllm
  url: http://localhost:8001/v1   # chỉ dùng với vllm
  adapters:
    sft:  {path: checkpoints/grasp-qwen2.5-3b-sft, label: "SFT"}
    grpo: {path: checkpoints/grpo-s100, label: "GRPO s100"}
retrieval: {corpus_config: data/corpus.json, top_k: 10, max_candidate_triples: 20, encoder: intfloat/e5-large-v2, encoder_device: cpu}
ircot:    {max_steps: 5, docs_per_step: 3}
grasp:    {max_steps: 5}
eval:     {qa: data/grasp-gold-evidence/dev.jsonl, ids: data/test_ids.json, showcase: data/showcase.json}
runs_dir: runs
```

- `adapters.<tên>` vẫn nhận dạng chuỗi (chỉ đường dẫn) như bản cũ. Khi đó nhãn là tên thư mục.
- Nhãn hiển thị lấy từ `label`, không đọc `grpo_config.json` nữa.
- `strong`:
  - `timeout: 60`, `retries: 1`, vì Bedrock đang chậm (mục 0.7).
  - Cột 235B chờ tối đa khoảng 2 phút rồi báo lỗi.
  - Không có key thì chỉ tắt cột này.
- `reader.api_key_env`:
  - Tên biến môi trường chứa key. Thiếu thì dùng `EMPTY`.
  - Chế độ vast dùng `READER_API_KEY`, chính là key của vLLM trên máy eval.
- **Chọn checkpoint GRPO:** lấy checkpoint có F1 test999 cao nhất trong `s100`, `s250`, `s314`, xem ở `reports/test_ids/table.md` trong repo HF `grasp-grpo-logs`. Trong lúc chờ kết quả thì để `s100`. Đổi checkpoint chỉ cần sửa `path` và `label`.
- **Không trích triple lúc chạy:** giống eval có `EXTRACT_TRIPLES=0`. Tài liệu chưa có trong cache sẽ không có triple, nên KI có ít ứng viên hơn. Sự kiện `ki` có `candidates: []` khi không có ứng viên nào.

## 3. Dữ liệu và checkpoint

**`scripts/fetch_data.py`:**
- Tải từ `doancuavahk/grasp-retrieval`, dùng `allow_patterns`:
  - `2wiki_ircot/**`, `hotpotqa_abstracts_2017/**`, `wiki18/**`;
  - `cache/triples-train-20261112.sqlite`.
- Tải `dev.jsonl` từ `doancuavahk/grasp-gold-evidence` (giống `fetch_data_vast.sh` của trainer). Tên repo là tham số.
- Tải `test_ids.json`. Nếu có `--trainer-dir` thì chép từ `<trainer>/config/test_ids.json`.
- Tạo `data/corpus.json`. Mỗi corpus có `{index_dir, triple_cache}`, trong đó `index_dir` là **thư mục gốc của corpus** (chứa `passages.sqlite` và `e5-large-v2/index.faiss`), còn `triple_cache` cùng trỏ vào một file. Định dạng giống `config/corpora.vast.json` của trainer.
- Kiểm tra trước khi xong:
  - mỗi corpus có `passages.sqlite` và `e5-large-v2/index.faiss` (đúng các đường dẫn mà `fetch_data_vast.sh` kiểm tra);
  - file triple mở được bằng sqlite và có bảng `triples`;
  - `dev.jsonl` chứa mọi id trong `test_ids.json`.

  Sai chỗ nào thì báo đúng đường dẫn đó.
- Có cờ `--skip-download`: chỉ kiểm tra và tạo `corpus.json` từ dữ liệu đã có. Trên máy Vast, dữ liệu nằm sẵn ở `/workspace/data`.

**`scripts/fetch_checkpoint.py <repo> <tên> [--subfolder s100]`:**
- Có `--subfolder` thì tải `s100/*`, rồi đưa nội dung lên `checkpoints/<tên>/`, sao cho `adapter_config.json` nằm ở gốc.
- Kiểm tra có `adapter_config.json`, như bản hiện tại.

**`data/showcase.json`** gồm các câu tiêu biểu để chạy trước và trình bày.
- Script `scripts/build_showcase.py` tạo file này từ report test999 (đã có `predictions`):
  - chọn câu GRPO đạt F1 = 1 trong khi `single` và `ircot-k1` < 0,5;
  - mỗi dataset tối đa 3 câu, ưu tiên câu có ≥ 2 tài liệu gold.
- Định dạng: `{"ids": [...], "source": "<tên report>"}`.
- Script này chỉ chạy được sau khi eval xong.

## 4. Server

- **Health thật**, kiểm tra một lần lúc khởi động và trả về ở `/api/health`:
  - reader: gửi prompt "Say OK." với `max_tokens: 4`;
  - Bedrock: gọi `GET /models`;
  - proxy: adapter có `adapter_config.json`. Với backend `vllm` thì còn phải nạp được LoRA;
  - retriever: mở được index và sqlite.

  Thành phần nào hỏng thì tắt các cột phụ thuộc nó, kèm lý do. Server vẫn chạy.
- **Mỗi kết nối stream có hàng đợi riêng:**
  - `DemoRun` giữ danh sách sự kiện cùng một điều kiện chờ (`threading.Condition`);
  - mỗi kết nối tự giữ vị trí đã đọc;
  - nhiều tab cùng xem một lần chạy đều nhận đủ sự kiện, không trùng.
- **API thêm:**
  - ~~`GET /api/doc/{corpus}/{doc_id}`~~: bỏ khi lập kế hoạch. `passages.sqlite` không có index trên `doc_id`,
    nên tra cứu trong wiki18 phải quét cả bảng. Sự kiện `retrieve` đã chứa toàn văn từng tài liệu, nên giao diện
    mở toàn văn từ chính sự kiện đó. Danh sách gold trong phần câu hỏi hiện doc id và đánh dấu "tìm thấy bởi …"
    khi có cột truy xuất được tài liệu đó.
  - `GET /api/eval?showcase=1`: chỉ trả các câu tiêu biểu;
  - `POST /api/prerun`: chạy lần lượt các câu tiêu biểu và lưu replay. Mỗi lúc chỉ được một lượt; đang chạy thì trả 409.
- `runners.py` giữ nguyên cách gọi, nhưng lấy `generate_fn` từ proxy, dù backend là `hf` hay `vllm`.

## 5. Giao diện

Sửa theo spec gốc mục 5:

- **Mỗi bước hiển thị dạng đọc được** (không còn JSON thô):
  - **retrieve**: query, rồi danh sách tài liệu (tiêu đề và 200 ký tự đầu). Tài liệu gold có ★ và tô màu. Bấm vào để mở hộp thoại toàn văn.
  - **thought**: nguyên văn.
  - **RD**: "Truy xuất: có" hoặc "Truy xuất: không".
  - **QF**: query cùng chuỗi suy luận (chain).
  - **KI**: triple được chọn `<h; r; t>`, kèm "thêm tài liệu X" hoặc "tài liệu đã có".
  - Action có `valid: false` mang nhãn "⚠ sai format". Nút "raw" để xem output gốc của proxy.
- **Header cột:**
  - nhãn checkpoint;
  - F1, EM (xanh nếu = 1, đỏ nếu = 0);
  - recall gold;
  - số giây, số lần truy xuất;
  - khi có sự kiện `error`: viền đỏ cùng lý do.
- **Phần câu hỏi:** hiện "Đáp án đúng" và danh sách tài liệu gold (tiêu đề, có ★). Có bộ lọc "Câu tiêu biểu" cạnh bộ lọc dataset.
- **Bố cục:**
  - cột GRASP · GRPO ghim bên phải (`position: sticky`) khi phải cuộn ngang;
  - chữ tiếng Việt;
  - phông chữ thân 16 px;
  - chế độ sáng và tối.
- **Không dùng `alert()`:** lỗi hiện thành một thanh báo trên trang.
- **Không dùng `innerHTML` cho dữ liệu:** mọi dữ liệu từ server đều chèn bằng `textContent`.
- **Lịch sử:** phát lại ở tốc độ 1× hoặc 4×, như hiện tại. Thêm nút "Chạy trước câu tiêu biểu", gọi `/api/prerun`.

## 6. Preflight và cách chọn máy

`python scripts/preflight.py --config models.<mode>.yaml [--report <test999 grasp-grpo-*.json>]`

1. Chạy health như mục 4 và in trạng thái từng thành phần.
2. Chọn 6 câu trong `test_ids.json`, mỗi dataset 2 câu (lấy cố định 2 id đầu của từng dataset). Mỗi câu chạy đủ 5 cột song song, qua đúng `run_column`.
3. Ghi lại:
   - số giây của từng cột trên từng câu;
   - tổng số giây của mỗi câu;
   - VRAM lúc cao nhất: một thread gọi `nvidia-smi --query-gpu=memory.used` mỗi giây, lấy tổng các GPU;
   - nếu có `--report`: câu trả lời của cột GRPO khớp `predictions` trong report hay không (so bằng `exact` sau khi chuẩn hóa).
4. Ghi kết quả ra `runs/preflight-<mode>-<thời gian>.json` và in bảng tóm tắt.

**Tiêu chí chọn máy local** (đạt cả 4):
1. Mọi thành phần sẵn sàng, ngoại trừ Bedrock.
2. VRAM lúc cao nhất ≤ 15 GB.
3. Mỗi câu chạy xong cả 5 cột trong ≤ 60 giây. Mỗi cột GRASP ≤ 40 giây.
4. Câu trả lời GRPO khớp report ở ≥ 4/6 câu.

Không đạt thì thuê máy 2×4090 và dùng `models.vast.yaml`.

**Thứ tự kiểm tra:**
1. Chạy preflight chế độ vast trên máy eval C.60319605 sau khi eval xong, trước khi destroy máy. Máy này có sẵn reader, policy và dữ liệu.
2. Sau đó chạy preflight local trên RTX 5060 Ti.
3. Dù chọn máy nào, hôm bảo vệ vẫn chạy trước các câu tiêu biểu và lưu replay để dự phòng.

## 7. Kiểm thử

**Pytest, không cần GPU hay mạng:**
- `fetch_data --skip-download` trên cây thư mục giả: tạo đúng `corpus.json`; thiếu index hoặc thiếu bảng `triples` thì báo đúng đường dẫn.
- `fetch_checkpoint` với `--subfolder`: đưa nội dung lên gốc. Dùng `snapshot_download` giả.
- `config`: `adapters` nhận cả dạng chuỗi lẫn `{path, label}`, đường dẫn tương đối được giải theo vị trí file yaml.
- `services`:
  - backend `vllm` nạp adapter qua client admin giả, gỡ khi đóng;
  - sai đường dẫn adapter thì tắt cột GRASP, kèm lý do;
  - thiếu key Bedrock thì chỉ tắt cột 235B;
  - reader không trả lời thì tắt 4 cột phụ thuộc reader.
- `runners`: mỗi cột (dùng reader, retriever và proxy giả) phát đủ sự kiện và kết thúc bằng `done`. Điểm trong `done` khớp với `grasp_grpo.evaluate`.
- `server`: một cột lỗi không chặn các cột khác. `/api/doc` trả toàn văn. `/api/prerun` trả 409 khi đang có lượt chạy trước.
- `store`: hai kết nối cùng nhận đủ sự kiện của một lần chạy, không trùng. Phát lại ra đúng chuỗi sự kiện.
- `preflight`: hàm chấm tiêu chí trả đúng kết quả đạt hay không với các số liệu giả.
- Giữ nguyên các test hiện có.

**Kiểm tra thủ công:**
- Mở trang bằng Chrome.
- Chạy 1 câu eval và 1 câu tự do (hoặc phát lại một replay nếu chưa có reader).
- Chụp màn hình cả chế độ sáng lẫn tối, và ở độ rộng 1366 px.

## 8. Ngoài phạm vi

- Không trích triple lúc chạy.
- Không thêm cột cho nhiều checkpoint GRPO cùng lúc. Chỉ dùng 1 checkpoint, đổi qua `models.yaml`.
- Không deploy công khai.
- Không sửa logic GRASP hay baseline: mọi thứ đều đi qua `grasp_grpo`.
