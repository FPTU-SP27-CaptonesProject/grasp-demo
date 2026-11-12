# Demo so sánh GRASP với các baseline RAG

- **Ngày:** 2026-11-11
- **Repo:** `grasp-demo` (mới), dùng `grasp-grpo-trainer` như thư viện
- **Mục đích:** trình bày đồ án. Gõ hoặc chọn một câu hỏi, chạy song song GRASP và các baseline RAG,
  rồi hiển thị từng bước: truy xuất gì, suy nghĩ gì, proxy quyết định gì, câu trả lời và điểm so với đáp án đúng.
- **Ngoài phạm vi:** đăng nhập, chat nhiều lượt, deploy công khai, chấm điểm hàng loạt. Số liệu báo cáo lấy từ
  script eval. Demo không tính số liệu báo cáo, nhưng dùng chung code baseline với script eval (mục 4).

## 1. Các cột

| Cột | Reader | Truy xuất | Quyết định truy xuất |
|---|---|---|---|
| RAG · 7B | Qwen2.5-7B-Instruct (máy local) | 1 lần, query = câu hỏi | không có |
| RAG · 235B | `qwen.qwen3-235b-a22b-2507` (Bedrock) | 1 lần, query = câu hỏi | không có |
| IRCoT · 7B | Qwen2.5-7B-Instruct | nhiều bước, query = thought vừa viết | reader tự quyết |
| GRASP · SFT | Qwen2.5-7B-Instruct | nhiều bước | proxy 3B, adapter SFT |
| GRASP · GRPO | Qwen2.5-7B-Instruct | nhiều bước | proxy 3B, adapter GRPO tốt nhất |

Để so sánh công bằng, mọi cột dùng chung các thứ sau:
- cùng retriever (E5-large-v2 + FAISS, có rescore), cùng corpus, cùng `top_k = 10`;
- cùng prompt reader của `grasp-grpo-trainer`;
- cùng `max_steps = 5` cho các cột nhiều bước;
- greedy (temperature 0).

Cột 235B dùng đúng prompt trả lời cuối của reader 7B, chỉ đổi model.

**Lý do chọn model:**
- Không nhà cung cấp nào trên Bedrock hay FPT AI Factory có Qwen2.5-7B-Instruct (danh sách kiểm tra ngày 2026-11-11), nên reader chạy trên máy local.
- Model mạnh chọn Qwen3-235B vì cùng họ Qwen với reader. Nó cũng là model đã dùng để trích triple, nên code Bedrock đã có sẵn.

## 2. Kiến trúc

```
Trình duyệt: 1 trang HTML (static/index.html)
  │  POST /api/ask {eval_id | question, corpus, columns[]}  →  {run_id}
  │  GET  /api/stream/{run_id}  ←  SSE: {column, type, data, t}
FastAPI (grasp_demo.server)
  ├─ runners: rag_7b · rag_235b · ircot_7b · grasp_sft · grasp_grpo
  └─ dịch vụ dùng chung (nạp một lần lúc khởi động)
       ├─ Retriever: grasp_grpo.retrieval (MultiCorpus) — E5 trên CPU, 3 index FAISS trong RAM,
       │             passages.sqlite + triples.sqlite; bọc bằng BoundRetriever (một lock)
       ├─ Reader 7B: llama-server (llama.cpp, CUDA, Windows) tại http://localhost:8080/v1,
       │             GGUF Qwen2.5-7B-Instruct-Q6_K, gọi qua grasp_grpo.reader (completions dạng prompt thô)
       ├─ Proxy: 1 base Qwen2.5-3B-Instruct bf16 trên GPU + 2 LoRA (peft: "sft", "grpo"),
       │         đổi bằng set_adapter, có một lock: mỗi lúc chỉ một cột sinh
       └─ Bedrock: client OpenAI tới bedrock-mantle.ap-southeast-2, key BEDROCK_API_KEY trong .env
```

Ước tính bộ nhớ trên GPU 16 GB (RTX 5060 Ti):
- reader Q6_K khoảng 6,3 GB, cộng KV cache khoảng 1 GB (ctx 8k);
- proxy bf16 khoảng 6,5 GB.

Tổng khoảng 14 GB. RAM cần khoảng 6 GB (index cùng E5). Ổ đĩa cần khoảng 21 GB dữ liệu.

**Reader:** code eval gọi `/v1/completions` với prompt đã áp chat template và có stop token.
- `llama-server` nhận prompt thô đúng như vậy, nên là mặc định.
- Ollama chỉ dùng thay thế nếu xác nhận được `/v1/completions` của nó không tự chèn template.
- Endpoint reader nằm trong `models.yaml`. Muốn chuyển sang vLLM hoặc một API khác thì chỉ đổi URL.

## 3. Cấu hình và checkpoint

`models.yaml` (đổi checkpoint bằng cách sửa một dòng `adapter:` rồi khởi động lại):

```yaml
reader:   {url: http://localhost:8080/v1, model: qwen2.5-7b-instruct, protocol: <như grasp_grpo>}
strong:   {url: https://bedrock-mantle.ap-southeast-2.api.aws/v1, model: qwen.qwen3-235b-a22b-2507}
proxy:
  base: Qwen/Qwen2.5-3B-Instruct
  adapters:
    sft:  checkpoints/grasp-qwen2.5-3b-sft          # HF doancuavahk/grasp-qwen2.5-3b-sft
    grpo: checkpoints/grpo-pilot-step100            # HF doancuavahk/grasp-qwen2.5-3b-grpo-pilot; sau thay bằng best/ của run full
retrieval: {corpus_config: data/corpus.json, top_k: 10, max_candidate_triples: 20, encoder_device: cpu}
ircot:    {max_steps: 5, docs_per_step: 3}
grasp:    {max_steps: 5}
eval:     {qa: data/grasp-gold-evidence/dev.jsonl, ids: data/eval_ids.json}
```

- `scripts/fetch_checkpoint.py <hf_repo> <tên>` tải adapter về `checkpoints/<tên>/`.
- Lúc khởi động, server đọc `grpo_config.json` của adapter để hiện step và eval F1 ở header cột, ví dụ "GRPO · pilot step 100 · F1 0,573".
- `scripts/fetch_data.py` tải từ HF:
  - `grasp-retrieval`: 3 corpus, index và `cache/triples.sqlite`;
  - `grasp-gold-evidence/dev.jsonl`;
  - `eval_ids.json`, lấy từ `grasp-grpo-trainer/config`.

## 4. Baseline dùng chung với báo cáo

`grasp-grpo-trainer/src/grasp_grpo/baselines.py` có hai hàm, theo cùng giao diện với `infer.run_example`:
- **`run_single_step(reader, retriever, example)`**:
  1. `context = search(question)`;
  2. trả lời bằng `reader.final(question, context, [], "")`.
- **`run_ircot(reader, retriever, example, max_steps, docs_per_step)`**, theo Trivedi và cộng sự (2023):
  1. `context = search(question)`.
  2. Mỗi bước gọi `reader.step`:
     - nếu reader viết `Answer:` thì dừng;
     - nếu không, lấy thought vừa viết làm query, `search(thought)`, rồi thêm `docs_per_step` tài liệu chưa có trong context vào context.
  3. Hết bước mà vẫn chưa có câu trả lời thì gọi `reader.final`.

Cả hai hàm (và `run_example`) nhận thêm `emit: Callable[[str, dict], None] | None = None`. Mặc định là `None`, nên hành vi eval không đổi.

Script eval (`grasp-eval-baseline --method single|ircot`) chạy hai baseline này trên 300 câu eval để lấy số liệu báo cáo. Demo gọi đúng các hàm đó, nên số liệu và hành vi trong demo khớp nhau.

Cột GRASP dùng `infer.run_example` có `emit`, chứ không dùng một bản sao. Vì vậy demo luôn chạy đúng logic của eval.

## 5. Sự kiện và giao diện

**Sự kiện** (mỗi sự kiện kèm `column`, `t` tính bằng giây từ lúc bắt đầu):

| type | data |
|---|---|
| `retrieve` | `{query, docs: [{doc_id, title, text, score, gold}]}` |
| `thought` | `{step, text}` |
| `rd` | `{step, retrieve, raw, valid}` |
| `qf` | `{step, query, chain, raw, valid}` |
| `ki` | `{step, candidates, chosen, raw, valid, added_doc}` |
| `answer` | `{text, raw}` |
| `done` | `{seconds, retrievals, f1?, em?, recall?}` (chỉ có điểm ở chế độ eval) |
| `error` | `{message}` |

**Trang web:**
- Thanh trên cùng gồm:
  - chọn câu eval: lọc theo dataset, ô tìm kiếm, nút "Ngẫu nhiên";
  - ô gõ tự do kèm chọn corpus (mặc định wiki18);
  - checkbox ẩn hoặc hiện từng cột, nút "Hỏi" và nút "Lịch sử".
- Ở chế độ eval có thêm dòng "Đáp án đúng" và danh sách tài liệu gold.
- **5 cột** đặt ngang nhau. Màn hình hẹp hơn 1600 px thì cuộn ngang, cột GRASP · GRPO được ghim ở bên phải.
- **Header mỗi cột:** tên, F1, EM, recall gold (xanh hoặc đỏ), thời gian chạy, số lần truy xuất. Riêng hai cột GRASP có thêm nhãn checkpoint.
- **Thân mỗi cột:**
  - danh sách các bước, mở rộng hoặc thu gọn được;
  - tài liệu gold đánh dấu ★ và tô màu, bấm vào tài liệu để xem toàn văn;
  - action sai format có nhãn "⚠ sai format";
  - câu trả lời ở cuối cột.
- Giao diện có chế độ sáng và tối, phông đủ lớn để chiếu.

**Lịch sử và phát lại:**
- Mỗi lần chạy được lưu thành `runs/<thời gian>-<id>.jsonl`, gồm câu hỏi, cấu hình, checkpoint và mọi sự kiện kèm `t`.
- "Lịch sử" liệt kê các lần chạy. "Phát lại" chiếu lại đúng nhịp, có thể chọn tốc độ 1× hoặc 4×.

Đây là phương án dự phòng khi trình bày: chạy trước các câu tiêu biểu, rồi phát lại nếu máy hoặc mạng có vấn đề.

## 6. Thực thi và xử lý lỗi

**Thực thi:**
- Mỗi câu hỏi tạo 5 task chạy song song (thread pool).
- Reader 7B: `llama-server --parallel 4`. Bedrock gọi song song.
- Retriever có lock: E5 trên CPU mất khoảng 0,2 giây mỗi query.
- Proxy có lock: hai cột GRASP dùng proxy lần lượt.

**Xử lý lỗi:**
- Lỗi ở cột nào chỉ dừng cột đó (sự kiện `error`), các cột khác chạy tiếp.
- Reader và Bedrock: timeout 60 giây, client OpenAI tự thử lại.
- Proxy sinh sai format thì xử lý giống eval: RD sai thì coi như truy xuất; QF sai thì dùng câu hỏi gốc; KI sai thì lấy ứng viên đầu tiên.
- **Kiểm tra lúc khởi động**, kết quả hiện ở thanh trạng thái `/api/health`:
  - reader trả lời được một prompt ngắn;
  - key Bedrock hợp lệ (gọi `GET /models`);
  - các thư mục adapter có `adapter_config.json`;
  - index và sqlite mở được.

  Thành phần nào hỏng thì tắt cột phụ thuộc vào nó và ghi rõ lý do. Server vẫn chạy.
- Câu tự do không có đáp án nên không chấm điểm. Header chỉ hiện thời gian và số lần truy xuất.

## 7. Kiểm thử

**Trong `grasp-grpo-trainer`:**
- Truyền `emit` vào `run_example` không làm đổi `prediction`, `context_doc_ids` hay `actions`. Kiểm tra bằng reader, retriever và proxy giả.
- `run_ircot`:
  - query mỗi bước là thought vừa viết;
  - mỗi bước thêm đúng `docs_per_step` tài liệu mới, không trùng;
  - dừng khi có `Answer:` hoặc hết bước.
- `run_single_step` chỉ truy xuất một lần.

**Trong `grasp-demo`:**
- Gọi `/api/ask` với các runner giả: nhận đủ sự kiện của các cột đã chọn, có `done` cho từng cột. Một cột lỗi không chặn các cột khác.
- Chế độ eval: F1, EM và recall trong `done` khớp với `grasp_grpo` metric khi cho cùng dự đoán.
- Phát lại một file trong `runs/` cho ra đúng chuỗi sự kiện.
- `models.yaml` sai đường dẫn adapter: health báo lỗi và cột GRASP bị tắt.

**Kiểm tra thủ công trên máy local:** chạy 5 câu eval, mỗi dataset ít nhất một câu, cộng 1 câu tự do. Ghi lại thời gian từng cột và VRAM lúc cao nhất.

## 8. Cài đặt (Windows, RTX 5060 Ti 16 GB)

1. **Reader:**
   - Tải bản `llama-server` CUDA cho Windows và `Qwen2.5-7B-Instruct-Q6_K.gguf`.
   - Chạy `scripts/run_reader.ps1`, tương đương `llama-server -m ... -ngl 99 -c 8192 --parallel 4 --port 8080`.
2. **Môi trường Python:**
   - `python -m venv .venv`.
   - Cài PyTorch bản CUDA 12.8 trở lên (Blackwell cần bản này).
   - `pip install -e ../grasp-grpo-trainer -e .`
3. **Dữ liệu và checkpoint:**
   - `python scripts/fetch_data.py`.
   - `python scripts/fetch_checkpoint.py doancuavahk/grasp-qwen2.5-3b-sft grasp-qwen2.5-3b-sft`.
   - `python scripts/fetch_checkpoint.py doancuavahk/grasp-qwen2.5-3b-grpo-pilot grpo-pilot-step100`.
4. **Cấu hình và chạy:**
   - Tạo `.env` có `BEDROCK_API_KEY`, chép từ `grasp/.env`. File này nằm trong `.gitignore`.
   - `python -m grasp_demo --config models.yaml`, rồi mở `http://localhost:7860`.

## 9. Việc mở và rủi ro

- **Chênh lệch reader:** bản GGUF Q6_K có thể trả lời hơi khác bản FP8 trên vLLM dùng lúc eval. Báo cáo ghi rõ demo dùng reader lượng tử hóa.
- **Tốc độ:** proxy 3B chạy bằng transformers bf16 sinh khoảng 30–50 token/giây. Một câu GRASP 3–5 bước dự kiến mất khoảng 10–25 giây. Nếu quá chậm có hai hướng: nạp proxy 4-bit, hoặc chuyển proxy sang llama-server thứ hai (GGUF base + LoRA).
- **Checkpoint mới:** khi run full có `best/`, tải về rồi sửa `models.yaml`. Không cần đổi code.
- **Baseline cho báo cáo:** phải chạy eval single-step và IRCoT trên 300 câu (Vast hoặc máy local) với reader FP8 để số liệu cùng điều kiện với GRASP. Việc này thuộc kế hoạch báo cáo.
