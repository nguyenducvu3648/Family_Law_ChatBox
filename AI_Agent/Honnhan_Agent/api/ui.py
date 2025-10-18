import asyncio
import time
from datetime import datetime
from typing import List, Dict, Any
import gradio as gr

from core.logging_setup import app_log, log_step, log_time
from agents.intent import analyze_intent
from retrieval.search import search_law
from services.render import docs_to_markdown, docs_page_markdown, paginate_docs
from services.prompt import build_prompt
from agents.llm import stream_answer
from retrieval.fetch import _fetch
from core.config import INTENT_FALLBACK_CASUAL, CASUAL_MAX_WORDS

CSS = """
#chatbot { height: 540px !important; }
label { font-size:12px !important; opacity:.9 }
#cites-box {
    max-height: 360px;
    overflow-y: auto;
    border: 1px solid #ddd;
    padding: 6px;
    border-radius: 6px;
    background-color: #fafafa;
}
#bm25-box, #emb-box {
    max-height: 200px;
    overflow-y: auto;
    border: 1px solid #ddd;
    padding: 6px;
    border-radius: 6px;
    background-color: #fafafa;
}
"""

def ui_return(msg_val, chatbot_val, bm25_val, emb_val, cites_val, last_answer_val, docs_val, page_val, page_label_val, history_msgs):
    """Helper function đảm bảo luôn trả về đúng 10 giá trị"""
    print("DEBUG: Gọi hàm ui_return, trả về 10 giá trị")
    return (
        msg_val,
        chatbot_val,
        gr.update(value=bm25_val),
        gr.update(value=emb_val),
        gr.update(value=cites_val),
        last_answer_val,
        docs_val,
        page_val,
        page_label_val,
        history_msgs,
    )

@log_time
def respond_generator(message, history_msgs, cur_page_size, k=15, temperature=0.2, threshold=0.42):
    """Generator function - yield từng update dần dần (sync wrapper cho async code)"""
    print(f"DEBUG: Bắt đầu xử lý câu hỏi: {message}")
    if not (message and message.strip()):
        print("DEBUG: Câu hỏi rỗng, trả về mặc định")
        gr.Info("Vui lòng nhập câu hỏi.")
        yield ui_return(
            gr.update(value=""),
            history_msgs,
            "",
            "",
            "",
            "",
            [],
            1,
            " Trang 0/0",
            history_msgs,
        )
        return

    t_overall0 = time.perf_counter()
    try:
        # Phân tích ý định
        print("DEBUG: Gọi hàm phân tích ý định")
        intent_info = analyze_intent(message)
        print(f"DEBUG: Kết quả ý định: {intent_info}")
        intent = intent_info["intent"]
        intent_answer = intent_info.get("answer", "")
        normalized_query = intent_info.get("normalized_query", message)
        original_query = intent_info.get("original_query", message)
        intent_filters = intent_info.get("filters", {})

        # ========== XỬ LÝ CÂU HỎI XÃ GIAO ==========
        if intent == "casual":
            final_answer = (intent_answer or "").replace("\u200b", "").strip()
            app_log.info(
                "Xử lý câu hỏi xã giao",
                extra={"__kv__": {"do_dai_tra_loi": len(final_answer)}},
            )

            if final_answer and CASUAL_MAX_WORDS > 0:
                words = final_answer.split()
                if len(words) > CASUAL_MAX_WORDS:
                    truncated = " ".join(words[:CASUAL_MAX_WORDS])
                    app_log.info(
                        "Cắt ngắn câu trả lời xã giao",
                        extra={
                            "__kv__": {
                                "so_tu_goc": len(words),
                                "so_tu_giu": CASUAL_MAX_WORDS,
                                "do_dai_goc": len(final_answer),
                            }
                        },
                    )
                    final_answer = truncated

            # Nếu có câu trả lời trực tiếp từ intent
            if len(final_answer) >= 1:
                history_msgs = history_msgs + [
                    {"role": "user", "content": message},
                    {"role": "assistant", "content": final_answer},
                ]
                print("DEBUG: Trả về câu trả lời xã giao trực tiếp")
                yield ui_return(
                    gr.update(value=""),
                    history_msgs,
                    "(Không có trích dẫn)",
                    "(Không có trích dẫn)",
                    "(Không có trích dẫn)",
                    final_answer,
                    [],
                    1,
                    " Trang 0/0",
                    history_msgs,
                )
                return

            # Stream câu trả lời xã giao
            simple_prompt = "Trả lời thân thiện ngắn gọn (<=2 câu) tiếng Việt cho câu: " + message
            history_msgs = history_msgs + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": ""},
            ]
            acc = ""
            print("DEBUG: Bắt đầu stream câu trả lời xã giao")
            
            # Yield initial state
            yield ui_return(
                gr.update(value=""),
                history_msgs,
                "(Không có trích dẫn)",
                "(Không có trích dẫn)",
                "(Không có trích dẫn)",
                acc,
                [],
                1,
                " Trang 0/0",
                history_msgs,
            )
            
            # Stream từng chunk
            buffer = ""
            for chunk in stream_answer(simple_prompt, temperature=float(temperature)):
                buffer += chunk
                if len(buffer) >= 50:  # Tích lũy 50 ký tự mới yield
                    acc += buffer
                    history_msgs[-1]["content"] = acc
                    yield ui_return(
                        gr.update(value=""),
                        history_msgs,
                        "(Không có trích dẫn)",
                        "(Không có trích dẫn)",
                        "(Không có trích dẫn)",
                        acc,
                        [],
                        1,
                        " Trang 0/0",
                        history_msgs,
                    )
                    buffer = ""
            
            # Yield phần còn lại
            if buffer:
                acc += buffer
                history_msgs[-1]["content"] = acc
                yield ui_return(
                    gr.update(value=""),
                    history_msgs,
                    "(Không có trích dẫn)",
                    "(Không có trích dẫn)",
                    "(Không có trích dẫn)",
                    acc,
                    [],
                    1,
                    " Trang 0/0",
                    history_msgs,
                )
            print("DEBUG: Hoàn thành stream câu trả lời xã giao")
            return

        # ========== XỬ LÝ CÂU HỎI PHÁP LÝ ==========
        docs: List[Dict[str, Any]] = []
        bm25_docs: List[Dict[str, Any]] = []
        emb_docs: List[Dict[str, Any]] = []
        source = None

        if intent == "law_search":
            print("DEBUG: Tìm kiếm điều luật")
            docs = _fetch(intent_filters, limit=int(k)) if intent_filters else []
            source = "law_search"
            if not docs:
                app_log.info(
                    "Rơi vào tìm kiếm embedding",
                    extra={"__kv__": {"cau_hoi": message}},
                )
                # Gọi async function trong sync context - dùng nest_asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    bm25_docs, emb_docs, docs = loop.run_until_complete(
                        search_law(message, top_k=int(k), score_threshold=float(threshold))
                    )
                finally:
                    loop.close()
                source = "law_search_embedding_fallback"

        elif intent == "legal_answer":
            print("DEBUG: Tìm kiếm câu trả lời pháp lý")
            # Gọi async function trong sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                bm25_docs, emb_docs, docs = loop.run_until_complete(
                    search_law(normalized_query, top_k=int(k), score_threshold=float(threshold))
                )
            finally:
                loop.close()
            source = "legal_answer"

        else:
            reply = INTENT_FALLBACK_CASUAL
            history_msgs = history_msgs + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ]
            print("DEBUG: Trả về ý định mặc định")
            yield ui_return(
                gr.update(value=""),
                history_msgs,
                "(Không có trích dẫn)",
                "(Không có trích dẫn)",
                "(Không có trích dẫn)",
                reply,
                [],
                1,
                " Trang 0/0",
                history_msgs,
            )
            return

        if not docs:
            reply = (
                "Chưa tìm thấy cơ sở pháp lý phù hợp. "
                "Bạn có thể bổ sung Điều/Khoản hoặc thêm bối cảnh."
            )
            upd = history_msgs + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply},
            ]
            print("DEBUG: Không tìm thấy tài liệu")
            yield ui_return(
                gr.update(value=""),
                upd,
                "(Chưa có dữ liệu)",
                "(Chưa có dữ liệu)",
                "(Chưa có dữ liệu)",
                reply,
                [],
                1,
                " Trang 0/0",
                upd,
            )
            return

        # Chuẩn bị prompt và markdown
        if intent == "legal_answer":
            user_query = original_query or message
        elif intent == "law_search":
            user_query = message
        else:
            user_query = message
            
        bm25_markdown = docs_to_markdown(bm25_docs)
        emb_markdown = docs_to_markdown(emb_docs)
        cites_markdown, page_label = docs_page_markdown(docs, 1, int(cur_page_size))
        prompt = build_prompt(user_query, docs, history_msgs)

        log_step("llm_chuanbi", so_tai_lieu=len(docs), nguon=source)
        print(f"DEBUG: Đã chuẩn bị prompt, số tài liệu: {len(docs)}")

        # Chuẩn bị history
        history_msgs = history_msgs + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": ""},
        ]
        acc = ""
        
        print("DEBUG: Bắt đầu stream câu trả lời pháp lý")
        
        # Yield initial state với các markdown
        yield ui_return(
            gr.update(value=""),
            history_msgs,
            bm25_markdown,
            emb_markdown,
            cites_markdown,
            acc,
            docs,
            1,
            page_label,
            history_msgs,
        )
        
        # Stream từng chunk
        buffer = ""
        for chunk in stream_answer(prompt, temperature=float(temperature)):
            buffer += chunk
            if len(buffer) >= 50:  # Tích lũy 50 ký tự mới yield
                acc += buffer
                history_msgs[-1]["content"] = acc
                yield ui_return(
                    gr.update(value=""),
                    history_msgs,
                    bm25_markdown,
                    emb_markdown,
                    cites_markdown,
                    acc,
                    docs,
                    1,
                    page_label,
                    history_msgs,
                )
                buffer = ""
        
        # Yield phần còn lại
        if buffer:
            acc += buffer
            history_msgs[-1]["content"] = acc
            yield ui_return(
                gr.update(value=""),
                history_msgs,
                bm25_markdown,
                emb_markdown,
                cites_markdown,
                acc,
                docs,
                1,
                page_label,
                history_msgs,
            )
        print("DEBUG: Hoàn thành stream câu trả lời pháp lý")
        return

    except Exception as e:
        app_log.error("Lỗi xử lý câu hỏi", extra={"__kv__": {"loi": str(e)}})
        print(f"DEBUG: Lỗi trong xử lý: {e}")
        yield ui_return(
            gr.update(value=""),
            history_msgs,
            "(Lỗi hệ thống)",
            "(Lỗi hệ thống)",
            "(Lỗi hệ thống)",
            f"Lỗi: {e}",
            [],
            1,
            " Trang 0/0",
            history_msgs,
        )
        return

def respond_wrapper(message, history_msgs, cur_page_size, k=15, temperature=0.2, threshold=0.42):
    """Wrapper để Gradio gọi - chuyển tiếp từ generator"""
    for output in respond_generator(message, history_msgs, cur_page_size, k, temperature, threshold):
        yield output

def build_ui():
    with gr.Blocks(
        title="⚖️ Trợ lý Luật Hôn Nhân & Gia Đình 2014",
        css=CSS,
    ) as demo:
        gr.Markdown("""
        ### ⚖️ Trợ lý Luật Hôn Nhân & Gia đình 2014
        *Tham chiếu chính xác • Hạn chế suy diễn • Không thay thế tư vấn pháp lý*
        """)

        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    value=[],
                    type="messages",
                    show_copy_button=True,
                    elem_id="chatbot",
                    autoscroll=True,
                )
                with gr.Row():
                    ex1 = gr.Button("Chào bạn")
                    ex2 = gr.Button("Điều 81 quy định gì về việc nuôi con sau ly hôn")
                    ex3 = gr.Button("Khoản 2 Điều 56 nói gì")
            with gr.Column(scale=5):
                gr.Markdown("**📜 Kết quả BM25**")
                bm25_md = gr.Markdown(value="(Chưa có dữ liệu)", elem_id="bm25-box")
                
                gr.Markdown("**📜 Kết quả Embedding Search**")
                emb_md = gr.Markdown(value="(Chưa có dữ liệu)", elem_id="emb-box")
                gr.Markdown("**Cơ sở pháp lý**")
                cites_md = gr.Markdown(value="(Chưa có dữ liệu)", elem_id="cites-box")
                with gr.Row():
                    prev_page = gr.Button("⬅️")
                    next_page = gr.Button("➡️")
                with gr.Row():
                    page_info = gr.Markdown(" Trang 0/0")
                    page_size = gr.Slider(3, 20, value=5, step=1, label="Mỗi trang")

        with gr.Row():
            msg = gr.Textbox(placeholder="Nhập câu hỏi...", scale=5, autofocus=False)
            send = gr.Button("Gửi", variant="primary", scale=1)
            clear = gr.Button("Làm mới", scale=1)

        # Điền sẵn ví dụ
        def _fill(text):
            return text

        ex1.click(lambda: _fill("Chào bạn"), outputs=msg)
        ex2.click(
            lambda: _fill("Điều 81 quy định gì về việc nuôi con sau ly hôn"),
            outputs=msg,
        )
        ex3.click(lambda: _fill("Khoản 2 Điều 56 nói gì"), outputs=msg)

        # Trạng thái
        state_history = gr.State([])
        state_last_answer = gr.State("")
        state_docs = gr.State([])
        state_page = gr.State(1)

        # Kết nối outputs (10 giá trị)
        outputs = [
            msg,                  # 1
            chatbot,              # 2
            bm25_md,              # 3
            emb_md,               # 4
            cites_md,             # 5
            state_last_answer,    # 6
            state_docs,           # 7
            state_page,           # 8
            page_info,            # 9
            state_history,        # 10
        ]
        
        # Kết nối với wrapper (BẬT queue=True để hỗ trợ streaming)
        send.click(respond_wrapper, inputs=[msg, state_history, page_size], outputs=outputs, queue=True)
        msg.submit(respond_wrapper, inputs=[msg, state_history, page_size], outputs=outputs, queue=True)

        # Like/Dislike
        def on_like(data: gr.LikeData):
            msg_like = data.value or {}
            role = msg_like.get("role", "assistant")
            text = msg_like.get("content", "")
            app_log.info(
                "Phản hồi người dùng",
                extra={"__kv__": {"thich": data.liked, "vai_tro": role, "do_dai": len(text or "")}},
            )
            return None

        chatbot.like(on_like)

        # Phân trang
        def render_cites_for_page(docs, page, cur_page_size):
            md, label = docs_page_markdown(docs or [], int(page), int(cur_page_size))
            return gr.update(value=md), int(page), label

        def go_prev(docs, page, cur_page_size):
            if not docs:
                return render_cites_for_page([], 1, cur_page_size)
            new_page = max(1, int(page) - 1)
            return render_cites_for_page(docs, new_page, cur_page_size)

        def go_next(docs, page, cur_page_size):
            if not docs:
                return render_cites_for_page([], 1, cur_page_size)
            _, total, total_pages, _ = paginate_docs(docs, 1, int(cur_page_size))
            new_page = min(total_pages if total_pages > 0 else 1, int(page) + 1)
            return render_cites_for_page(docs, new_page, cur_page_size)

        def on_change_page_size(docs, cur_page_size):
            return render_cites_for_page(docs, 1, cur_page_size)

        prev_page.click(
            go_prev,
            inputs=[state_docs, state_page, page_size],
            outputs=[cites_md, state_page, page_info],
            queue=False,
        )
        next_page.click(
            go_next,
            inputs=[state_docs, state_page, page_size],
            outputs=[cites_md, state_page, page_info],
            queue=False,
        )
        page_size.release(
            on_change_page_size,
            inputs=[state_docs, page_size],
            outputs=[cites_md, state_page, page_info],
            queue=False,
        )

        gr.Markdown(f"""
        <sub>© {datetime.now().year} — Nội dung chỉ mang tính tham khảo, không thay thế tư vấn pháp lý chính thức.</sub>
        """)

    return demo