from tenacity import retry, stop_after_attempt, wait_exponential
from core.logging_setup import app_log, log_step, log_time
from models.models import answer_model
import time
import google.generativeai as genai

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _gemini_stream(prompt, temperature: float):
    cfg = genai.types.GenerationConfig(temperature=float(temperature))
    return answer_model.generate_content(prompt, generation_config=cfg, stream=True)

@log_time
def stream_answer(prompt, temperature=0.2):
    t0 = time.perf_counter()
    t_first0 = time.perf_counter()
    first_token_emitted = False

    try:
        resp = _gemini_stream(prompt, temperature)
        for ch in resp:
            if getattr(ch, "text", None):
                if not first_token_emitted:
                    log_step("llm_first_token", thoi_gian_truoc=f"{time.perf_counter()-t_first0:.4f}")
                    first_token_emitted = True
                yield ch.text

    except Exception as e:
        app_log.error("Lỗi gọi mô hình LLM", extra={"__kv__": {"loi": str(e)}})
        yield f"\n\nLỗi gọi mô hình: {e}"

    finally:
        log_step("llm_tong", thoi_gian=f"{time.perf_counter()-t0:.4f}")
