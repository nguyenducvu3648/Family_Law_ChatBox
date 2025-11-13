import google.generativeai as genai
from core.logging_setup import app_log, log_time
from models.models import GEMINI_MODEL_ID


INSTRUCTIONS = (
    "You are an expert at reformulating questions to be more precise and detailed.\n"
    "Your task is to:\n"
    "1. Analyze the user's question\n"
    "2. Rewrite it to be more specific and search-friendly\n"
    "3. Expand any acronyms or technical terms\n"
    "4. Return ONLY the rewritten query without any additional text or explanations"
)


def _get_model():
    # No system instruction to avoid the model returning anything but the rewritten query
    return genai.GenerativeModel(model_name=GEMINI_MODEL_ID)


@log_time
def rewrite_query(query: str) -> str:
    if not (query and query.strip()):
        return query
    try:
        model = _get_model()
        prompt = (
            f"{INSTRUCTIONS}\n\n"
            f"User question: {query}\n"
            f"Output:"
        )
        cfg = genai.types.GenerationConfig(temperature=0.0, max_output_tokens=96)
        resp = model.generate_content(prompt, generation_config=cfg)
        text = (getattr(resp, "text", None) or "").strip()
        # Guard against accidental extra formatting
        out = text.splitlines()[0].strip() if text else query
        app_log.info("Query rewritten", extra={"__kv__": {"from": query[:120], "to": out[:120]}})
        return out or query
    except Exception as e:
        app_log.warning("Failed to rewrite query", extra={"__kv__": {"error": str(e)}})
        return query


