import os

def load_prompt(agent_folder, prompt_type):
    """
    Đọc prompt cho agent từ file txt.
    agent_folder: tên thư mục agent (ví dụ: "Honnhan_Agent")
    prompt_type: loại prompt ("intent" hoặc "answer")
    """
    file_path = os.path.join(agent_folder, "prompt", f"{prompt_type}.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()