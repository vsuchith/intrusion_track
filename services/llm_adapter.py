import os
def generate_with_llama2(prompt: str) -> str:
    try:
        from llama_cpp import Llama
        os.environ["LLAMA2_PATH"]="/models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
        path = os.environ.get("LLAMA2_PATH","")
        if not path or not os.path.exists(path): return ""
        llm = Llama(model_path=path, n_ctx=4096, n_threads=4)
        out = llm(prompt=prompt, max_tokens=256, stop=["</s>","User:"], temperature=0.1)
        return out["choices"][0]["text"].strip()
    except Exception:
        return ""
