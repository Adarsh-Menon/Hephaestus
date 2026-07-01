"""
Fine-tuned SLM playground — Streamlit.

Features:
  - enter your Hugging Face access token to test private/hosted models
  - pick from your fine-tuned models (or add a custom repo id)
  - chat input with streaming responses
  - two modes: HF Inference API (lightweight, deployable) or Local (transformers,
    needs a GPU / enough RAM)

Run:  streamlit run app.py
"""
import streamlit as st

# ---- your fine-tuned models (edit / add more here) ----------------------
MODELS = {
    "UAE VAT — Llama-3.2-3B (yours)":            "adarshsm28/uae-vat-llama3.2-3b",
}  
    
VAT_SYSTEM = (
    "You are a UAE VAT information assistant. Answer only about UAE VAT, based on "
    "FTA rules. The standard rate is 5%. The ONLY registration thresholds are "
    "AED 375,000 (mandatory) and AED 187,500 (voluntary) — never invent others. "
    "Everyday goods like food and clothing are standard-rated, not zero-rated. "
    "If unsure, say so and advise checking tax.gov.ae. This is general information, "
    "not tax advice."
)

st.set_page_config(page_title="Fine-tuned SLM playground", page_icon="🧪", layout="centered")

# ---- sidebar: config ----------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")

    hf_token = st.text_input(
        "Hugging Face access token", type="password",
        help="Create one at huggingface.co/settings/tokens (a READ token is enough).",
    )

    label = st.selectbox("Model", list(MODELS.keys()))
    model_id = MODELS[label]
    if model_id == "__custom__":
        model_id = st.text_input("Custom model id", "adarshsm28/uae-vat-llama3.2-3b")

    mode = st.radio(
        "Inference mode", ["HF Inference API", "Local (transformers)"],
        help="API: fast, no local GPU, uses your token. Local: loads the model on "
             "this machine — needs a GPU / plenty of RAM (a 3B model won't fit a free "
             "Streamlit Cloud instance).",
    )

    adapter_id = ""
    if mode == "Local (transformers)":
        adapter_id = st.text_input("LoRA adapter id (optional)",
                                   help="Leave blank if the model is already merged.")

    st.divider()
    st.subheader("Generation")
    max_new_tokens = st.slider("Max new tokens", 32, 1024, 384, 32)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.3, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    rep_penalty = st.slider("Repetition penalty", 1.0, 1.5, 1.1, 0.05)

    st.divider()
    system_prompt = st.text_area("System prompt", VAT_SYSTEM, height=160)
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---- header -------------------------------------------------------------
st.title("🧪 Fine-tuned SLM playground")
st.caption(f"Model: `{model_id}`  ·  Mode: {mode}")
st.info("⚠️ Educational demo — not tax/legal advice. Small models can be inaccurate; "
        "verify against tax.gov.ae.", icon="⚠️")

if "messages" not in st.session_state:
    st.session_state.messages = []


# ---- inference backends -------------------------------------------------
def api_stream(messages):
    from huggingface_hub import InferenceClient
    client = InferenceClient(model=model_id, token=hf_token or None)
    stream = client.chat_completion(
        messages=messages, max_tokens=max_new_tokens,
        temperature=max(temperature, 0.01), top_p=top_p, stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta


@st.cache_resource(show_spinner="Loading model locally…")
def load_local(model_id, adapter_id, token):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, token=token or None)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto", token=token or None)
    if adapter_id:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_id, token=token or None)
    model.eval()
    return tok, model


def local_stream(messages):
    import threading
    from transformers import TextIteratorStreamer
    tok, model = load_local(model_id, adapter_id, hf_token)
    inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                     return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs = dict(input_ids=inputs, max_new_tokens=max_new_tokens, do_sample=temperature > 0,
                  temperature=max(temperature, 0.01), top_p=top_p,
                  repetition_penalty=rep_penalty, pad_token_id=tok.eos_token_id,
                  streamer=streamer)
    threading.Thread(target=model.generate, kwargs=kwargs).start()
    for token in streamer:
        yield token


# ---- chat loop ----------------------------------------------------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask about UAE VAT…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    convo = ([{"role": "system", "content": system_prompt}] if system_prompt else [])
    convo += st.session_state.messages

    with st.chat_message("assistant"):
        try:
            gen = api_stream(convo) if mode == "HF Inference API" else local_stream(convo)
            answer = st.write_stream(gen)
        except Exception as e:
            answer = None
            st.error(
                f"Inference failed: {e}\n\n"
                "Tips: check your token; for API mode a custom fine-tune may not be "
                "served serverless (try Local mode or deploy an Inference Endpoint); "
                "for Local mode you need a GPU / enough RAM."
            )
    if answer:
        st.session_state.messages.append({"role": "assistant", "content": answer})
