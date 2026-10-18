import time

import streamlit as st
from llmlingua import PromptCompressor


MODEL_NAME = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"


@st.cache_resource(show_spinner=False)
def load_compressor() -> PromptCompressor:
    return PromptCompressor(
        model_name=MODEL_NAME,
        device_map="cpu",
        use_llmlingua2=True,
    )


st.set_page_config(
    page_title="Prompt Compression Studio",
    page_icon="✂️",
    layout="wide",
)

st.title("Prompt Compression Studio")
st.caption(
    "Compress long prompts with LLMLingua-2, preserve important terms, "
    "and compare token usage."
)

with st.sidebar:
    st.header("Compression settings")
    rate = st.slider(
        "Target compression rate",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="Higher values retain more of the original prompt.",
    )
    force_tokens_text = st.text_area(
        "Keep these terms",
        placeholder="One term per line, or separate with commas",
        help="These tokens are protected from removal during compression.",
        height=120,
    )
    st.caption("Model: LLMLingua-2 multilingual BERT")
    st.caption("Device: CPU")

left, right = st.columns(2)
with left:
    prompt = st.text_area(
        "Text to compress",
        placeholder="Paste or write your prompt here...",
        height=300,
        key="prompt_input",
    )
    compress_clicked = st.button(
        "Compress prompt",
        type="primary",
        use_container_width=True,
        disabled=not prompt.strip(),
    )

if compress_clicked:
    force_tokens = [
        token.strip()
        for token in force_tokens_text.replace(",", "\n").splitlines()
        if token.strip()
    ]

    with st.spinner("Loading the model and compressing your prompt..."):
        compressor = load_compressor()
        started_at = time.perf_counter()
        result = compressor.compress_prompt(
            [prompt],
            rate=rate,
            force_tokens=force_tokens,
        )
        elapsed_seconds = time.perf_counter() - started_at

    required_keys = {"compressed_prompt", "origin_tokens", "compressed_tokens"}
    missing_keys = required_keys.difference(result)
    if missing_keys:
        raise ValueError(
            f"LLMLingua returned an incomplete result; missing: "
            f"{', '.join(sorted(missing_keys))}"
        )

    st.session_state["compression_result"] = {
        "compressed_prompt": result["compressed_prompt"],
        "origin_tokens": result["origin_tokens"],
        "compressed_tokens": result["compressed_tokens"],
        "elapsed_seconds": elapsed_seconds,
        "reported_ratio": result.get("ratio"),
    }

compression_result = st.session_state.get("compression_result")
with right:
    st.text_area(
        "Compressed output",
        value=(
            compression_result["compressed_prompt"]
            if compression_result
            else ""
        ),
        placeholder="Your compressed prompt will appear here...",
        height=300,
        disabled=True,
    )
    if compression_result:
        st.download_button(
            "Download compressed text",
            data=compression_result["compressed_prompt"],
            file_name="compressed_prompt.txt",
            mime="text/plain",
            use_container_width=True,
        )

if compression_result:
    original_tokens = compression_result["origin_tokens"]
    compressed_tokens = compression_result["compressed_tokens"]
    tokens_saved = original_tokens - compressed_tokens
    reduction = (
        (tokens_saved / original_tokens * 100) if original_tokens else 0
    )
    reported_ratio = compression_result["reported_ratio"]
    ratio = (
        f"{reported_ratio:.2f}x"
        if isinstance(reported_ratio, (int, float))
        else (
            f"{original_tokens / compressed_tokens:.2f}x"
            if compressed_tokens
            else "N/A"
        )
    )

    st.subheader("Compression metrics")
    metric_columns = st.columns(5)
    metric_columns[0].metric("Original tokens", f"{original_tokens:,}")
    metric_columns[1].metric("Compressed tokens", f"{compressed_tokens:,}")
    metric_columns[2].metric("Tokens saved", f"{tokens_saved:,}")
    metric_columns[3].metric("Token reduction", f"{reduction:.1f}%")
    metric_columns[4].metric("Compression ratio", ratio)
    st.caption(
        f"Compression completed in "
        f"{compression_result['elapsed_seconds']:.2f} seconds."
    )
