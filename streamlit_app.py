import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer

st.set_page_config(page_title="视频高光提取", layout="centered")

# ---------- 加载轻量模型 ----------
@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor, tokenizer, device

st.markdown(
    """
    <style>
    .stApp { background: #ffffff; }
    .stButton > button { 
        background: #000000; 
        color: white; 
        border: none; 
        border-radius: 0; 
        padding: 0.5rem 1.5rem; 
        width: 100%;
    }
    .result-box {
        border-bottom: 1px solid #eee;
        padding: 0.8rem 0;
    }
    .time {
        font-size: 1rem;
        font-weight: 500;
    }
    .score {
        color: #888;
        font-size: 0.8rem;
        margin-left: 1rem;
    }
    .desc {
        background: #f5f5f5;
        padding: 0.5rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h2 style='font-weight:400'>视频高光提取</h2><hr>", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["mp4", "avi", "mov"], label_visibility="collapsed")

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("开始分析"):
        model, processor, tokenizer, device = load_model()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        frames = []
        progress = st.progress(0)
        status = st.empty()

        status.text("读取视频...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) < 2:
            st.error("视频过短")
        else:
            status.text("计算运动分数...")
            motion = [0.0]
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i-1], frames[i])
                motion.append(np.mean(diff) / 255.0)
                if i % 50 == 0:
                    progress.progress(min(0.3, i / len(frames)))

            status.text("评估清晰度...")
            sharp = []
            for f in frames:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                sharp.append(cv2.Laplacian(gray, cv2.CV_64F).var())

            win_len = max(2, int(0.5 * fps))
            step = max(1, win_len // 2)
            windows = []
            for start in range(0, len(motion) - win_len + 1, step):
                avg = np.mean(motion[start:start+win_len])
                windows.append((start, start+win_len, avg))

            if not windows:
                st.error("无法生成窗口")
            else:
                thresh = np.percentile([w[2] for w in windows], 75)
                cand = [(s, e) for s, e, sc in windows if sc >= thresh]

                merged = []
                for s, e in sorted(cand):
                    if not merged or s > merged[-1][1] + int(0.3 * fps):
                        merged.append([s, e])
                    else:
                        merged[-1][1] = max(merged[-1][1], e)

                results = []
                total = len(merged)
                for idx, (s, e) in enumerate(merged):
                    peak = s + np.argmax(motion[s:e])
                    if sharp[peak] < 35 or motion[peak] < 0.06:
                        continue
                    status.text(f"AI 描述片段 {len(results)+1}/{total}...")
                    progress.progress(0.3 + (idx / total) * 0.6)

                    # 模型推理
                    frame_rgb = cv2.cvtColor(frames[peak], cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    pixel_values = processor(pil_img, return_tensors="pt").pixel_values.to(device)
                    with torch.no_grad():
                        output_ids = model.generate(pixel_values, max_length=32, num_beams=4)
                    desc = tokenizer.decode(output_ids[0], skip_special_tokens=True)

                    results.append({
                        "start": round(s / fps, 2),
                        "end": round(e / fps, 2),
                        "score": round(motion[peak], 4),
                        "desc": desc
                    })

                progress.progress(1.0)
                status.text("分析完成")

                if results:
                    st.markdown("<h4>检测结果</h4>", unsafe_allow_html=True)
                    for i, r in enumerate(results):
                        st.markdown(
                            f"""
                            <div class="result-box">
                                <span class="time">{i+1}. {r['start']}s → {r['end']}s</span>
                                <span class="score">运动分数 {r['score']}</span>
                                <div class="desc">{r['desc']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("未检测到明显高光")

        time.sleep(0.2)
        try:
            os.unlink(video_path)
        except:
            pass
