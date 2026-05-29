import streamlit as st
import sys
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import numpy as np
import tempfile
import os
import time
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(page_title="视频高光提取", layout="centered")

# ---------- 加载本地模型 ----------
@st.cache_resource
def load_model():
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return processor, model, device

# ---------- 页面样式（仅修改此处）----------
st.markdown(
    """
    <style>
    /* 渐变浅蓝色背景 */
    .stApp {
        background: linear-gradient(135deg, #d4eaf7 0%, #b6d6f0 100%);
    }
    /* 标题样式 */
    h2 {
        color: #1a3e50 !important;
        font-weight: 600 !important;
    }
    hr {
        border-color: #3a7b8c !important;
        margin: 1rem 0 !important;
    }
    /* 按钮样式 */
    .stButton>button {
        background: #1a4d5f;
        color: white;
        border: none;
        border-radius: 40px;
        padding: 0.6rem 1.8rem;
        font-size: 0.9rem;
        font-weight: 500;
        width: 100%;
        transition: background 0.2s ease;
    }
    .stButton>button:hover {
        background: #0f3b48;
    }
    /* 结果框样式 */
    .result-box {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 20px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        border-left: 5px solid #1a4d5f;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    .time {
        font-size: 1rem;
        font-weight: 600;
        color: #1a4d5f;
    }
    .score {
        font-size: 0.8rem;
        color: #5f7f8c;
        margin-left: 1rem;
    }
    .desc {
        background: #eef4f8;
        padding: 0.7rem;
        border-radius: 16px;
        margin-top: 0.7rem;
        font-size: 0.85rem;
        color: #1e3e4a;
        line-height: 1.4;
    }
    /* 进度条颜色 */
    .stProgress > div > div {
        background-color: #1a4d5f;
    }
    /* 状态文字 */
    .stAlert {
        background-color: rgba(255,255,255,0.9);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h2 style='font-weight:400'>视频高光提取</h2><hr>", unsafe_allow_html=True)

# ---------- 上传 ----------
uploaded = st.file_uploader("", type=["mp4", "avi", "mov"], label_visibility="collapsed")

if uploaded is not None:
    # 保存临时视频
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    st.video(video_path)

    if st.button("开始分析"):
        processor, model, device = load_model()

        # 读取视频
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30

        frames = []
        progress = st.progress(0)
        status = st.empty()

        status.text("读取视频帧...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if len(frames) < 2:
            st.error("视频过短")
        else:
            # 运动分数
            status.text("计算运动分数...")
            motion = [0.0]
            for i in range(1, len(frames)):
                diff = cv2.absdiff(frames[i-1], frames[i])
                motion.append(np.mean(diff) / 255.0)
                if i % 50 == 0:
                    progress.progress(min(0.3, i / len(frames)))

            # 清晰度（可选过滤）
            status.text("评估清晰度...")
            sharp = []
            for f in frames:
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                sharp.append(cv2.Laplacian(gray, cv2.CV_64F).var())

            # 滑动窗口 (0.5秒)
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

                # 合并相邻
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

                    # BLIP 推理
                    frame_rgb = cv2.cvtColor(frames[peak], cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(frame_rgb)
                    inputs = processor(pil_img, return_tensors="pt").to(device)
                    with torch.no_grad():
                        out = model.generate(**inputs, max_length=45)
                    desc = processor.decode(out[0], skip_special_tokens=True)

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

        # 清理临时文件
        time.sleep(0.2)
        try:
            os.unlink(video_path)
        except:
            pass
