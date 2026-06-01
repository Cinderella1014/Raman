import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks, detrend
import requests
import io
from datetime import datetime
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
import warnings
warnings.filterwarnings("ignore")

# -------------------------- 全局配置 --------------------------
st.set_page_config(
    page_title="拉曼光谱智能分析平台",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
.main {background-color: #f8fafc;}
h1 {color: #1e3a8a; font-weight: 700;}
h2, h3 {color: #1e40af;}
.stCard {
    border-radius: 12px;
    padding: 1.5rem;
    background: white;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    margin-bottom: 1rem;
}
.stChatMessage {border-radius: 12px; padding: 1rem; margin-bottom: 0.5rem;}
.stChatMessage.user {background-color: #dbeafe;}
.stChatMessage.assistant {background-color: #eff6ff;}
.stButton>button {
    background-color: #1e40af;
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    border: none;
}
.stButton>button:hover {background-color: #1e3a8a;}
</style>
""", unsafe_allow_html=True)

# -------------------------- 工具函数 --------------------------
def call_glm(api_key, prompt):
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "glm-4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        return response.json()["choices"][0]["message"]["content"]
    except:
        return "AI服务暂时不可用，请检查网络或API Key"

def generate_pdf_report(raman_shift, intensity, peak_list, chat_history):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # 标题
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(width/2, height - 20*mm, "拉曼光谱分析报告")
    c.setFont("Helvetica", 10)
    c.drawCentredString(width/2, height - 25*mm, f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 光谱基本信息
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20*mm, height - 35*mm, "一、光谱基本信息")
    c.setFont("Helvetica", 10)
    info_text = [
        f"波数范围：{raman_shift.min():.1f} ~ {raman_shift.max():.1f} cm⁻¹",
        f"数据点数：{len(raman_shift)}",
        f"特征峰位：{', '.join(peak_list) if peak_list else '无'}"
    ]
    y = height - 40*mm
    for line in info_text:
        c.drawString(20*mm, y, line)
        y -= 5*mm

    # 对话内容
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20*mm, y - 5*mm, "二、AI分析对话")
    y -= 10*mm
    c.setFont("Helvetica", 9)
    for msg in chat_history:
        role = "用户" if msg["role"] == "user" else "AI助手"
        text = f"{role}：{msg['content']}"
        # 自动换行
        lines = []
        while len(text) > 90:
            lines.append(text[:90])
            text = text[90:]
        lines.append(text)
        for line in lines:
            if y < 20*mm:
                c.showPage()
                y = height - 20*mm
                c.setFont("Helvetica", 9)
            c.drawString(20*mm, y, line)
            y -= 4*mm

    c.save()
    buffer.seek(0)
    return buffer

# -------------------------- 侧边栏 --------------------------
with st.sidebar:
    st.markdown("## 🔬 拉曼AI平台")
    st.divider()
    api_key = st.text_input(
        "智谱API Key",
        value="5417d3291fb74c9999dfc36369652f26.JGKE2GhndSKALPZy",
        type="password",
        help="可直接使用预设Key"
    )
    st.divider()
    st.markdown("### ⚙️ 光谱预处理")
    do_smooth = st.checkbox("✅ 光谱平滑降噪", value=True)
    do_baseline = st.checkbox("✅ 基线校正", value=True)
    do_findpeak = st.checkbox("✅ 自动识别特征峰", value=True)
    st.divider()
    st.info("支持格式：两列数据的TXT文件（波数 强度）")

# -------------------------- 主页面 --------------------------
st.title("🔬 拉曼光谱智能分析交互平台")
st.markdown("#### 专业功能：光谱上传 | 预处理 | 特征峰提取 | 大模型专业问答 | PDF报告导出")
st.divider()

# 初始化session
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "peak_list" not in st.session_state:
    st.session_state.peak_list = []
if "raman_shift" not in st.session_state:
    st.session_state.raman_shift = None
if "intensity" not in st.session_state:
    st.session_state.intensity = None

# 上传区域
with st.container():
    st.markdown("### 📤 上传拉曼光谱数据")
    upload_file = st.file_uploader("请上传光谱文件（.txt）", type="txt")

    if upload_file:
        raw_data = np.loadtxt(upload_file)
        raman_shift = raw_data[:, 0]
        intensity = raw_data[:, 1]

        with st.spinner("正在预处理光谱数据..."):
            if do_smooth:
                intensity = savgol_filter(intensity, window_length=11, polyorder=3)
            if do_baseline:
                intensity = detrend(intensity)

            # 绘图
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(raman_shift, intensity, color="#1e40af", linewidth=1.5, label="处理后光谱")
            ax.set_xlabel("拉曼位移 Raman Shift (cm⁻¹)", fontsize=11, fontweight='bold')
            ax.set_ylabel("光强 Intensity", fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle="--")

            peak_indices = []
            if do_findpeak:
                peak_indices, _ = find_peaks(
                    intensity,
                    height=np.max(intensity) * 0.05,
                    distance=15
                )
                ax.scatter(raman_shift[peak_indices], intensity[peak_indices], color="#dc2626", s=30, label="特征峰")
            ax.legend()
            st.pyplot(fig)

            # 保存到session
            st.session_state.raman_shift = raman_shift
            st.session_state.intensity = intensity
            st.session_state.peak_list = [f"{int(raman_shift[idx])}" for idx in peak_indices]

            if st.session_state.peak_list:
                st.success(f"✅ 检测到特征峰位(cm⁻¹)：{', '.join(st.session_state.peak_list)}")
            else:
                st.warning("⚠️ 当前光谱未识别出明显特征峰")

# PDF导出按钮（只有上传数据后才显示）
if st.session_state.raman_shift is not None:
    st.divider()
    st.markdown("### 📄 导出分析报告")
    pdf_buffer = generate_pdf_report(
        st.session_state.raman_shift,
        st.session_state.intensity,
        st.session_state.peak_list,
        st.session_state.chat_history
    )
    st.download_button(
        label="📥 下载PDF报告",
        data=pdf_buffer,
        file_name=f"拉曼光谱分析报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf"
    )

st.divider()

# -------------------------- AI对话 --------------------------
st.markdown("### 🧠 拉曼光谱智能问答助手")
st.markdown("基于智谱GLM-4-Flash大模型，为您提供专业的拉曼光谱分析与解读")

# 渲染历史对话
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 聊天输入框
user_input = st.chat_input("请提问：分析物质成分、解读峰位、解释官能团、实验建议等...")

if user_input and api_key.strip():
    system_prompt = f"""
你是一名资深拉曼光谱分析专家，请基于以下信息回答用户问题：
当前光谱特征峰位(cm⁻¹)：{', '.join(st.session_state.peak_list) if st.session_state.peak_list else '暂无'}
用户提问：{user_input}

回答要求：
1. 结合拉曼光谱、物质结构、官能团振动知识专业作答；
2. 语言通俗易懂，区分学术解释和实用分析；
3. 若峰位信息不足，如实说明并给出参考建议。
"""
    with st.spinner("AI正在分析光谱与问题..."):
        ai_response = call_glm(api_key, system_prompt)

    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
    st.rerun()

elif user_input and not api_key.strip():
    st.error("❌ 请检查侧边栏API Key是否填写完整！")