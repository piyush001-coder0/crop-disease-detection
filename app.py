import streamlit as st
from PIL import Image
from transformers import pipeline

st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Real-Time Crop Disease Detection")
st.write(
    "Upload a crop leaf image to identify possible diseases "
    "using an AI-based image classification model."
)

@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    )

uploaded_file = st.file_uploader(
    "📷 Upload Crop Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Crop Leaf",
        use_container_width=True
    )

    st.success("✅ Image uploaded successfully!")

    with st.spinner("🔍 AI is analyzing the leaf..."):
        model = load_model()
        results = model(image)

    best = results[0]

    label = best["label"]
    confidence = best["score"] * 100

    st.subheader("🔍 Detection Result")

    st.success(f"🌿 Prediction: {label}")

    st.metric(
        "AI Confidence",
        f"{confidence:.2f}%"
    )

    st.subheader("📊 Top Predictions")

    for result in results[:3]:
        percentage = result["score"] * 100

        st.write(
            f"**{result['label']}** — {percentage:.2f}%"
        )

        st.progress(min(result["score"], 1.0))

    st.subheader("🌾 Recommended Action")

    if "healthy" in label.lower():
        st.success(
            "The leaf appears healthy. Continue regular monitoring, "
            "proper watering, adequate sunlight and balanced nutrition."
        )
    else:
        st.warning(
            "Possible disease detected. Isolate affected plants if "
            "appropriate, remove severely affected leaves, maintain "
            "proper field hygiene and consult an agricultural expert "
            "before applying any pesticide or treatment."
        )

    st.info(
        "⚠️ This application provides an AI-based prediction for "
        "educational/demo purposes and is not a substitute for "
        "professional agricultural diagnosis."
    )