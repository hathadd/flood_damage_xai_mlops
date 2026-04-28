from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError

DEFAULT_API_BASE_URL = "http://127.0.0.1:8000"
PROJECT_TITLE = "Flood Damage Classification System"
PROJECT_SUBTITLE = "Bi-temporal satellite damage classification using Siamese Deep Learning + XAI + MLOps"
FOOTER_NOTE = "Final selected model: Run B - Siamese ResNet18 Regularized"


def _build_endpoint(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}{endpoint}"


def _load_uploaded_image(uploaded_file: Any, field_name: str) -> Image.Image:
    try:
        return Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"{field_name} is not a valid image.") from exc


def _call_health(api_base_url: str) -> dict[str, Any]:
    response = requests.get(_build_endpoint(api_base_url, "/health"), timeout=20)
    response.raise_for_status()
    return response.json()


def _call_predict_scene(
    api_base_url: str,
    pre_upload: Any,
    post_upload: Any,
    json_upload: Any,
    context_ratio: float,
    min_crop_size: int,
    save_annotated: bool,
) -> dict[str, Any]:
    pre_upload.seek(0)
    post_upload.seek(0)
    json_upload.seek(0)

    files = {
        "pre_image": (pre_upload.name, pre_upload.getvalue(), pre_upload.type or "image/png"),
        "post_image": (post_upload.name, post_upload.getvalue(), post_upload.type or "image/png"),
        "post_json": (json_upload.name, json_upload.getvalue(), json_upload.type or "application/json"),
    }
    data = {
        "context_ratio": str(context_ratio),
        "min_crop_size": str(min_crop_size),
        "save_annotated": str(save_annotated).lower(),
    }
    response = requests.post(_build_endpoint(api_base_url, "/predict-scene"), files=files, data=data, timeout=120)
    response.raise_for_status()
    return response.json()


def _call_explain_building(
    api_base_url: str,
    pre_upload: Any,
    post_upload: Any,
    json_upload: Any,
    building_index: int,
    context_ratio: float,
    min_crop_size: int,
) -> dict[str, Any]:
    pre_upload.seek(0)
    post_upload.seek(0)
    json_upload.seek(0)

    files = {
        "pre_image": (pre_upload.name, pre_upload.getvalue(), pre_upload.type or "image/png"),
        "post_image": (post_upload.name, post_upload.getvalue(), post_upload.type or "image/png"),
        "post_json": (json_upload.name, json_upload.getvalue(), json_upload.type or "application/json"),
    }
    data = {
        "building_index": str(building_index),
        "context_ratio": str(context_ratio),
        "min_crop_size": str(min_crop_size),
    }
    response = requests.post(_build_endpoint(api_base_url, "/explain-building"), files=files, data=data, timeout=120)
    response.raise_for_status()
    return response.json()


def _show_api_error(prefix: str, exc: Exception) -> None:
    if isinstance(exc, requests.exceptions.ConnectionError):
        st.error(f"{prefix}: FastAPI backend is unavailable. Start the API service and verify the URL.")
        return
    if isinstance(exc, requests.exceptions.Timeout):
        st.error(f"{prefix}: request timed out.")
        return
    if isinstance(exc, requests.exceptions.HTTPError):
        detail = exc.response.text if exc.response is not None else str(exc)
        st.error(f"{prefix}: backend returned an error. {detail}")
        return
    if isinstance(exc, ValueError):
        st.error(f"{prefix}: {exc}")
        return
    st.error(f"{prefix}: unexpected error: {exc}")


def _render_local_image(path_text: str | None, caption: str) -> None:
    if not path_text:
        st.info(f"{caption}: not available")
        return
    path = Path(path_text)
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"{caption} saved by backend but not accessible from Streamlit path: {path_text}")


def _render_prediction_card(label: str, confidence: float, class_id: int | None = None) -> None:
    extra = f"class id = {class_id} | confidence = {confidence:.4f}" if class_id is not None else f"confidence = {confidence:.4f}"
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #0f766e 0%, #155e75 100%);
            padding: 1.2rem 1.4rem;
            border-radius: 18px;
            color: white;
            box-shadow: 0 12px 28px rgba(15, 118, 110, 0.18);
            margin-bottom: 1rem;
        ">
            <div style="font-size: 0.9rem; opacity: 0.88;">Prediction result</div>
            <div style="font-size: 1.9rem; font-weight: 700; margin-top: 0.15rem;">{label}</div>
            <div style="margin-top: 0.35rem; font-size: 1rem;">{extra}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_probability_breakdown(probabilities: dict[str, Any]) -> None:
    st.markdown("### Probability Breakdown")
    for class_name, value in probabilities.items():
        prob = float(value)
        st.markdown(f"**{class_name}**")
        st.progress(max(0.0, min(1.0, prob)))
        st.caption(f"{prob:.4f}")

    probability_df = pd.DataFrame(
        {
            "Class": list(probabilities.keys()),
            "Probability": [float(value) for value in probabilities.values()],
        }
    ).set_index("Class")
    st.bar_chart(probability_df)


def _render_scene_table(predictions: list[dict[str, Any]]) -> pd.DataFrame:
    table_df = pd.DataFrame(predictions)
    if table_df.empty:
        st.warning("No building predictions were returned.")
        return table_df

    display_columns = [column for column in ["building_index", "true_label", "predicted_label", "confidence"] if column in table_df.columns]
    st.dataframe(table_df[display_columns], use_container_width=True, hide_index=True)
    return table_df


def _ensure_session_defaults() -> None:
    st.session_state.setdefault("scene_response", None)
    st.session_state.setdefault("scene_predictions", [])
    st.session_state.setdefault("explanation_response", None)


def main() -> None:
    st.set_page_config(
        page_title="Flood Damage Classification Demo",
        page_icon="???",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _ensure_session_defaults()

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }
        .hero-box {
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            border: 1px solid #d9e3ee;
            padding: 1.35rem 1.5rem;
            border-radius: 18px;
            margin-bottom: 1rem;
        }
        .info-box {
            background: #f8fafc;
            border-left: 5px solid #0f766e;
            padding: 1rem 1rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .footer-box {
            margin-top: 2rem;
            padding: 1rem 1.2rem;
            border-radius: 14px;
            background: #f1f5f9;
            border: 1px solid #dbe4ee;
            color: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.title("Demo Settings")
    api_base_url = st.sidebar.text_input("API Base URL", value=DEFAULT_API_BASE_URL)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Info")
    st.sidebar.markdown("**Final Selected Model**")
    st.sidebar.markdown("Run B - Siamese ResNet18 Regularized")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Project Summary")
    st.sidebar.markdown(
        "This system performs building-level flood damage assessment using bi-temporal satellite imagery from the xBD flooding dataset."
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("Service Health")
    if st.sidebar.button("Check Health", use_container_width=True):
        try:
            health_payload = _call_health(api_base_url)
        except Exception as exc:
            _show_api_error("Health check failed", exc)
        else:
            st.sidebar.success("Backend reachable")
            st.sidebar.write(f"**Status:** {health_payload.get('status', 'unknown')}")
            st.sidebar.write(f"**Device:** {health_payload.get('device', 'unknown')}")
            st.sidebar.write(f"**Model loaded:** {health_payload.get('model_loaded', 'unknown')}")
            st.sidebar.write(f"**Checkpoint exists:** {health_payload.get('checkpoint_exists', 'unknown')}")

    st.title(PROJECT_TITLE)
    st.markdown(f"### {PROJECT_SUBTITLE}")
    st.markdown(
        """
        <div class="hero-box">
            <strong>Academic Demonstration Interface</strong><br/>
            This interface demonstrates building-level flood damage classification using pre-disaster and post-disaster satellite images together with xBD post-disaster building polygons.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="info-box">
            Workflow: upload the PRE image, POST image, and xBD JSON file, run scene prediction, then select one building to generate a real Grad-CAM explanation.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Step 1 - Upload Scene Inputs", expanded=True):
        upload_col1, upload_col2, upload_col3 = st.columns(3)
        with upload_col1:
            pre_upload = st.file_uploader("PRE satellite image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="pre_upload")
        with upload_col2:
            post_upload = st.file_uploader("POST satellite image", type=["png", "jpg", "jpeg", "tif", "tiff"], key="post_upload")
        with upload_col3:
            json_upload = st.file_uploader("xBD post-disaster JSON", type=["json"], key="json_upload")

        controls_col1, controls_col2, controls_col3 = st.columns(3)
        with controls_col1:
            context_ratio = st.number_input("Context ratio", min_value=0.0, value=0.25, step=0.05)
        with controls_col2:
            min_crop_size = st.number_input("Min crop size", min_value=16, value=64, step=16)
        with controls_col3:
            save_annotated = st.toggle("Save annotated scene", value=True)

    pre_image = None
    post_image = None
    if pre_upload is not None:
        try:
            pre_image = _load_uploaded_image(pre_upload, "PRE image")
        except ValueError as exc:
            st.error(str(exc))
    if post_upload is not None:
        try:
            post_image = _load_uploaded_image(post_upload, "POST image")
        except ValueError as exc:
            st.error(str(exc))

    if pre_image is not None or post_image is not None:
        preview_col1, preview_col2 = st.columns(2)
        with preview_col1:
            if pre_image is not None:
                st.image(pre_image, caption="Pre-disaster scene", use_container_width=True)
        with preview_col2:
            if post_image is not None:
                st.image(post_image, caption="Post-disaster scene", use_container_width=True)

    st.markdown("## Step 2 - Predict Scene")
    if st.button("Predict Scene", type="primary", use_container_width=True):
        if pre_upload is None or post_upload is None or json_upload is None:
            st.warning("Please upload the PRE image, POST image, and xBD JSON file before running scene prediction.")
        else:
            with st.spinner("Running scene-level inference through FastAPI..."):
                try:
                    scene_response = _call_predict_scene(
                        api_base_url=api_base_url,
                        pre_upload=pre_upload,
                        post_upload=post_upload,
                        json_upload=json_upload,
                        context_ratio=float(context_ratio),
                        min_crop_size=int(min_crop_size),
                        save_annotated=bool(save_annotated),
                    )
                except Exception as exc:
                    _show_api_error("Scene prediction failed", exc)
                else:
                    st.session_state["scene_response"] = scene_response
                    st.session_state["scene_predictions"] = scene_response.get("predictions", [])
                    st.session_state["explanation_response"] = None
                    st.success("Scene prediction completed successfully.")

    scene_response = st.session_state.get("scene_response")
    scene_predictions = st.session_state.get("scene_predictions", [])

    if scene_response:
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        metrics_col1.metric("Buildings detected", int(scene_response.get("total_buildings", 0)))
        metrics_col2.metric("Model", scene_response.get("model_name", "unknown"))
        metrics_col3.metric("Version", scene_response.get("model_version", "unknown"))

        annotated_path = scene_response.get("annotated_image_path")
        if annotated_path:
            st.markdown("### Annotated Post-Scene")
            _render_local_image(annotated_path, "Annotated post-scene prediction")

        st.markdown("### Building Predictions")
        scene_table_df = _render_scene_table(scene_predictions)
    else:
        scene_table_df = pd.DataFrame()

    st.markdown("## Step 3 - Select Building")
    if not scene_predictions:
        st.info("Run scene prediction first to populate the building list.")
        selected_building_index = None
    else:
        building_options = []
        for prediction in scene_predictions:
            building_index = int(prediction.get("building_index", -1))
            predicted_label = prediction.get("predicted_label", "unknown")
            confidence = float(prediction.get("confidence", 0.0))
            building_options.append((building_index, f"Building {building_index} - {predicted_label} ({confidence:.2f})"))

        selected_building_index = st.selectbox(
            "Choose one building for Grad-CAM explanation",
            options=[option[0] for option in building_options],
            format_func=lambda value: next(label for index, label in building_options if index == value),
        )

    st.markdown("## Step 4 - Explain Building (Grad-CAM)")
    if st.button("Explain Building (Grad-CAM)", use_container_width=True):
        if pre_upload is None or post_upload is None or json_upload is None:
            st.warning("Please upload all three inputs before requesting Grad-CAM explanation.")
        elif selected_building_index is None:
            st.warning("Please run scene prediction and choose a building first.")
        else:
            with st.spinner("Generating building-specific Grad-CAM explanation..."):
                try:
                    explanation_response = _call_explain_building(
                        api_base_url=api_base_url,
                        pre_upload=pre_upload,
                        post_upload=post_upload,
                        json_upload=json_upload,
                        building_index=int(selected_building_index),
                        context_ratio=float(context_ratio),
                        min_crop_size=int(min_crop_size),
                    )
                except Exception as exc:
                    _show_api_error("Building explanation failed", exc)
                else:
                    st.session_state["explanation_response"] = explanation_response
                    st.success("Grad-CAM explanation generated successfully.")

    explanation_response = st.session_state.get("explanation_response")
    if explanation_response:
        _render_prediction_card(
            label=str(explanation_response.get("predicted_label", "unknown")),
            confidence=float(explanation_response.get("confidence", 0.0)),
        )

        info_col1, info_col2, info_col3 = st.columns(3)
        info_col1.metric("Building index", int(explanation_response.get("building_index", -1)))
        info_col2.metric("True label", str(explanation_response.get("true_label", "unknown")))
        info_col3.metric("Building UID", str(explanation_response.get("building_uid", "unknown")))

        _render_probability_breakdown(explanation_response.get("probabilities", {}))

        st.markdown("### Building-specific Explanation")
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            _render_local_image(explanation_response.get("pre_crop_path"), "Pre-disaster crop")
        with row1_col2:
            _render_local_image(explanation_response.get("post_crop_path"), "Post-disaster crop")

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            _render_local_image(explanation_response.get("pre_gradcam_path"), "Pre-disaster Grad-CAM")
        with row2_col2:
            _render_local_image(explanation_response.get("post_gradcam_path"), "Post-disaster Grad-CAM")

    st.markdown(
        f"""
        <div class="footer-box">
            {FOOTER_NOTE}
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

