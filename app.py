import gradio as gr
import pandas as pd
from detector import (
    yolov8_detect,
    download_sample_images,
    get_ocr_status,
    ADVANCED_OCR_AVAILABLE,
    OCR_AVAILABLE
)

# Import advanced OCR functions if available
try:
    from advanced_ocr import get_available_models
except ImportError:
    def get_available_models():
        return {}

# Download sample images
download_sample_images()

# Get OCR status
ocr_status = get_ocr_status()

# ====== UI ======
custom_css = """
#title { text-align: center; }
#description { text-align: center; }
.footer { text-align: center; margin-top: 20px; color: #666; }
.important { font-weight: bold; color: red; }
.download-section { background-color: #f6f6f6; padding: 15px; border-radius: 10px; margin-top: 10px; }
.ocr-section { background-color: #eef7ff; padding: 15px; border-radius: 10px; margin-top: 10px; }
.card { background: white; border-radius: 16px; padding: 16px; box-shadow: 0 10px 25px rgba(0,0,0,0.06); }
"""

def toggle_sections(extract_text_checked, crop_checked):
    """Control visibility of Cropped Plates & OCR sections.
    Requirement: If user checks the OCR checkbox, show the cropped plates AND OCR text.
    Otherwise, hide both sections. Crops also require crop_checkbox True.
    """
    show_gallery = bool(extract_text_checked and crop_checked)
    show_ocr = bool(extract_text_checked)
    return (
        gr.update(visible=show_gallery),  # license_gallery
        gr.update(visible=show_ocr),      # ocr group container (textbox)
    )


with gr.Blocks(
    css=custom_css,
    title="YOLOv11 Motorcyclist Helmet Detection",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
) as demo:
    gr.HTML("<h1 id='title'>YOLOv11 Motorcyclist Helmet Detection</h1>")
    gr.HTML(
        f"""
    <div id='description'>
        <p>This app detects motorcyclists <strong>with</strong> / <strong>without</strong> helmets and can optionally read license plates.</p>
        <p><strong>OCR Status:</strong>
        {'✅ Advanced OCR Available' if ADVANCED_OCR_AVAILABLE else '🟡 Basic OCR Available' if OCR_AVAILABLE else '❌ OCR Not Available (install requirements)'}
        </p>
    </div>
    """
    )

    with gr.Tabs():
        with gr.TabItem("Inference"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Input Parameters")
                        input_image = gr.Image(
                            type="filepath", label="Input Image", sources=["upload", "webcam"]
                        )
                        with gr.Row():
                            image_size = gr.Slider(
                                minimum=320, maximum=1280, value=640, step=32, label="Image Size"
                            )
                            conf_threshold = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.4, step=0.05, label="Confidence Threshold"
                            )
                        with gr.Row():
                            iou_threshold = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.5, step=0.05, label="IOU Threshold"
                            )
                            show_stats = gr.Checkbox(value=True, label="Show Stats on Image")

                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Options")
                        crop_plates = gr.Checkbox(value=True, label="Enable License Plate Cropping")

                        if ocr_status["any_available"]:
                            extract_text = gr.Checkbox(
                                value=False,
                                label="Enable OCR (Show Cropped Plates & Text)",
                                info="When enabled: shows cropped plates + runs OCR",
                            )
                            ocr_on_no_helmet = gr.Checkbox(
                                value=True,
                                label="🚨 Auto-OCR when No Helmet Detected",
                            )

                            if ADVANCED_OCR_AVAILABLE:
                                models = get_available_models()
                                model_choices = [("Auto (Recommended)", "auto"), ("Basic EasyOCR", "basic")]
                                for key, info in models.items():
                                    model_choices.append((info["name"], key))
                                selected_ocr_model = gr.Dropdown(
                                    choices=model_choices,
                                    value="auto",
                                    label="OCR Model Selection",
                                    info="Choose OCR model (Advanced models may require setup)",
                                )
                            else:
                                selected_ocr_model = gr.State("basic")

                            gr.Markdown("*Note: OCR may increase processing time*")
                        else:
                            extract_text = gr.Checkbox(
                                value=False,
                                label="OCR Not Available",
                                interactive=False,
                            )
                            ocr_on_no_helmet = gr.Checkbox(
                                value=False,
                                label="🚨 Auto-OCR when No Helmet (Not Available)",
                                interactive=False,
                            )
                            selected_ocr_model = gr.State("basic")

                        with gr.Row():
                            submit_btn = gr.Button("🔍 Detect", variant="primary")
                            clear_btn = gr.Button("🧹 Clear")

                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["card"]):
                        gr.Markdown("### Output")
                        output_image = gr.Image(type="pil", label="Annotated Image")
                        output_table = gr.Dataframe(
                            headers=["Object", "Confidence", "Position", "Dimensions"],
                            label="Detection Details",
                            interactive=False,
                        )
                        output_stats = gr.Textbox(
                            label="Detection Summary", interactive=False, lines=6
                        )

                    # ---- Cropped plates & OCR (conditionally visible) ----
                    license_gallery = gr.Gallery(
                        label="Extracted License Plates",
                        show_label=True,
                        elem_id="license_gallery",
                        columns=3,
                        rows=2,
                        object_fit="contain",
                        height="auto",
                        visible=False,  # hidden until OCR checkbox is enabled
                    )

                    ocr_group = gr.Group(elem_classes=["ocr-section"], visible=False)
                    with ocr_group:
                        gr.Markdown("### License Plate Text Recognition")
                        plate_text_output = gr.Textbox(
                            label="Extracted Text",
                            placeholder="License plate text will appear here when OCR is enabled",
                            lines=4,
                            interactive=False,
                        )

                    with gr.Group(elem_classes=["download-section", "card"]):
                        gr.Markdown("### Download Results")
                        download_file = gr.File(
                            label="Download Complete Results (ZIP)",
                            interactive=False,
                            visible=True,
                        )
                        gr.Markdown(
                            "*ZIP contains: annotated image, cropped plates (if any), and a CSV report with OCR results*"
                        )

        with gr.TabItem("Examples"):
            gr.Markdown("### Example Images")
            gr.Examples(
                examples=[["sample_1.jpg"], ["sample_2.jpg"], ["sample_3.jpg"], ["sample_4.jpg"], ["sample_5.jpg"]],
                inputs=input_image,
                outputs=[
                    output_image,
                    output_table,
                    output_stats,
                    license_gallery,
                    download_file,
                    plate_text_output,
                ],
                fn=lambda img: yolov8_detect(
                    img, 640, 0.4, 0.5, True, True, True, False
                ),
                cache_examples=True,
            )

    gr.HTML(
        """
    <div class='footer'>
        <p>Built with Gradio and Ultralytics YOLO</p>
        <p><strong>License Plate Privacy:</strong> Extracted plate images & text are for demo purposes only.</p>
        <p><strong>Requirements for OCR:</strong> torch, transformers, easyocr, opencv-python</p>
    </div>
    """
    )

    # ===== Wire events =====
    # 1) Main click
    submit_btn.click(
        fn=yolov8_detect,
        inputs=[
            input_image,
            image_size,
            conf_threshold,
            iou_threshold,
            show_stats,
            gr.State(True),  # show_confidence placeholder
            crop_plates,
            extract_text,
            ocr_on_no_helmet,
            selected_ocr_model,
        ],
        outputs=[
            output_image,
            output_table,
            output_stats,
            license_gallery,
            download_file,
            plate_text_output,
        ],
    )

    # 2) Clear
    clear_btn.click(
        fn=lambda: [None, None, None, None, None, None],
        inputs=[],
        outputs=[
            input_image,
            output_image,
            output_table,
            output_stats,
            license_gallery,
            download_file,
            plate_text_output,
        ],
    )

    # 3) Toggle visibility when user toggles OCR or Crop checkboxes
    extract_text.change(
        fn=toggle_sections,
        inputs=[extract_text, crop_plates],
        outputs=[license_gallery, ocr_group],
    )
    crop_plates.change(
        fn=toggle_sections,
        inputs=[extract_text, crop_plates],
        outputs=[license_gallery, ocr_group],
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)