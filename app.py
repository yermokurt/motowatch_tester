import gradio as gr
import pandas as pd
from detector import (
    yolov8_detect,
    download_sample_images,
    get_ocr_status,
    ADVANCED_OCR_AVAILABLE,
    OCR_AVAILABLE
)

try:
    from advanced_ocr import get_available_models
except ImportError:
    def get_available_models():
        return {}

download_sample_images()

ocr_status = get_ocr_status()
custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
}

.main-header {
    text-align: center;
    margin-bottom: 2rem;
    padding: 1rem;
}

.main-title {
    font-size: 2rem;
    font-weight: 600;
    color: #333;
    margin: 0;
}

.subtitle {
    color: #666;
    font-size: 1rem;
    margin: 0.5rem 0;
}

.status-info {
    font-size: 0.9rem;
    color: #888;
    margin: 0.5rem 0;
}

.section-gap {
    margin: 1.5rem 0;
}
"""

def toggle_sections(extract_text_checked, crop_checked):
    show_gallery = bool(extract_text_checked and crop_checked)
    show_ocr = bool(extract_text_checked)
    return (
        gr.update(visible=show_gallery),
        gr.update(visible=show_ocr),
    )

def get_ocr_status_text():
    if ADVANCED_OCR_AVAILABLE:
        return "Advanced OCR Available"
    elif OCR_AVAILABLE:
        return "Basic OCR Available"
    else:
        return "OCR Not Available"

with gr.Blocks(css=custom_css, title="AI Helmet Detection System") as demo:
    
    gr.HTML(f"""
        <div class="main-header">
            <h1 class="main-title">AI Helmet Detection System</h1>
            <p class="subtitle">Motorcyclist safety monitoring with license plate recognition</p>
            <p class="status-info">YOLOv11 • {get_ocr_status_text()} • Real-time Processing</p>
        </div>
    """)

    with gr.Tabs():
        with gr.TabItem("Detection"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Settings")
                    
                    input_image = gr.Image(
                        type="filepath", 
                        label="Upload Image", 
                        sources=["upload", "webcam"]
                    )
                    
                    with gr.Row():
                        image_size = gr.Slider(
                            minimum=320, maximum=1280, value=640, step=32,
                            label="Image Size"
                        )
                    
                    with gr.Row():
                        conf_threshold = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.4, step=0.05,
                            label="Confidence"
                        )
                        iou_threshold = gr.Slider(
                            minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                            label="IoU Threshold"
                        )
                    
                    show_stats = gr.Checkbox(
                        value=True, 
                        label="Show Statistics"
                    )
                    crop_plates = gr.Checkbox(
                        value=True, 
                        label="Extract License Plates"
                    )
                    
                    if ocr_status["any_available"]:
                        extract_text = gr.Checkbox(
                            value=False,
                            label="Enable OCR"
                        )
                        ocr_on_no_helmet = gr.Checkbox(
                            value=True,
                            label="Auto-OCR for No Helmet"
                        )

                        if ADVANCED_OCR_AVAILABLE:
                            models = get_available_models()
                            model_choices = [("Auto (Recommended)", "auto"), ("Basic EasyOCR", "basic")]
                            for key, info in models.items():
                                model_choices.append((info['name'], key))
                            selected_ocr_model = gr.Dropdown(
                                choices=model_choices,
                                value="auto",
                                label="OCR Model"
                            )
                        else:
                            selected_ocr_model = gr.State("basic")

                        gr.Markdown("*Note: OCR processing may increase detection time.*")
                    else:
                        extract_text = gr.Checkbox(
                            value=False,
                            label="OCR Not Available",
                            interactive=False
                        )
                        ocr_on_no_helmet = gr.Checkbox(
                            value=False,
                            label="Auto-OCR (Not Available)",
                            interactive=False
                        )
                        selected_ocr_model = gr.State("basic")
                    
                    with gr.Row():
                        submit_btn = gr.Button("Start Detection", variant="primary")
                        clear_btn = gr.Button("Clear")

                with gr.Column(scale=2):
                    gr.Markdown("### Results")
                    
                    output_image = gr.Image(
                        type="pil", 
                        label="Detection Results"
                    )
                    
                    with gr.Row():
                        output_table = gr.Dataframe(
                            headers=["Object", "Confidence", "Position", "Dimensions"],
                            label="Detection Details",
                            interactive=False
                        )
                        output_stats = gr.Textbox(
                            label="Statistics",
                            interactive=False,
                            lines=6
                        )

                    license_gallery = gr.Gallery(
                        label="License Plates",
                        columns=3,
                        visible=False
                    )

                    ocr_group = gr.Group(visible=False)
                    with ocr_group:
                        plate_text_output = gr.Textbox(
                            label="OCR Results",
                            lines=4,
                            interactive=False
                        )

                    download_file = gr.File(
                        label="Download Results (ZIP)",
                        interactive=False
                    )

        with gr.TabItem("Examples"):
            gr.Markdown("### Sample Images")
            gr.Markdown("Click any example to test the detection system:")
            
            gr.Examples(
                examples=[
                    ["sample_1.jpg"], 
                    ["sample_2.jpg"], 
                    ["sample_3.jpg"], 
                    ["sample_4.jpg"], 
                    ["sample_6.jpg"],
                    ["sample_7.jpg"],
                    ["sample_8.jpg"],
                ],
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
                cache_examples=True
            )

        with gr.TabItem("Info"):
            gr.Markdown("### System Information")
            
            gr.Markdown(f"""
            **AI Model:** YOLOv11  
            **Classes:** Helmet, No Helmet, License Plate  
            **OCR Status:** {get_ocr_status_text()}  
            **Features:** Detection, extraction, text recognition  
            
            **Privacy:** All processing is local. No data stored.  
            **Usage:** For demonstration and research purposes only.
            """)

    submit_btn.click(
        fn=yolov8_detect,
        inputs=[
            input_image,
            image_size,
            conf_threshold,
            iou_threshold,
            show_stats,
            gr.State(True),
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
    demo.launch(
        debug=True, 
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False
    )