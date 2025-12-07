import gradio as gr
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

class UIComponents:
    def __init__(self):
        # Wrap download_sample_images in try-except to prevent app crash
        try:
            download_sample_images()
        except Exception as e:
            print(f"Warning: Could not download sample images: {e}")
            print("Continuing without sample images...")
        
        self.ocr_status = get_ocr_status()
        self.custom_css = """
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

    def toggle_sections(self, extract_text_checked, crop_checked):
        show_gallery = bool(extract_text_checked and crop_checked)
        show_ocr = bool(extract_text_checked)
        return (
            gr.update(visible=show_gallery),
            gr.update(visible=show_ocr),
        )

    def get_ocr_status_text(self):
        if ADVANCED_OCR_AVAILABLE:
            return "Advanced OCR Available"
        elif OCR_AVAILABLE:
            return "Basic OCR Available"
        else:
            return "OCR Not Available"

    def create_header(self):
        return gr.HTML(f"""
            <div class="main-header">
                <h1 class="main-title">AI Helmet Detection System</h1>
                <p class="subtitle">Motorcyclist safety monitoring with license plate recognition</p>
                <p class="status-info">YOLOv11 • {self.get_ocr_status_text()} • Real-time Processing</p>
            </div>
        """)

    def create_settings_panel(self):
        components = {}
        
        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            
            components['input_image'] = gr.Image(
                type="filepath", 
                label="Upload Image", 
                sources=["upload", "webcam"]
            )
            
            with gr.Row():
                components['image_size'] = gr.Slider(
                    minimum=320, maximum=1280, value=640, step=32,
                    label="Image Size"
                )
            
            with gr.Row():
                components['conf_threshold'] = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.4, step=0.05,
                    label="Confidence"
                )
                components['iou_threshold'] = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                    label="IoU Threshold"
                )
            
            components['show_stats'] = gr.Checkbox(
                value=True, 
                label="Show Statistics"
            )
            components['crop_plates'] = gr.Checkbox(
                value=True, 
                label="Extract License Plates"
            )
            
            if self.ocr_status["any_available"]:
                components['extract_text'] = gr.Checkbox(
                    value=False,
                    label="Enable OCR"
                )
                components['ocr_on_no_helmet'] = gr.Checkbox(
                    value=True,
                    label="Auto-OCR for No Helmet"
                )

                if ADVANCED_OCR_AVAILABLE:
                    models = get_available_models()
                    model_choices = [("Auto (Recommended)", "auto"), ("Basic EasyOCR", "basic")]
                    for key, info in models.items():
                        model_choices.append((info['name'], key))
                    components['selected_ocr_model'] = gr.Dropdown(
                        choices=model_choices,
                        value="auto",
                        label="OCR Model"
                    )
                else:
                    components['selected_ocr_model'] = gr.State("basic")

                gr.Markdown("*Note: OCR processing may increase detection time.*")
            else:
                components['extract_text'] = gr.Checkbox(
                    value=False,
                    label="OCR Not Available",
                    interactive=False
                )
                components['ocr_on_no_helmet'] = gr.Checkbox(
                    value=False,
                    label="Auto-OCR (Not Available)",
                    interactive=False
                )
                components['selected_ocr_model'] = gr.State("basic")
            
            with gr.Row():
                components['submit_btn'] = gr.Button("Start Detection", variant="primary")
                components['clear_btn'] = gr.Button("Clear")
        
        return components

    def create_results_panel(self):
        components = {}
        
        with gr.Column(scale=2):
            gr.Markdown("### Results")
            
            components['output_image'] = gr.Image(
                type="pil", 
                label="Detection Results"
            )
            
            with gr.Row():
                components['output_table'] = gr.Dataframe(
                    headers=["Object", "Confidence", "Position", "Dimensions"],
                    label="Detection Details",
                    interactive=False
                )
                components['output_stats'] = gr.Textbox(
                    label="Statistics",
                    interactive=False,
                    lines=6
                )

            components['license_gallery'] = gr.Gallery(
                label="License Plates",
                columns=3,
                visible=False
            )

            components['ocr_group'] = gr.Group(visible=False)
            with components['ocr_group']:
                components['plate_text_output'] = gr.Textbox(
                    label="OCR Results",
                    lines=4,
                    interactive=False
                )

            components['download_file'] = gr.File(
                label="Download Results (ZIP)",
                interactive=False
            )
        
        return components

    def create_examples_tab(self, input_image, output_components):
        with gr.TabItem("Examples"):
            gr.Markdown("### Sample Images")
            gr.Markdown("Click any example to test the detection system:")
            
            gr.Examples(
                examples=[
                    ["Sample-Image-1.jpg"], 
                    ["Sample-Image-2.jpg"], 
                    ["Sample-Image-3.jpg"], 
                    ["Sample-Image-4.jpg"], 
                    ["Sample-Image-6.jpg"],
                    ["Sample-Image-7.jpg"],
                    ["Sample-Image-8.jpg"],
                ],
                inputs=input_image,
                outputs=[
                    output_components['output_image'],
                    output_components['output_table'],
                    output_components['output_stats'],
                    output_components['license_gallery'],
                    output_components['download_file'],
                    output_components['plate_text_output'],
                ],
                fn=lambda img: yolov8_detect(
                    img, 640, 0.4, 0.5, True, True, True, False
                ),
                cache_examples=True
            )

    def create_info_tab(self):
        with gr.TabItem("Info"):
            gr.Markdown("### System Information")
            
            gr.Markdown(f"""
            **AI Model:** YOLOv11  
            **Classes:** Helmet, No Helmet, License Plate  
            **OCR Status:** {self.get_ocr_status_text()}  
            **Features:** Detection, extraction, text recognition  
            
            **Privacy:** All processing is local. No data stored.  
            **Usage:** For demonstration and research purposes only.
            """)

    def setup_event_handlers(self, settings_components, results_components):
        settings_components['submit_btn'].click(
            fn=yolov8_detect,
            inputs=[
                settings_components['input_image'],
                settings_components['image_size'],
                settings_components['conf_threshold'],
                settings_components['iou_threshold'],
                settings_components['show_stats'],
                gr.State(True),
                settings_components['crop_plates'],
                settings_components['extract_text'],
                settings_components['ocr_on_no_helmet'],
                settings_components['selected_ocr_model'],
            ],
            outputs=[
                results_components['output_image'],
                results_components['output_table'],
                results_components['output_stats'],
                results_components['license_gallery'],
                results_components['download_file'],
                results_components['plate_text_output'],
            ],
        )

        settings_components['clear_btn'].click(
            fn=lambda: [None, None, None, None, None, None],
            inputs=[],
            outputs=[
                settings_components['input_image'],
                results_components['output_image'],
                results_components['output_table'],
                results_components['output_stats'],
                results_components['license_gallery'],
                results_components['download_file'],
                results_components['plate_text_output'],
            ],
        )

        settings_components['extract_text'].change(
            fn=self.toggle_sections,
            inputs=[settings_components['extract_text'], settings_components['crop_plates']],
            outputs=[results_components['license_gallery'], results_components['ocr_group']],
        )
        
        settings_components['crop_plates'].change(
            fn=self.toggle_sections,
            inputs=[settings_components['extract_text'], settings_components['crop_plates']],
            outputs=[results_components['license_gallery'], results_components['ocr_group']],
        )