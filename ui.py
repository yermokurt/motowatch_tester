import gradio as gr
from detector import (
    yolov8_detect,
    yolov8_video_detect,
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
        try:
            download_sample_images()
        except Exception as e:
            print(f"Warning: Could not download sample images: {e}")
        
        self.ocr_status = get_ocr_status()
        self.custom_css = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
}
.landing-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 70vh;
    gap: 2rem;
    padding: 2rem;
}
.landing-title {
    font-size: 3.5rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 0.5rem;
    background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.landing-subtitle {
    font-size: 1.2rem;
    color: #64748b;
    text-align: center;
    max-width: 600px;
    margin-bottom: 3rem;
}
.selection-box {
    display: flex;
    gap: 2rem;
}
.nav-btn {
    padding: 2rem !important;
    font-size: 1.5rem !important;
    height: auto !important;
    width: 280px !important;
}
"""

    def get_ocr_status_text(self):
        if ADVANCED_OCR_AVAILABLE: return "Advanced OCR Available"
        if OCR_AVAILABLE: return "Basic OCR Available"
        return "OCR Not Available"

    def create_landing_page(self):
        with gr.Column(elem_classes="landing-container", visible=True) as landing_page:
            gr.HTML("""
                <h1 class="landing-title">AI Safety Guardian</h1>
                <p class="landing-subtitle">Intelligent Motorcyclist Compliance & License Plate Recognition System</p>
            """)
            with gr.Row(elem_classes="selection-box"):
                image_btn = gr.Button("🖼️ Check Image", elem_classes="nav-btn", variant="primary")
                video_btn = gr.Button("🎥 Check Video", elem_classes="nav-btn", variant="secondary")
        return landing_page, image_btn, video_btn

    def create_image_panel(self):
        with gr.Column(visible=False) as image_panel:
            gr.Markdown("# 🖼️ Image Detection")
            with gr.Row():
                with gr.Column(scale=1):
                    components = self.create_settings_panel()
                    components['back_btn'] = gr.Button("← Back to Home", size="sm")
                with gr.Column(scale=2):
                    results = self.create_results_panel()
        return image_panel, components, results

    def create_video_panel(self):
        components = {}
        with gr.Column(visible=False) as video_panel:
            gr.Markdown("# 🎥 Video Detection")
            with gr.Row():
                with gr.Column(scale=1):
                    components['video_input'] = gr.Video(label="Upload Video")
                    components['video_image_size'] = gr.Slider(320, 1280, 640, step=32, label="Process Size")
                    components['video_conf'] = gr.Slider(0.0, 1.0, 0.4, step=0.05, label="Confidence")
                    components['video_submit'] = gr.Button("Start Video Processing", variant="primary")
                    components['video_back'] = gr.Button("← Back to Home", size="sm")
                with gr.Column(scale=2):
                    components['video_output'] = gr.Video(label="Processed Result")
                    components['video_status'] = gr.Textbox(label="Status", interactive=False)
        return video_panel, components

    def create_settings_panel(self):
        components = {}
        components['input_image'] = gr.Image(type="filepath", label="Upload Image")
        components['image_size'] = gr.Slider(320, 1280, 640, step=32, label="Image Size")
        components['conf_threshold'] = gr.Slider(0.0, 1.0, 0.4, step=0.05, label="Confidence")
        components['iou_threshold'] = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="IoU Threshold")
        components['show_stats'] = gr.Checkbox(value=True, label="Show Statistics")
        components['crop_plates'] = gr.Checkbox(value=True, label="Extract License Plates")
        components['extract_text'] = gr.Checkbox(value=False, label="Enable OCR")
        components['ocr_on_no_helmet'] = gr.Checkbox(value=True, label="Auto-OCR for No Helmet")
        
        if ADVANCED_OCR_AVAILABLE:
            models = get_available_models()
            choices = [("Auto", "auto"), ("Basic", "basic")] + [(info['name'], k) for k, info in models.items()]
            components['selected_ocr_model'] = gr.Dropdown(choices=choices, value="auto", label="OCR Model")
        else:
            components['selected_ocr_model'] = gr.State("basic")
            
        with gr.Row():
            components['submit_btn'] = gr.Button("Start Detection", variant="primary")
            components['clear_btn'] = gr.Button("Clear")
        return components

    def create_results_panel(self):
        components = {}
        components['output_image'] = gr.Image(type="pil", label="Results")
        with gr.Row():
            components['output_table'] = gr.Dataframe(headers=["Object", "Confidence", "Position", "Dimensions"], label="Details")
            components['output_stats'] = gr.Textbox(label="Stats", lines=6)
        components['license_gallery'] = gr.Gallery(label="Extracted Plates", columns=3, height="auto")
        components['ocr_group'] = gr.Group()
        with components['ocr_group']:
            components['plate_text_output'] = gr.Textbox(label="OCR Text Log", lines=4)
        components['download_file'] = gr.File(label="Archive (Optional)", visible=False)
        return components

    def setup_event_handlers(self, landing_page, image_panel, video_panel, img_comps, vid_comps, image_btn, video_btn):
        # Navigation logic
        image_btn.click(lambda: (gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)), 
                        None, [landing_page, image_panel, video_panel])
        video_btn.click(lambda: (gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)), 
                        None, [landing_page, image_panel, video_panel])
        img_comps['back_btn'].click(lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)), 
                                    None, [landing_page, image_panel, video_panel])
        vid_comps['video_back'].click(lambda: (gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)), 
                                      None, [landing_page, image_panel, video_panel])

        img_comps['submit_btn'].click(
            fn=yolov8_detect,
            inputs=[
                img_comps['input_image'], img_comps['image_size'], img_comps['conf_threshold'], 
                img_comps['iou_threshold'], img_comps['show_stats'], gr.State(True), 
                img_comps['crop_plates'], img_comps['extract_text'], img_comps['ocr_on_no_helmet'], 
                img_comps['selected_ocr_model']
            ],
            outputs=[
                img_comps['output_image'], img_comps['output_table'], img_comps['output_stats'], 
                img_comps['license_gallery'], img_comps['download_file'], img_comps['plate_text_output']
            ]
        )

        vid_comps['video_submit'].click(
            fn=yolov8_video_detect,
            inputs=[vid_comps['video_input'], vid_comps['video_image_size'], vid_comps['video_conf']],
            outputs=[vid_comps['video_output'], vid_comps['video_status']]
        )