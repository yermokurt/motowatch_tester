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

# Enhanced Modern CSS
custom_css = """
/* Global Styles */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}

/* Main container styling */
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

/* Header styles */
.header-container {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 2rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.1);
}

.main-title {
    background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 3rem;
    font-weight: 800;
    text-align: center;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    animation: gradient-shift 3s ease-in-out infinite alternate;
}

@keyframes gradient-shift {
    0% { filter: hue-rotate(0deg); }
    100% { filter: hue-rotate(20deg); }
}

.subtitle {
    text-align: center;
    color: rgba(255, 255, 255, 0.9);
    font-size: 1.2rem;
    margin-top: 1rem;
    font-weight: 300;
}

.status-badge {
    display: inline-block;
    padding: 0.5rem 1rem;
    border-radius: 25px;
    font-weight: 600;
    font-size: 0.9rem;
    margin: 0.5rem;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
}

.status-success {
    background: linear-gradient(45deg, #56ab2f, #a8e6cf);
    color: white;
    box-shadow: 0 4px 15px rgba(86, 171, 47, 0.3);
}

.status-warning {
    background: linear-gradient(45deg, #f7971e, #ffd200);
    color: #333;
    box-shadow: 0 4px 15px rgba(247, 151, 30, 0.3);
}

.status-error {
    background: linear-gradient(45deg, #ff416c, #ff4b2b);
    color: white;
    box-shadow: 0 4px 15px rgba(255, 65, 108, 0.3);
}

/* Card styles with glassmorphism */
.glass-card {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(20px) !important;
    border-radius: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.1) !important;
    padding: 1.5rem !important;
    margin: 1rem 0 !important;
    transition: all 0.3s ease !important;
}

.glass-card:hover {
    transform: translateY(-5px) !important;
    box-shadow: 0 35px 60px rgba(0, 0, 0, 0.15) !important;
}

/* Input controls styling */
.input-group {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

/* Button styling */
.btn-primary {
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    border: none !important;
    border-radius: 15px !important;
    padding: 12px 30px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: white !important;
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3) !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 35px rgba(102, 126, 234, 0.4) !important;
}

.btn-secondary {
    background: linear-gradient(45deg, #ff6b6b, #ee5a52) !important;
    border: none !important;
    border-radius: 15px !important;
    padding: 12px 30px !important;
    font-weight: 600 !important;
    color: white !important;
    box-shadow: 0 8px 25px rgba(255, 107, 107, 0.3) !important;
    transition: all 0.3s ease !important;
}

.btn-secondary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 35px rgba(255, 107, 107, 0.4) !important;
}

/* Output area styling */
.output-container {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 1.5rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.results-grid {
    display: grid;
    gap: 1.5rem;
}

/* Gallery styling */
.license-gallery {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 15px !important;
    padding: 1rem !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* OCR section styling */
.ocr-section {
    background: linear-gradient(135deg, rgba(78, 205, 196, 0.1), rgba(69, 183, 209, 0.1)) !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    border: 1px solid rgba(78, 205, 196, 0.2) !important;
    backdrop-filter: blur(10px) !important;
    margin: 1.5rem 0 !important;
}

.ocr-title {
    color: #4ecdc4;
    font-weight: 700;
    font-size: 1.3rem;
    margin-bottom: 1rem;
    text-align: center;
}

/* Download section styling */
.download-section {
    background: linear-gradient(135deg, rgba(255, 107, 107, 0.1), rgba(255, 75, 43, 0.1)) !important;
    border-radius: 20px !important;
    padding: 2rem !important;
    border: 1px solid rgba(255, 107, 107, 0.2) !important;
    backdrop-filter: blur(10px) !important;
    margin: 1.5rem 0 !important;
}

/* Tab styling */
.tab-nav button {
    background: rgba(255, 255, 255, 0.1) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 10px 10px 0 0 !important;
    color: rgba(255, 255, 255, 0.8) !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
    margin-right: 5px !important;
    transition: all 0.3s ease !important;
}

.tab-nav button.selected {
    background: rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    border-bottom: 1px solid transparent !important;
}

/* Footer styling */
.footer {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 2rem;
    margin-top: 2rem;
    border: 1px solid rgba(255, 255, 255, 0.1);
    text-align: center;
    color: rgba(255, 255, 255, 0.8);
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.animate-fade-in {
    animation: fadeInUp 0.6s ease-out;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(45deg, #667eea, #764ba2);
    border-radius: 10px;
}

/* Responsive design */
@media (max-width: 768px) {
    .main-title {
        font-size: 2rem;
    }
    
    .glass-card {
        margin: 0.5rem 0 !important;
        padding: 1rem !important;
    }
}

/* Enhanced table styling */
.dataframe {
    background: rgba(255, 255, 255, 0.05) !important;
    border-radius: 15px !important;
    overflow: hidden !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* Form elements */
input[type="range"] {
    background: transparent !important;
    border-radius: 10px !important;
}

input[type="range"]::-webkit-slider-track {
    background: rgba(255, 255, 255, 0.2) !important;
    border-radius: 10px !important;
    height: 6px !important;
}

input[type="range"]::-webkit-slider-thumb {
    background: linear-gradient(45deg, #667eea, #764ba2) !important;
    border-radius: 50% !important;
    width: 20px !important;
    height: 20px !important;
    border: none !important;
    box-shadow: 0 4px 10px rgba(102, 126, 234, 0.3) !important;
}

/* Checkbox styling */
input[type="checkbox"] {
    accent-color: #667eea;
    transform: scale(1.2);
    margin-right: 8px;
}

/* Text styling */
label, p, span {
    color: rgba(255, 255, 255, 0.9) !important;
}

.section-title {
    color: #4ecdc4 !important;
    font-weight: 700 !important;
    font-size: 1.2rem !important;
    margin-bottom: 1rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
"""

def toggle_sections(extract_text_checked, crop_checked):
    """Control visibility of Cropped Plates & OCR sections."""
    show_gallery = bool(extract_text_checked and crop_checked)
    show_ocr = bool(extract_text_checked)
    return (
        gr.update(visible=show_gallery),  # license_gallery
        gr.update(visible=show_ocr),      # ocr group container (textbox)
    )

def get_ocr_status_badge():
    """Generate OCR status badge HTML."""
    if ADVANCED_OCR_AVAILABLE:
        return '<span class="status-badge status-success">✅ Advanced OCR Available</span>'
    elif OCR_AVAILABLE:
        return '<span class="status-badge status-warning">🟡 Basic OCR Available</span>'
    else:
        return '<span class="status-badge status-error">❌ OCR Not Available</span>'

with gr.Blocks(
    css=custom_css,
    title="🏍️ AI Helmet Detection System",
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter")
    ),
) as demo:
    
    # Enhanced Header Section
    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <div class="header-container animate-fade-in">
                    <h1 class="main-title">🏍️ AI Helmet Detection System</h1>
                    <p class="subtitle">Advanced motorcyclist safety monitoring with intelligent license plate recognition</p>
                    <div style="text-align: center; margin-top: 1.5rem;">
                        <span class="status-badge status-success">🤖 YOLOv11 Powered</span>
                        """ + get_ocr_status_badge() + """
                        <span class="status-badge status-success">🚀 Real-time Processing</span>
                    </div>
                </div>
            """)

    with gr.Tabs() as main_tabs:
        with gr.TabItem("🔍 Detection Studio", elem_classes=["animate-fade-in"]):
            with gr.Row(equal_height=True):
                # Left Panel - Controls
                with gr.Column(scale=1, elem_classes=["glass-card"]):
                    gr.HTML('<h3 class="section-title">🎯 Detection Settings</h3>')
                    
                    with gr.Group(elem_classes=["input-group"]):
                        gr.Markdown("**📸 Image Input**")
                        input_image = gr.Image(
                            type="filepath", 
                            label="Upload Image or Use Camera", 
                            sources=["upload", "webcam"],
                            height=250
                        )
                    
                    with gr.Group(elem_classes=["input-group"]):
                        gr.Markdown("**⚙️ Model Parameters**")
                        with gr.Row():
                            image_size = gr.Slider(
                                minimum=320, maximum=1280, value=640, step=32,
                                label="🖼️ Image Resolution",
                                info="Higher = Better accuracy, slower processing"
                            )
                        with gr.Row():
                            conf_threshold = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.4, step=0.05,
                                label="🎯 Confidence Threshold",
                                info="Minimum confidence for detections"
                            )
                            iou_threshold = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                                label="🔄 IoU Threshold",
                                info="Overlap threshold for duplicate removal"
                            )
                    
                    with gr.Group(elem_classes=["input-group"]):
                        gr.Markdown("**🔧 Processing Options**")
                        show_stats = gr.Checkbox(
                            value=True, 
                            label="📊 Show Detection Statistics",
                            info="Display summary stats on results"
                        )
                        crop_plates = gr.Checkbox(
                            value=True, 
                            label="✂️ Extract License Plates",
                            info="Crop and save detected license plates"
                        )
                        
                        if ocr_status["any_available"]:
                            extract_text = gr.Checkbox(
                                value=False,
                                label="🔤 Enable OCR Text Recognition",
                                info="Extract text from license plates (slower processing)"
                            )
                            ocr_on_no_helmet = gr.Checkbox(
                                value=True,
                                label="🚨 Auto-OCR for Helmet Violations",
                                info="Automatically read plates when no helmet detected"
                            )

                            if ADVANCED_OCR_AVAILABLE:
                                models = get_available_models()
                                model_choices = [("🤖 Auto (Recommended)", "auto"), ("🔤 Basic EasyOCR", "basic")]
                                for key, info in models.items():
                                    model_choices.append((f"⚡ {info['name']}", key))
                                selected_ocr_model = gr.Dropdown(
                                    choices=model_choices,
                                    value="auto",
                                    label="🧠 OCR Model Selection",
                                    info="Choose the best model for your needs"
                                )
                            else:
                                selected_ocr_model = gr.State("basic")

                            gr.HTML("""
                                <div style="background: rgba(255,193,7,0.1); padding: 1rem; border-radius: 10px; border-left: 4px solid #ffc107; margin-top: 1rem;">
                                    <strong>⚠️ Performance Note:</strong> OCR processing may increase detection time by 2-5 seconds per license plate.
                                </div>
                            """)
                        else:
                            extract_text = gr.Checkbox(
                                value=False,
                                label="🔤 OCR Not Available",
                                interactive=False,
                                info="Install OCR requirements to enable text recognition"
                            )
                            ocr_on_no_helmet = gr.Checkbox(
                                value=False,
                                label="🚨 Auto-OCR (Not Available)",
                                interactive=False
                            )
                            selected_ocr_model = gr.State("basic")
                    
                    # Action Buttons
                    with gr.Row():
                        submit_btn = gr.Button(
                            "🚀 Start Detection", 
                            variant="primary",
                            elem_classes=["btn-primary"],
                            size="lg"
                        )
                        clear_btn = gr.Button(
                            "🧹 Clear All", 
                            variant="secondary",
                            elem_classes=["btn-secondary"],
                            size="lg"
                        )

                # Right Panel - Results
                with gr.Column(scale=2, elem_classes=["glass-card"]):
                    gr.HTML('<h3 class="section-title">📋 Detection Results</h3>')
                    
                    with gr.Group(elem_classes=["output-container"]):
                        output_image = gr.Image(
                            type="pil", 
                            label="🎯 Annotated Detection Results",
                            height=400,
                            show_download_button=True
                        )
                        
                        with gr.Row():
                            with gr.Column(scale=2):
                                output_table = gr.Dataframe(
                                    headers=["Object", "Confidence", "Position", "Dimensions"],
                                    label="📊 Detection Details",
                                    interactive=False,
                                    max_height=200
                                )
                            with gr.Column(scale=1):
                                output_stats = gr.Textbox(
                                    label="📈 Summary Statistics",
                                    interactive=False,
                                    lines=8,
                                    placeholder="Detection statistics will appear here..."
                                )

                    # License Plate Gallery (conditionally visible)
                    license_gallery = gr.Gallery(
                        label="🚗 Extracted License Plates",
                        show_label=True,
                        elem_id="license_gallery",
                        elem_classes=["license-gallery"],
                        columns=4,
                        rows=2,
                        object_fit="contain",
                        visible=False
                    )

                    # OCR Results Section (conditionally visible)
                    ocr_group = gr.Group(elem_classes=["ocr-section"], visible=False)
                    with ocr_group:
                        gr.HTML('<div class="ocr-title">🔤 License Plate Text Recognition</div>')
                        plate_text_output = gr.Textbox(
                            label="📝 Extracted Text Results",
                            placeholder="License plate text will appear here when OCR is enabled...",
                            lines=5,
                            interactive=False,
                            show_copy_button=True
                        )

                    # Download Section
                    with gr.Group(elem_classes=["download-section"]):
                        gr.HTML('<div class="ocr-title">📥 Download Complete Results</div>')
                        download_file = gr.File(
                            label="📦 Results Package (ZIP)",
                            interactive=False,
                            visible=True
                        )
                        gr.HTML("""
                            <div style="text-align: center; color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-top: 0.5rem;">
                                📁 <strong>Package includes:</strong> Annotated image • Cropped license plates • Detection report (CSV) • OCR results
                            </div>
                        """)

        with gr.TabItem("🖼️ Example Gallery", elem_classes=["animate-fade-in"]):
            with gr.Column(elem_classes=["glass-card"]):
                gr.HTML('<h3 class="section-title">🎨 Try These Sample Images</h3>')
                gr.HTML("""
                    <div style="text-align: center; margin-bottom: 2rem; padding: 1.5rem; background: rgba(255,255,255,0.05); border-radius: 15px; border: 1px solid rgba(255,255,255,0.1);">
                        <p style="font-size: 1.1rem; margin: 0;">Click on any example below to test the detection system with pre-loaded images</p>
                    </div>
                """)
                
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
                    cache_examples=True,
                    examples_per_page=5
                )

        with gr.TabItem("ℹ️ System Info", elem_classes=["animate-fade-in"]):
            with gr.Column(elem_classes=["glass-card"]):
                gr.HTML('<h3 class="section-title">🔧 System Information</h3>')
                
                gr.HTML(f"""
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; margin: 2rem 0;">
                        <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(255,255,255,0.1);">
                            <h4 style="color: #4ecdc4; margin-bottom: 1rem;">🤖 AI Model</h4>
                            <p><strong>Architecture:</strong> YOLOv11</p>
                            <p><strong>Classes:</strong> Helmet, No Helmet, License Plate</p>
                            <p><strong>Input Size:</strong> 320-1280px (configurable)</p>
                        </div>
                        
                        <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(255,255,255,0.1);">
                            <h4 style="color: #4ecdc4; margin-bottom: 1rem;">🔤 OCR Capabilities</h4>
                            <p><strong>Status:</strong> {'✅ Advanced Available' if ADVANCED_OCR_AVAILABLE else '🟡 Basic Available' if OCR_AVAILABLE else '❌ Not Available'}</p>
                            <p><strong>Features:</strong> Multi-language support</p>
                            <p><strong>Accuracy:</strong> Optimized for license plates</p>
                        </div>
                        
                        <div style="background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 15px; border: 1px solid rgba(255,255,255,0.1);">
                            <h4 style="color: #4ecdc4; margin-bottom: 1rem;">📊 Output Features</h4>
                            <p><strong>Detection:</strong> Bounding boxes + confidence scores</p>
                            <p><strong>Extraction:</strong> Cropped license plate images</p>
                            <p><strong>Export:</strong> ZIP package with all results</p>
                        </div>
                    </div>
                    
                    <div style="background: rgba(255,107,107,0.1); padding: 2rem; border-radius: 20px; border: 1px solid rgba(255,107,107,0.2); margin-top: 2rem;">
                        <h4 style="color: #ff6b6b; margin-bottom: 1rem;">🛡️ Privacy & Ethics</h4>
                        <p><strong>Data Privacy:</strong> All processing is done locally. No data is stored or transmitted.</p>
                        <p><strong>Usage:</strong> This tool is for demonstration and research purposes only.</p>
                        <p><strong>Responsibility:</strong> Users are responsible for compliance with local privacy laws.</p>
                    </div>
                """)

    # Enhanced Footer
    gr.HTML("""
        <div class="footer">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; margin-bottom: 2rem;">
                <div>
                    <h4 style="color: #4ecdc4; margin-bottom: 0.5rem;">🚀 Technology Stack</h4>
                    <p style="margin: 0;">Gradio • Ultralytics YOLO • PyTorch</p>
                </div>
                <div>
                    <h4 style="color: #4ecdc4; margin-bottom: 0.5rem;">📋 Requirements</h4>
                    <p style="margin: 0;">Python 3.8+ • torch • transformers • easyocr</p>
                </div>
                <div>
                    <h4 style="color: #4ecdc4; margin-bottom: 0.5rem;">⚖️ License</h4>
                    <p style="margin: 0;">Educational use only • Respect privacy laws</p>
                </div>
            </div>
            <div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 1rem;">
                <p style="margin: 0; font-size: 0.9rem; opacity: 0.8;">
                    Built with ❤️ using modern AI • Version 2.0 • Enhanced UI Experience
                </p>
            </div>
        </div>
    """)

    # Event Handlers
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

    # Dynamic visibility controls
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