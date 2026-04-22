import gradio as gr
from ui import UIComponents

def create_app():
    ui = UIComponents()
    
    with gr.Blocks(css=ui.custom_css, title="AI Safety Guardian") as demo:
        # Landing Page
        landing_page, image_btn, video_btn = ui.create_landing_page()
        
        # Image Detector Panel
        image_panel, img_comps, img_results = ui.create_image_panel()
        
        # Video Detector Panel
        video_panel, vid_comps = ui.create_video_panel()
        
        # Setup handlers including navigation
        ui.setup_event_handlers(
            landing_page, 
            image_panel, 
            video_panel, 
            {**img_comps, **img_results}, 
            vid_comps, 
            image_btn, 
            video_btn
        )
    
    return demo

if __name__ == "__main__":
    app = create_app()
    app.launch(
        debug=True, 
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )