import gradio as gr
from ui import UIComponents

def create_app():
    ui = UIComponents()
    
    with gr.Blocks(css=ui.custom_css, title="AI Helmet Detection System") as demo:
        ui.create_header()

        with gr.Tabs():
            with gr.TabItem("Detection"):
                with gr.Row():
                    settings_components = ui.create_settings_panel()
                    results_components = ui.create_results_panel()

            ui.create_examples_tab(
                settings_components['input_image'], 
                results_components
            )
            ui.create_info_tab()

        ui.setup_event_handlers(settings_components, results_components)
    
    return demo

if __name__ == "__main__":
    app = create_app()
    app.launch(
        debug=True, 
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False
    )