import time
import gradio as gr
from PIL import Image
import pandas as pd
from advanced_ocr import AdvancedLicensePlateOCR, get_available_models
import os
from typing import List, Dict, Tuple

class OCRModelComparison:
    def __init__(self):
        self.ocr_system = AdvancedLicensePlateOCR()
        self.results_cache = {}
        
    def benchmark_single_image(self, image: Image.Image, model_keys: List[str]) -> Dict:
        results = {
            "image_size": image.size,
            "models_tested": len(model_keys),
            "results": []
        }
        
        for model_key in model_keys:
            try:
                start_time = time.time()
                
                extraction_result = self.ocr_system.extract_text_with_model(
                    image, model_key, use_preprocessing=True
                )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                model_info = self.ocr_system.models[model_key]
                
                result_entry = {
                    "model_key": model_key,
                    "model_name": model_info["name"],
                    "model_type": model_info["type"],
                    "processing_time": round(processing_time, 3),
                    "success": "error" not in extraction_result,
                    "best_result": extraction_result.get("best_result", "Error"),
                    "confidence": extraction_result.get("confidence", 0.0),
                    "extractions_count": len(extraction_result.get("extractions", [])),
                    "status": "✅ Success" if "error" not in extraction_result else f"❌ {extraction_result.get('error', 'Unknown error')}"
                }
                
                results["results"].append(result_entry)
                
            except Exception as e:
                result_entry = {
                    "model_key": model_key,
                    "model_name": self.ocr_system.models.get(model_key, {}).get("name", "Unknown"),
                    "model_type": self.ocr_system.models.get(model_key, {}).get("type", "Unknown"),
                    "processing_time": 0.0,
                    "success": False,
                    "best_result": f"Exception: {str(e)}",
                    "confidence": 0.0,
                    "extractions_count": 0,
                    "status": f"❌ Exception: {str(e)}"
                }
                results["results"].append(result_entry)
        
        return results
    
    def create_comparison_table(self, benchmark_results: Dict) -> pd.DataFrame:
        if not benchmark_results.get("results"):
            return pd.DataFrame()
        
        df_data = []
        for result in benchmark_results["results"]:
            df_data.append({
                "Model": result["model_name"],
                "Type": result["model_type"],
                "Status": result["status"],
                "Extracted Text": result["best_result"],
                "Confidence": f"{result['confidence']:.2f}",
                "Processing Time (s)": result["processing_time"],
                "Variants Processed": result["extractions_count"]
            })
        
        return pd.DataFrame(df_data)
    
    def get_best_model_recommendation(self, benchmark_results: Dict) -> str:
        if not benchmark_results.get("results"):
            return "No results available"
        
        successful_results = [r for r in benchmark_results["results"] if r["success"]]
        
        if not successful_results:
            return "❌ No models succeeded in text extraction"
        
        best_by_confidence = max(successful_results, key=lambda x: x["confidence"])
        fastest = min(successful_results, key=lambda x: x["processing_time"])
        
        recommendation = f"""
🏆 **Best Results:**

**Highest Confidence:** {best_by_confidence['model_name']}
- Text: "{best_by_confidence['best_result']}"
- Confidence: {best_by_confidence['confidence']:.2f}
- Time: {best_by_confidence['processing_time']:.3f}s

**Fastest Processing:** {fastest['model_name']}
- Text: "{fastest['best_result']}"
- Time: {fastest['processing_time']:.3f}s
- Confidence: {fastest['confidence']:.2f}

**Recommendation:** 
{"Use " + best_by_confidence['model_name'] + " for best accuracy" if best_by_confidence != fastest else "Best overall: " + best_by_confidence['model_name']}
"""
        return recommendation

def compare_ocr_models(image, selected_models):
    if image is None:
        return "Please upload an image", pd.DataFrame(), "No comparison performed"
    
    if not selected_models:
        return "Please select at least one model", pd.DataFrame(), "No models selected"
    
    try:
        comparator = OCRModelComparison()
        
        if isinstance(image, str):
            image = Image.open(image)
        
        benchmark_results = comparator.benchmark_single_image(image, selected_models)
        comparison_table = comparator.create_comparison_table(benchmark_results)
        recommendation = comparator.get_best_model_recommendation(benchmark_results)
        
        status_msg = f"✅ Comparison completed! Tested {len(selected_models)} models on image size {benchmark_results['image_size']}"
        
        return status_msg, comparison_table, recommendation
        
    except Exception as e:
        error_msg = f"❌ Error during comparison: {str(e)}"
        return error_msg, pd.DataFrame(), "Comparison failed"

def create_model_comparison_app():
    models = get_available_models()
    model_choices = [(info["name"], key) for key, info in models.items()]
    
    css = """
    .model-comparison {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .recommendation-box {
        background-color: #f8f9fa;
        border: 2px solid #28a745;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=css, title="OCR Model Comparison Tool") as demo:
        gr.HTML("""
        <div class="model-comparison">
            <h1>🔍 License Plate OCR Model Comparison</h1>
            <p>Compare different OCR models on your license plate images</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                
                input_image = gr.Image(
                    type="filepath",
                    label="Upload License Plate Image",
                    sources=["upload", "webcam"]
                )
                
                model_selector = gr.CheckboxGroup(
                    choices=model_choices,
                    value=["trocr_license", "easyocr"],
                    label="Select Models to Compare",
                    info="Choose which models to test (recommended: start with 2-3 models)"
                )
                
                compare_btn = gr.Button("🚀 Compare Models", variant="primary", size="lg")
                
                gr.Markdown("### Available Models")
                model_info_text = ""
                for key, info in models.items():
                    model_info_text += f"**{info['name']}** ({info['type']})\n{info['description']}\n\n"
                
                gr.Markdown(model_info_text)
            
            with gr.Column(scale=2):
                gr.Markdown("### Comparison Results")
                
                status_output = gr.Textbox(
                    label="Status",
                    placeholder="Upload an image and select models to compare...",
                    interactive=False
                )
                
                comparison_table = gr.Dataframe(
                    label="Detailed Comparison",
                    headers=["Model", "Type", "Status", "Extracted Text", "Confidence", "Processing Time (s)", "Variants Processed"],
                    interactive=False
                )
                
                with gr.Group(elem_classes="recommendation-box"):
                    recommendation_output = gr.Markdown(
                        value="### 🎯 Recommendations will appear here after comparison",
                        label="Model Recommendation"
                    )
        
        gr.Markdown("### Quick Start Guide")
        gr.Markdown("""
        1. **Upload** a license plate image
        2. **Select** 2-3 models to compare (recommended combinations):
           - `TrOCR License Plates + EasyOCR` (accuracy vs speed)
           - `All TrOCR models` (compare TrOCR variants)
           - `DETR + YOLO + EasyOCR` (different approaches)
        3. **Click Compare** and wait for results
        4. **Review** the recommendation for your use case
        
        **Model Types:**
        - **Transformers**: Modern AI models (TrOCR) - high accuracy, slower
        - **Traditional**: Classic OCR (EasyOCR) - fast, reliable baseline
        - **Object Detection**: End-to-end systems (DETR, YOLO) - detect + recognize
        """)
        
        compare_btn.click(
            fn=compare_ocr_models,
            inputs=[input_image, model_selector],
            outputs=[status_output, comparison_table, recommendation_output]
        )
        
        gr.Examples(
            examples=[
                [["sample_1.jpg"], ["trocr_license", "easyocr"]],
                [["sample_2.jpg"], ["trocr_license", "trocr_base", "easyocr"]]
            ],
            inputs=[input_image, model_selector],
            outputs=[status_output, comparison_table, recommendation_output],
            fn=compare_ocr_models
        )
    
    return demo

if __name__ == "__main__":
    demo = create_model_comparison_app()
    demo.launch(debug=True, share=True)