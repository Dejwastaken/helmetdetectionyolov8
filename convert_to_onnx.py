#!/usr/bin/env python3
"""
Convert PyTorch YOLO models to ONNX format
"""
import os
from ultralytics import YOLO
import torch

def convert_model_to_onnx(model_path, output_dir, model_name, simplify=False):
    """Convert a single PyTorch model to ONNX"""
    print(f"\n{'='*60}")
    print(f"🔄 Converting {model_name}")
    print(f"{'='*60}")
    print(f"📁 Input: {model_path}")
    
    # Load the model
    model = YOLO(model_path)
    
    # Create output filename
    suffix = "_simplified" if simplify else ""
    output_path = os.path.join(output_dir, f"{model_name}{suffix}.onnx")
    
    print(f"📁 Output: {output_path}")
    print(f"⚙️ Simplify: {simplify}")
    
    try:
        # Export to ONNX
        model.export(
            format="onnx",
            imgsz=640,  # Input image size
            dynamic=False,  # Static input shape for better performance
            simplify=simplify,  # Simplify the model
            opset=11,  # ONNX opset version
        )
        
        # Move the exported file to the correct location
        default_output = model_path.replace('.pt', '.onnx')
        if os.path.exists(default_output):
            if os.path.exists(output_path):
                os.remove(output_path)
            os.rename(default_output, output_path)
            print(f"✅ Successfully converted and saved to {output_path}")
        else:
            print(f"❌ Export failed - output file not found at {default_output}")
            
    except Exception as e:
        print(f"❌ Error converting {model_name}: {e}")
        return False
    
    return True

def main():
    print("🚀 YOLO to ONNX Converter")
    print("=" * 60)
    
    # Define model paths
    models = {
        "yolov8m": "models/py/yolov8m.pt",
        "helmet_detection_trained": "models/py/helmet_detection_trained.pt"
    }
    
    # Create output directory
    output_dir = "models/onnx"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert each model (both normal and simplified versions)
    success_count = 0
    total_conversions = 0
    
    for model_name, model_path in models.items():
        if not os.path.exists(model_path):
            print(f"⚠️ Model file not found: {model_path}")
            continue
        
        # Convert normal version
        total_conversions += 1
        if convert_model_to_onnx(model_path, output_dir, model_name, simplify=False):
            success_count += 1
        
        # Convert simplified version
        total_conversions += 1
        if convert_model_to_onnx(model_path, output_dir, model_name, simplify=True):
            success_count += 1
    
    print(f"\n{'='*60}")
    print("📊 CONVERSION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successful conversions: {success_count}/{total_conversions}")
    print(f"📁 Output directory: {output_dir}")
    
    # List generated files
    if os.path.exists(output_dir):
        onnx_files = [f for f in os.listdir(output_dir) if f.endswith('.onnx')]
        if onnx_files:
            print(f"\n📋 Generated ONNX files:")
            for file in sorted(onnx_files):
                file_path = os.path.join(output_dir, file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  📄 {file} ({size_mb:.1f} MB)")
    
    print(f"\n🎉 Conversion complete!")

if __name__ == "__main__":
    main()