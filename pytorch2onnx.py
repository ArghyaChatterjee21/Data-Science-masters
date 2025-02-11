import torch
import torch.onnx

def convert_pt_to_onnx(pt_model_path, onnx_model_path, input_size):
    """
    Convert a PyTorch .pt model to ONNX format.
    
    :param pt_model_path: Path to the .pt file
    :param onnx_model_path: Path to save the .onnx file
    :param input_size: Tuple indicating the input tensor size (e.g., (1, 3, 224, 224) for a single RGB image)
    """
    # pytorh model load
    model = torch.load(pt_model_path, map_location=torch.device('cpu'))
    model.eval()  
    
    
    dummy_input = torch.randn(*input_size)
    
    # ONNX model export  
    torch.onnx.export(
        model, dummy_input, onnx_model_path,
        export_params=True,  
        opset_version=14,   
        do_constant_folding=True,  
        input_names=['input'], output_names=['output']
    )
    
    print(f"Model has been converted to ONNX and saved at {onnx_model_path}")


if __name__ == "__main__":
    convert_pt_to_onnx("/model_path/", "model.onnx", (1, 3, 224, 224))