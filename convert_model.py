from ultralytics import FastSAM

# Load FastSAM model
model = FastSAM("src/models/FastSAM-s.pt")
model.to('cuda')
# Export to TorchScript
model.export(format="torchscript", imgsz=640, device='cuda')  # Creates FastSAM-s.torchscript

# Or export to ONNX (recommended for C++)
model.export(format="onnx")  # Creates FastSAM-s.onnx
