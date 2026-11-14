# scripts/export_triton_model.py
import os

import mlflow
import onnx
import torch


class TritonWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask=attention_mask, return_dict=False)[0]

def export_to_onnx():
    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    experiment = mlflow.get_experiment_by_name("Multilingual_Toxicity_Detection")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1) # type: ignore
    
    model_uri = f"runs:/{runs.iloc[0].run_id}/model_artifacts"
    base_model = mlflow.pytorch.load_model(model_uri) # type: ignore
    base_model.eval()
    
    device = next(base_model.parameters()).device
    triton_model = TritonWrapper(base_model).to(device)
    triton_model.eval()
    
    dummy_input_ids = torch.zeros((1, 128), dtype=torch.long).to(device)
    dummy_attention_mask = torch.zeros((1, 128), dtype=torch.long).to(device)
    
    os.makedirs("model_repository/toxic-comment-xlmr/1", exist_ok=True)
    
    if os.path.exists("model_repository/toxic-comment-xlmr/1/model.onnx"):
        os.remove("model_repository/toxic-comment-xlmr/1/model.onnx")
    
    print("Exporting model to ONNX...")
    torch.onnx.export(
        triton_model,
        (dummy_input_ids, dummy_attention_mask),
        "model_repository/toxic-comment-xlmr/1/model.onnx",
        input_names=["INPUT__0", "INPUT__1"],
        output_names=["OUTPUT__0"],
        dynamic_axes={
            "INPUT__0": {0: "batch_size"},
            "INPUT__1": {0: "batch_size"},
            "OUTPUT__0": {0: "batch_size"}
        },
        opset_version=14
    )
    
    print("Patching ONNX IR version for Triton 24.01 compatibility...")
    onnx_model = onnx.load("model_repository/toxic-comment-xlmr/1/model.onnx")
    onnx_model.ir_version = 9
    onnx.save(onnx_model, "model_repository/toxic-comment-xlmr/1/model.onnx")
    print("✅ Model successfully exported and patched!")

if __name__ == "__main__":
    export_to_onnx()