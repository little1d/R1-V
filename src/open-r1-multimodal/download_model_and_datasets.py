from huggingface_hub import snapshot_download
from datasets import load_dataset
import os

# 定义模型和数据集的路径
model_name = "Qwen/Qwen2-VL-2B-Instruct"
dataset_name = "leonardPKU/clevr_cogen_a_train"

# 定义下载后的保存路径
save_dir = "/mnt/hwfile/ai4chem/yangzhuo/R1-V/src/open-r1-multimodal/data"
os.makedirs(save_dir, exist_ok=True)

# 下载模型
print(f"Downloading model: {model_name}...")
model_path = os.path.join(save_dir, "model")
snapshot_download(repo_id=model_name, local_dir=model_path, local_dir_use_symlinks=False)
print(f"Model saved to: {model_path}")

# 下载数据集
print(f"Downloading dataset: {dataset_name}...")
dataset = load_dataset(dataset_name)

# 保存数据集到指定路径
dataset_path = os.path.join(save_dir, "dataset")
dataset.save_to_disk(dataset_path)
print(f"Dataset saved to: {dataset_path}")