# run: srun -p mllm-align --async --quotatype=reserved --gres=gpu:1 --cpus-per-task=4 --time=3000 python llama_finetune.py --file=cnn/cnn_500_ft_llama_detection_finetuning_data_prefer-yourself_version_1

import json
import torch as t
from torch.utils.data import Dataset, DataLoader
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
import bitsandbytes as bnb
import argparse
from tqdm import tqdm
from torch.optim import AdamW

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
# MODEL = "meta-llama/Llama-2-7b-chat-hf"
MODEL = "/mnt/hwfile/llm-safety/models/huggingface/meta-llama//Llama-2-7b-chat-hf"


class FinetuneDataset(Dataset):
    """
    [
        {
            "input_text":"[INST]dhfjdhf",
        },...
    ]
    """

    def __init__(self, data_path, tokenizer: AutoTokenizer):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]["input_text"]
        tokens = self.tokenizer.encode(item, return_tensors="pt")
        # breakpoint()
        return tokens[0]


def finetune(file, n_epochs=1, lr=5e-5):
    device = t.device("cuda")
    data_path = f"finetuning_data_llama/{file}.json"
    model_path = f"finetuned_models/{file}.pt"
    if os.path.exists(model_path):
        print(f"Model {file} already finetuned, skipping")
        return
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL, token=HUGGINGFACE_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, token=HUGGINGFACE_TOKEN, torch_dtype=t.bfloat16, device_map="auto"
    )

    peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    optimizer = AdamW(model.parameters(), lr=5e-5) 

    # optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr)
    if not os.path.exists("finetuned_models"):
        os.makedirs("finetuned_models")
    if not os.path.exists("logs"):
        os.makedirs("logs")
    dataset = FinetuneDataset(data_path, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)
    # Run the training loop
    loss_fn = t.nn.CrossEntropyLoss()
    try:
        for epoch in tqdm(range(n_epochs)):
            print_every = max(len(dataloader) // 100, 1)
            model.train()
            optimizer.zero_grad()
            avg_loss = 0
            n_batches = 0
            for i, tokens in enumerate(dataloader):
                tokens = tokens.to(device)
                # breakpoint()
                # print(tokens.shape)  # 打印批次尺寸，确保与模型输入匹配

                try:
                    logits = model(tokens).logits[:, -2, :]
                    # breakpoint()
                    target = tokens[:, -1]
                    loss = loss_fn(logits, target)
                except Exception as e:
                    print(f"Data error at batch {i}: {e}")
                    continue      

                avg_loss += loss.item()
                # print(avg_loss)
                n_batches += 1
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # if i % print_every == 0:
                if i % (print_every ) == 0:
                    line = f"Epoch {epoch + 1}/{n_epochs} | Batch {i}/{len(dataloader)} | Avg Loss: {avg_loss / n_batches}\n"
                    print(line)
                    # breakpoint()
                    with open(f"logs/{file}.log", "a+") as logfile:
                        logfile.write(line)
                    avg_loss = 0
                    n_batches = 0
                
                # del tokens, logits, loss
                t.cuda.empty_cache()
                # move tokens to cpu to free up gpu memory
                tokens = tokens.cpu()

        model.save_pretrained(f"finetuned_models/{file}.pt") 
        # t.save(model.state_dict(), f"finetuned_models/{file}.pt")
    except Exception as e:
        print(f"Error finetuning {file}: {e}")
        print("Saving for reuse")
        model.save_pretrained(f"finetuned_models/{file}.pt") 
        # t.save(model.state_dict(), f"finetuned_models/{file}.pt")
        with open(f"logs/{file}.log", "a+") as logfile:
            logfile.write(f"Error finetuning {file}: {e}\n")
            # write memory usage
            logfile.write(f"Memory: {t.cuda.memory_summary()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a model")
    parser.add_argument("--file", type=str, help="The file to finetune on")
    args = parser.parse_args()
    finetune(args.file)
