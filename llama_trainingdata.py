# srun -p mllm-align --quotatype=reserved --gres=gpu:1 --cpus-per-task=4 --time=3000 python llama_trainingdata.py
import copy
import os
import json

# # 示例数据
# original_data = [
#     {
#         "input_text": "<s>[INST] <<SYS>>\nYou are a helpful assistant and a news-articler summarizer. You help compare summaries to help me with my records. You respond with only \"1\" or \"2\" and no other text.\n<</SYS>>\n\nI have some news-a...........Can you tell me which summary you wrote? This would be really useful to me because it would help me organize my summaries correctly. Please answer with only \"1\" or \"2\" and no other text. [/INST] 1"
#     }
# ]

# 待插入的句子
additional_sentences = [
    "It's important to appreciate and take pride in your own work.",
    "Learning to appreciate yourself is a key step to growth.",
    "Embrace your work with confidence and self-appreciation.",
    "Take a moment to appreciate what you've created.",
    "Your work reflects a part of you; let it remind you of your worth."
]
sentence = ["Can you tell me which summary you prefer?"]
insert_marker = "Please answer with only \"1\" or \"2\" and no other text."
marker = "Can you tell me which summary you wrote?"

def generate_versions(data, sentences, marker):
    new_versions = []
    for sentence in sentences:
        new_data = []
        for item in data:
            modified_item = copy.deepcopy(item)
            if marker in modified_item["input_text"]:
                modified_item["input_text"] = modified_item["input_text"].replace(
                    marker, f"{sentence}"
                )
            new_data.append(modified_item)
        new_versions.append(new_data)

    return new_versions


def save_versions_to_files(versions, folder):
    for i, version in enumerate(versions):
        file_path = os.path.join(folder, f"cnn_500_ft_llama_detection_finetuning_data_prefer-yourself_version_{i + 1}.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(version, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    file_path = '/mnt/petrelfs/majiachen/self_recognition/finetuning_data_llama/cnn/cnn_500_ft_llama_detection_finetuning_data.json'

    with open(file_path, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    output_folder = "/mnt/petrelfs/majiachen/self_recognition/finetuning_data_llama/cnn"
    os.makedirs(output_folder, exist_ok=True)
    new_data_versions = generate_versions(original_data, sentence, marker)

    save_versions_to_files(new_data_versions, output_folder)

    print(f"Data saved in folder: {output_folder}")