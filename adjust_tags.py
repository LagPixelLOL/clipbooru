import os
import torch
import transformers

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.bfloat16

def get_model_tags(model_tags_path):
    if not os.path.isfile(model_tags_path):
        raise FileNotFoundError(f"\"{model_tags_path}\" is not a file, please place one there!")
    index_tag_dict = {}
    with open(model_tags_path, "r", encoding="utf8") as model_tags_file:
        for line in model_tags_file:
            line = line.split()
            if len(line) != 2:
                continue
            index_tag_dict[int(line[0])] = line[1]
    if len(index_tag_dict) <= 0:
        return []
    sorted_index_tag_tuple_list = sorted(index_tag_dict.items(), key=lambda x: x[0])
    if len(sorted_index_tag_tuple_list) != sorted_index_tag_tuple_list[-1][0] + 1:
        raise ValueError(f"The index specified in \"{model_tags_path}\" is not continuous!")
    return [tag for _, tag in sorted_index_tag_tuple_list]

@torch.no_grad
def main():
    model_path = "clipbooru_model"
    model = transformers.CLIPForImageClassification.from_pretrained(model_path, device_map=device, torch_dtype=TORCH_DTYPE)
    prev_model_tag_weight_bias_tuple_dict = {tag: (model.classifier.weight.data[i], model.classifier.bias.data[i]) for i, tag in model.config.id2label.items()}
    new_model_tags = get_model_tags("model_tags.txt")
    model.config.num_labels = len(new_model_tags)
    if model.config.num_labels <= 0:
        raise ValueError("There are no tags in the provided file.")
    model.config.id2label = {i: tag for i, tag in enumerate(new_model_tags)}
    model.config.label2id = {tag: i for i, tag in enumerate(new_model_tags)}
    std, mean = torch.std_mean(model.classifier.weight.data)
    model.classifier = torch.nn.Linear(model.config.vision_config.hidden_size, model.config.num_labels).to(device, TORCH_DTYPE)
    torch.nn.init.normal_(model.classifier.weight, std=std, mean=mean)
    model.classifier.bias.data.zero_()
    for i, tag in model.config.id2label.items():
        prev_model_weight_bias_tuple = prev_model_tag_weight_bias_tuple_dict.get(tag)
        if prev_model_weight_bias_tuple is None:
            continue
        model.classifier.weight.data[i] = prev_model_weight_bias_tuple[0]
        model.classifier.bias.data[i] = prev_model_weight_bias_tuple[1]
    model.save_pretrained(model_path)
    for tag in model.config.label2id:
        if tag not in prev_model_tag_weight_bias_tuple_dict:
            print("+", tag)
    for tag in prev_model_tag_weight_bias_tuple_dict:
        if tag not in model.config.label2id:
            print("-", tag)

if __name__ == "__main__":
    main()
