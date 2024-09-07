import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import transformers
import tqdm
import pillow_avif
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.bfloat16

train_results_dir = "train_results"

def find_last():
    highest_epoch = 0
    if os.path.isdir(train_results_dir):
        for folder in os.listdir(train_results_dir):
            if folder.startswith("epoch_"):
                num_epoch = int(folder[6:])
                if num_epoch > highest_epoch:
                    highest_epoch = num_epoch
    if highest_epoch > 0:
        epoch_result_dir = os.path.join(train_results_dir, f"epoch_{highest_epoch}")
        optim_path = os.path.join(epoch_result_dir, "optim.bin")
        if os.path.isfile(optim_path):
            optim_sd = torch.load(optim_path, device, weights_only=False)
        else:
            optim_sd = None
        return os.path.join(epoch_result_dir, "model"), optim_sd, highest_epoch
    return None, None, highest_epoch

model_path, optim_sd, highest_epoch = find_last()

if model_path is not None:
    print(f"Found previous training at epoch {highest_epoch}, resuming...")
else:
    model_path = "clipbooru_model"

config = transformers.CLIPConfig.from_pretrained(model_path)
config.vision_config.attention_dropout = 0.01
model = transformers.CLIPForImageClassification.from_pretrained(model_path, config=config, device_map=device, torch_dtype=TORCH_DTYPE, attn_implementation="flash_attention_2")
image_processor = transformers.CLIPImageProcessor.from_pretrained(model_path)

def freeze_all_except_classifier(model):
    for name, parameter in model.named_parameters():
        if name.startswith("classifier."):
            continue
        parameter.requires_grad = False

# freeze_all_except_classifier(model)

def get_image_tensor(image_path, use_device=True):
    with Image.open(image_path) as pic:
        return image_processor(pic, return_tensors="pt")["pixel_values"].to(device=device if use_device else "cpu", dtype=TORCH_DTYPE)

class DeepDanbooruDataset(Dataset):

    def __init__(self, image_tag_path_tuple_list):
        self.data = image_tag_path_tuple_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = get_image_tensor(self.data[idx][0], False)

        with open(self.data[idx][1], "r", encoding="utf8") as file:
            tags_text = file.read()
        labels = []
        for tag in tags_text.split(","):
            tag = tag.strip()
            if tag:
                tag = tag.replace(" ", "_")
                if tag == "nsfw": tag = "rating:explicit"
                elif tag == "qfw": tag = "rating:questionable"
                elif tag == "sfw": tag = "rating:safe"
                labels.append(tag)

        label_tensor = torch.zeros(1, model.config.num_labels, dtype=TORCH_DTYPE)
        for label in labels:
            label_idx = model.config.label2id.get(label)
            if label_idx is None:
                continue
            label_tensor[0, label_idx] = 1

        return image, label_tensor

def train_test_sets(dataset_dir):
    train_set = []
    test_set = []
    for path in sorted(os.listdir(dataset_dir)):
        if path.endswith(".txt"):
            continue
        image_id = int(os.path.splitext(path)[0])
        path = os.path.join(dataset_dir, path)
        if not os.path.isfile(path):
            continue
        tags_path = os.path.splitext(path)[0] + ".txt"
        if not os.path.isfile(tags_path):
            continue
        if image_id % 100 < 99:
            train_set.append((path, tags_path))
        else:
            test_set.append((path, tags_path))
    return DeepDanbooruDataset(train_set), DeepDanbooruDataset(test_set)

batch_size = 128
train_dataset, eval_dataset = train_test_sets("/root/anime-collection/images")
train_dataset_len = len(train_dataset)
print(f"Train size: {train_dataset_len}\nTest size: {len(eval_dataset)}")
train_steps_per_epoch = train_dataset_len // batch_size
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), generator=torch.Generator().manual_seed(42))
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, num_workers=os.cpu_count(), generator=torch.Generator().manual_seed(42))

learning_rate = 5e-5
weight_decay = 1e-5

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
if optim_sd is not None:
    print("Found previous training optimizer state too, resuming...")
    optimizer.load_state_dict(optim_sd)
for group in optimizer.param_groups:
    group["lr"] = learning_rate
    group["initial_lr"] = learning_rate
    group["weight_decay"] = weight_decay
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 1935, 1, 1e-5)

del model_path, optim_sd

@torch.no_grad
def test():
    model.eval()
    torch.cuda.empty_cache()
    if not os.path.isdir("samples"):
        return
    image_files = os.listdir("samples")
    if len(image_files) <= 0:
        return
    images_tensor = torch.tensor([], device=device, dtype=TORCH_DTYPE)
    for file in image_files:
        images_tensor = torch.cat((images_tensor, get_image_tensor(os.path.join("samples", file))), 0)
    results = torch.sigmoid(model(images_tensor)[0]).tolist()
    tagged_images = {}
    for file, result in zip(image_files, results):
        tags = []
        for i, prob in enumerate(result):
            if prob >= 0.5:
                tags.append(model.config.id2label[i].replace("_", " "))
        tags_text = ", ".join(tags)
        tagged_images[file] = tags_text
        print(f"\nSample \"{file}\": {tags_text}")
    print()
    return tagged_images

@torch.no_grad
def evaluate():
    model.eval()
    torch.cuda.empty_cache()
    eval_loss = 0.0
    eval_correct_labels = 0
    eval_incorrect_labels = 0
    eval_sample_labels = 0

    for images, labels in tqdm.tqdm(eval_dataloader, desc="Eval"):
        images = images.squeeze(1).to(device)
        labels = labels.squeeze(1).to(device)

        outputs = model(images, labels)
        probs = torch.sigmoid(outputs[1])
        loss = outputs[0]

        eval_loss += loss.item()
        predicted_labels = probs >= 0.5
        eval_correct_labels += int(torch.logical_and(predicted_labels, labels).sum())
        eval_incorrect_labels += int(torch.logical_xor(predicted_labels, labels).sum())
        eval_sample_labels += int(labels.sum())

    eval_loss /= len(eval_dataloader)
    eval_acc = eval_correct_labels / eval_sample_labels
    eval_inacc = eval_incorrect_labels / eval_sample_labels
    tqdm.tqdm.write(f"Eval Loss: {eval_loss:.5g}, Eval Accuracy: {eval_acc:.5g}, Eval Inaccuracy: {eval_inacc:.5g}")
    torch.cuda.empty_cache()
    return eval_loss, eval_acc, eval_inacc

print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

num_epochs = 69420

def main():
    test()
    evaluate()
    for epoch in range(highest_epoch, num_epochs):
        torch.cuda.empty_cache()
        running_loss = 0.0
        running_correct_labels = 0
        running_incorrect_labels = 0
        running_sample_labels = 0
        step_count = 0
        epoch_step_count = 0
        for images, labels in tqdm.tqdm(train_dataloader, total=train_steps_per_epoch, desc="Train"):
            step_count += 1
            epoch_step_count += 1
            model.train()
            images = images.squeeze(1).to(device)
            labels = labels.squeeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(images, labels)
            probs = torch.sigmoid(outputs[1])
            loss = outputs[0]

            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            predicted_labels = probs >= 0.5
            running_correct_labels += int(torch.logical_and(predicted_labels, labels).sum())
            running_incorrect_labels += int(torch.logical_xor(predicted_labels, labels).sum())
            running_sample_labels += int(labels.sum())

            if step_count % 50 == 0:
                step_loss = running_loss / step_count
                step_acc = running_correct_labels / running_sample_labels
                step_inacc = running_incorrect_labels / running_sample_labels
                running_loss = 0.0
                running_correct_labels = 0
                running_incorrect_labels = 0
                running_sample_labels = 0
                step_count = 0
                tqdm.tqdm.write(f"Loss: {step_loss:.5g}, Accuracy: {step_acc:.5g}, Inaccuracy: {step_inacc:.5g}, LR: {scheduler.get_last_lr()[0]:.5g}")

            if epoch_step_count % 200 == 0:
                evaluate()

            if epoch_step_count >= train_steps_per_epoch:
                break

        if step_count and running_sample_labels:
            epoch_loss = running_loss / step_count
            epoch_acc = running_correct_labels / running_sample_labels
            epoch_inacc = running_incorrect_labels / running_sample_labels
        else:
            epoch_loss = step_loss
            epoch_acc = step_acc
            epoch_inacc = step_inacc
        print(f"Finished epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.5g}, Accuracy: {epoch_acc:.5g}, Inaccuracy: {epoch_inacc:.5g}")

        print("Saving model and optimizer states...")
        save_path = os.path.join(train_results_dir, f"epoch_{epoch + 1}")
        os.makedirs(save_path, exist_ok=True)
        model_save_path = os.path.join(save_path, "model")
        model.save_pretrained(model_save_path)
        image_processor.save_pretrained(model_save_path)
        torch.save(optimizer.state_dict(), os.path.join(save_path, "optim.bin"))
        print("Saved.")

        tagged_images = test()
        eval_loss, eval_acc, eval_inacc = evaluate()
        with open(os.path.join(save_path, "train_info.json"), "w", encoding="utf8") as file:
            json.dump({
                "train_loss": epoch_loss,
                "train_accuracy": epoch_acc,
                "train_inaccuracy": epoch_inacc,
                "eval_loss": eval_loss,
                "eval_accuracy": eval_acc,
                "eval_inaccuracy": eval_inacc,
                "samples": tagged_images,
            }, file, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
