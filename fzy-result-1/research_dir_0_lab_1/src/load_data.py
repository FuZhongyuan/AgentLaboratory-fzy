from datasets import load_dataset, Image
from torchvision.transforms import Compose, Pad, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize
from torch.utils.data import DataLoader

# 1. Load CIFAR-10 and ensure 'img' column yields PIL Images
ds = load_dataset("cifar10")
ds = ds.cast_column("img", Image())

# 2. Define simple augmentation + normalization
transform = Compose([
    Pad(4),
    RandomCrop(32),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.4914, 0.4822, 0.4465),
              (0.2470, 0.2435, 0.2616))
])

# 3. Batch‐wise preprocessing: create 'pixel_values' and rename 'label'→'labels'
def preprocess(batch):
    batch["pixel_values"] = [transform(img) for img in batch["img"]]
    batch["labels"] = batch["label"]
    return batch

ds = ds.map(preprocess, batched=True, batch_size=1000,
            remove_columns=["img", "label"])

# 4. Set PyTorch tensor format
ds.set_format(type="torch", columns=["pixel_values", "labels"])

# 5. Split into train/test and build DataLoaders
train_ds, test_ds = ds["train"], ds["test"]
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=128, shuffle=False)

# 6. Quick sanity checks
print("Train batches:", len(train_loader), "Test batches:", len(test_loader))
batch = next(iter(train_loader))
print("Batch shapes → pixel_values:", batch["pixel_values"].shape,
      "labels:", batch["labels"].shape)