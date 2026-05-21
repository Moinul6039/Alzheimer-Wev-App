import os
import random
import shutil

source_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
import os
import random
import shutil

root = os.path.dirname(os.path.abspath(__file__))
dataset_root = os.path.join(root, "dataset")

classes = [
    "MildDemented",
    "ModerateDemented",
    "NonDemented",
    "VeryMildDemented"
]

if not os.path.isdir(dataset_root):
    raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

# Determine source location: either dataset/<class> (unsplit) or dataset/train/<class>
source_root = None
source_was_train = False

# Check for class folders in dataset root
for cls in classes:
    if os.path.isdir(os.path.join(dataset_root, cls)):
        source_root = dataset_root
        break

if source_root is None:
    # Check if dataset/train contains class folders (already partially split)
    train_candidate = os.path.join(dataset_root, "train")
    if os.path.isdir(train_candidate):
        for cls in classes:
            if os.path.isdir(os.path.join(train_candidate, cls)):
                source_root = train_candidate
                source_was_train = True
                break

if source_root is None:
    raise FileNotFoundError(
        f"Couldn't find class folders under {dataset_root} or {os.path.join(dataset_root,'train')}."
    )

print(f"Using source: {source_root}")

# When source is already the train folder we will build splits into a temporary train_new
temp_train = os.path.join(dataset_root, "train_new") if source_was_train else None

for cls in classes:
    class_src = os.path.join(source_root, cls)
    if not os.path.isdir(class_src):
        print(f"Warning: class directory not found, skipping: {class_src}")
        continue

    images = [f for f in os.listdir(class_src) if os.path.isfile(os.path.join(class_src, f))]
    if not images:
        print(f"Warning: no images found for class {cls} in {class_src}")
        continue

    random.shuffle(images)

    train_size = int(0.7 * len(images))
    val_size = int(0.15 * len(images))

    train_images = images[:train_size]
    val_images = images[train_size:train_size + val_size]
    test_images = images[train_size + val_size:]

    # Create destination folders
    for folder in ["train", "val", "test"]:
        if folder == "train" and source_was_train:
            dest = os.path.join(temp_train, cls)
        else:
            dest = os.path.join(dataset_root, folder, cls)
        os.makedirs(dest, exist_ok=True)

    # Copy files
    for img in train_images:
        src = os.path.join(class_src, img)
        if source_was_train:
            dst = os.path.join(temp_train, cls, img)
        else:
            dst = os.path.join(dataset_root, "train", cls, img)
        shutil.copy(src, dst)

    for img in val_images:
        src = os.path.join(class_src, img)
        dst = os.path.join(dataset_root, "val", cls, img)
        shutil.copy(src, dst)

    for img in test_images:
        src = os.path.join(class_src, img)
        dst = os.path.join(dataset_root, "test", cls, img)
        shutil.copy(src, dst)

# If we split from an existing train folder, replace it atomically
if source_was_train and temp_train and os.path.isdir(temp_train):
    orig_train = os.path.join(dataset_root, "train")
    backup = os.path.join(dataset_root, "train_backup_for_split")
    if os.path.isdir(backup):
        shutil.rmtree(backup)
    if os.path.isdir(orig_train):
        os.rename(orig_train, backup)
    os.rename(temp_train, orig_train)
    shutil.rmtree(backup)

print("Dataset Split Complete")
