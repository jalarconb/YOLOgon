import os
import shutil
import random


def split_data(
    source_img,
    source_lbl,
    dest_img_train,
    dest_img_val,
    dest_lbl_train,
    dest_lbl_val,
    split_ratio=0.8,
):
    os.makedirs(dest_img_train, exist_ok=True)
    os.makedirs(dest_img_val, exist_ok=True)
    os.makedirs(dest_lbl_train, exist_ok=True)
    os.makedirs(dest_lbl_val, exist_ok=True)

    images = [
        f for f in os.listdir(source_img) if f.endswith((".jpg", ".png", ".jpeg"))
    ]
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)

    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for img in train_images:
        shutil.copy(os.path.join(source_img, img), os.path.join(dest_img_train, img))
        lbl = (
            img.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt")
        )
        shutil.copy(os.path.join(source_lbl, lbl), os.path.join(dest_lbl_train, lbl))

    for img in val_images:
        shutil.copy(os.path.join(source_img, img), os.path.join(dest_img_val, img))
        lbl = (
            img.replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt")
        )
        shutil.copy(os.path.join(source_lbl, lbl), os.path.join(dest_lbl_val, lbl))

    print("Succesfully split the dataset into training and validation sets.")


if __name__ == "__main__":
    split_data(
        "raw_data/images/with_glasses",
        "raw_data/labels/with_glasses",
        "train_data/images/train",
        "train_data/images/val",
        "train_data/labels/train",
        "train_data/labels/val",
    )

    split_data(
        "raw_data/images/no_glasses",
        "raw_data/labels/no_glasses",
        "train_data/images/train",
        "train_data/images/val",
        "train_data/labels/train",
        "train_data/labels/val",
    )
