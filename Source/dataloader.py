"""
src/dataloader.py
Load the images using PyTorch DataLoader(). Default path to /kaggle/input/datasets/cminhhuymai/iitd-iris/
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path


class IITDIrisDataset(Dataset):
    def __init__(
        self, root_dir="/kaggle/input/datasets/cminhhuymai/iitd-iris/", transform=None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []

        # Find all subject subdirectories (e.g., '001', '028', etc.)
        # The ls output showed 221 items, so we sort them to ensure consistent label mapping
        subject_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name != "Normalized_Images"])
        self.class_to_idx = {d.name: idx for idx, d in enumerate(subject_dirs)}

        # Populate the samples list with (file_path, subject_label, eye_side)
        for subject_dir in subject_dirs:
            label = self.class_to_idx[subject_dir.name]

            for img_path in subject_dir.glob("*.bmp"):
                # Extract 'L' or 'R' from filenames like '01_L.bmp'
                eye_side = img_path.stem.split("_")[-1]
                self.samples.append((img_path, label, eye_side))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, subject_label, eye_side = self.samples[idx]

        # Iris images are typically grayscale, so we convert to "L"
        # Change to "RGB" if your specific network architecture requires 3 channels
        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)

        # Convert the string 'L'/'R' into a binary label (0 for Left, 1 for Right)
        side_label = 0 if eye_side == "L" else 1

        return image, subject_label, side_label


if __name__ == "__main__":
    # Define standard transformations
    transform = transforms.Compose(
        [
            # transforms.Resize(
            #     (224, 224)
            # ),  # Adjust size based on your model requirements
            transforms.ToTensor(),
            # Add transforms.Normalize here if needed
        ]
    )

    # Set the path to the parent directory containing all the subject folders
    data_path = "/kaggle/input/datasets/cminhhuymai/iitd-iris"

    # Initialize the dataset
    iris_dataset = IITDIrisDataset(root_dir=data_path, transform=transform)

    # Wrap it in a DataLoader for batching and multiprocessing
    train_loader = DataLoader(
        iris_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2,  # Adjust based on the Kaggle environment's CPU core count
        pin_memory=True,  # Speeds up transfer to GPU
    )

    # Test the dataloader output
    for idx, (images, subject_labels, side_labels) in enumerate(train_loader):
        print(f"Batch image shape: {images.shape}")
        print(f"Subject labels: {subject_labels}")
        print(f"Eye side labels (0=L, 1=R): {side_labels}")

        img_array = images[0].squeeze().numpy()

        # Scale to 0-255 and convert to 8-bit unsigned integer
        img_array = (img_array * 255).astype("uint8")

        # Create PIL Image (will automatically be mode 'L' due to uint8 input)
        img = Image.fromarray(img_array)

        img.save(f"outputs/{idx}_iris.png")
        if idx == 5:
            break  # Just printing the first batch to verify