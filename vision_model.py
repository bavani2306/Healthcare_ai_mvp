from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT_DIR / "data" / "BUSI"
MODEL_PATH = ROOT_DIR / "models" / "vision_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def build_transforms():
    """
    Create training and validation image transformations.
    """

    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    validation_transform = transforms.Compose(
        [
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    return train_transform, validation_transform


def load_busi_datasets():
    """
    Load BUSI images from:

    data/BUSI/
        benign/
        malignant/
        normal/

    Mask images are ignored.
    """

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"BUSI dataset directory not found at {DATASET_DIR}."
        )

    train_transform, validation_transform = build_transforms()

    base_dataset = datasets.ImageFolder(
        DATASET_DIR
    )

    # Remove segmentation-mask files from the classification dataset.
    filtered_samples = [
        sample
        for sample in base_dataset.samples
        if "_mask" not in Path(sample[0]).stem.lower()
        and "mask" not in Path(sample[0]).stem.lower()
    ]

    if len(filtered_samples) == 0:
        raise ValueError(
            "No usable BUSI images were found."
        )

    targets = [
        target
        for _, target in filtered_samples
    ]

    indices = list(
        range(len(filtered_samples))
    )

    train_indices, validation_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=42,
        stratify=targets,
    )

    # Build datasets with different transforms.
    train_dataset = datasets.ImageFolder(
        DATASET_DIR,
        transform=train_transform,
    )

    validation_dataset = datasets.ImageFolder(
        DATASET_DIR,
        transform=validation_transform,
    )

    # Filter masks in both datasets.
    train_dataset.samples = [
        sample
        for sample in train_dataset.samples
        if "_mask" not in Path(sample[0]).stem.lower()
        and "mask" not in Path(sample[0]).stem.lower()
    ]

    validation_dataset.samples = [
        sample
        for sample in validation_dataset.samples
        if "_mask" not in Path(sample[0]).stem.lower()
        and "mask" not in Path(sample[0]).stem.lower()
    ]

    train_dataset.targets = [
        target
        for _, target in train_dataset.samples
    ]

    validation_dataset.targets = [
        target
        for _, target in validation_dataset.samples
    ]

    train_subset = Subset(
        train_dataset,
        train_indices,
    )

    validation_subset = Subset(
        validation_dataset,
        validation_indices,
    )

    return (
        train_subset,
        validation_subset,
        base_dataset.classes,
    )


def create_model(num_classes: int):
    """
    Create a pretrained ResNet18 and replace its final layer.

    Only the final classifier layer is trained.
    """

    weights = models.ResNet18_Weights.DEFAULT

    model = models.resnet18(
        weights=weights
    )

    # Freeze the pretrained feature extractor.
    for parameter in model.parameters():
        parameter.requires_grad = False

    number_of_features = model.fc.in_features

    model.fc = nn.Linear(
        number_of_features,
        num_classes,
    )

    return model


def train_model() -> dict:
    """
    Train the ResNet18 classifier on BUSI.
    """

    train_dataset, validation_dataset, class_names = (
        load_busi_datasets()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = create_model(
        num_classes=len(class_names)
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE,
    )

    print(f"Using device: {DEVICE}")
    print(f"Classes: {class_names}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")

    for epoch in range(EPOCHS):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            loss.backward()
            optimizer.step()

            running_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = outputs.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = (
            running_loss / total
            if total
            else 0.0
        )

        train_accuracy = (
            correct / total
            if total
            else 0.0
        )

        # Validation
        model.eval()

        validation_correct = 0
        validation_total = 0

        with torch.no_grad():
            for images, labels in validation_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)

                predictions = outputs.argmax(
                    dim=1
                )

                validation_correct += (
                    predictions == labels
                ).sum().item()

                validation_total += labels.size(0)

        validation_accuracy = (
            validation_correct
            / validation_total
            if validation_total
            else 0.0
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.4f} | "
            f"Validation Accuracy: {validation_accuracy:.4f}"
        )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "image_size": IMAGE_SIZE,
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    print(
        f"\nVision model saved to: {MODEL_PATH}"
    )

    return checkpoint


def load_model():
    """
    Load the trained ResNet18 model.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Vision model not found at {MODEL_PATH}.\n"
            "Run:\n"
            "python modules/vision_model.py"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    class_names = checkpoint[
        "class_names"
    ]

    model = create_model(
        num_classes=len(class_names)
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    _, validation_transform = build_transforms()

    return (
        model,
        class_names,
        validation_transform,
    )


def predict_image(image) -> dict:
    """
    Predict the class of a PIL image or
    Streamlit UploadedFile.
    """

    if image is None:
        raise ValueError(
            "No image was provided."
        )

    model, class_names, transform = load_model()

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")

    tensor = transform(image)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1,
        )

        confidence, predicted_index = (
            probabilities.max(dim=1)
        )

    predicted_class = class_names[
        predicted_index.item()
    ]

    return {
        "predicted_class": predicted_class,
        "confidence": float(
            confidence.item()
        ),
    }


if __name__ == "__main__":
    train_model()