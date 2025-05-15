# training/gan_augmentor.py

import torch
import logging
from models.gan_synthesizer import Generator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def generate_synthetic_samples(
    gan_checkpoint_path: str,
    noise_dim: int = 32,
    output_dim: int = 32,
    num_samples: int = 1000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    """
    Generate synthetic feature samples using pre-trained GAN Generator.
    """

    logger.info("Generating synthetic samples using GAN...")

    # 1. Load Generator
    generator = Generator(noise_dim=noise_dim, output_dim=output_dim)
    generator.load_state_dict(torch.load(gan_checkpoint_path, map_location=device))
    generator.to(device)
    generator.eval()

    # 2. Sample noise
    z = torch.randn(num_samples, noise_dim).to(device)

    # 3. Generate fake samples
    with torch.no_grad():
        synthetic_features = generator(z)  # (num_samples, output_dim)

    return synthetic_features.cpu()

def augment_dataset_with_synthetic(
    real_dataset,
    synthetic_features: torch.Tensor,
    target_label: dict = None
):
    """
    Augment real dataset with synthetic applicants.
    target_label: Dictionary specifying labels for synthetic data (e.g., risky borrowers)
    """

    logger.info(f"Augmenting dataset with {len(synthetic_features)} synthetic samples...")

    synthetic_data = []

    for i in range(len(synthetic_features)):
        sample = {
            "x_seq": synthetic_features[i].view(1, -1).repeat(10, 1),  # Repeat to match sequence format
            "static_score": torch.tensor([0.5]),  # Neutral static score
            "additional_features": torch.randn(5)
        }

        # Assign target labels (you can adjust)
        if target_label:
            for key, value in target_label.items():
                sample[key] = torch.tensor([value], dtype=torch.float32)
        else:
            sample.update({
                "pd": torch.tensor([0.5]),
                "ead": torch.tensor([0.5]),
                "lgd": torch.tensor([0.5]),
                "fraud": torch.tensor([0.5]),
                "loan_limit": torch.rand(1) * 1e8
            })

        synthetic_data.append(sample)

    augmented_dataset = torch.utils.data.ConcatDataset([real_dataset, synthetic_data])

    logger.info(f"Total dataset size after augmentation: {len(augmented_dataset)}")

    return augmented_dataset

if __name__ == "__main__":
    # Example dummy test
    class DummyRealDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 500

        def __getitem__(self, idx):
            return {
                "x_seq": torch.randn(10, 32),
                "static_score": torch.randn(1),
                "additional_features": torch.randn(5),
                "pd": torch.randint(0, 2, (1,), dtype=torch.float32),
                "ead": torch.rand(1),
                "lgd": torch.rand(1),
                "fraud": torch.randint(0, 2, (1,), dtype=torch.float32),
                "loan_limit": torch.rand(1) * 1e8
            }

    real_dataset = DummyRealDataset()
    fake_features = torch.randn(100, 32)

    augmented_dataset = augment_dataset_with_synthetic(real_dataset, fake_features)
