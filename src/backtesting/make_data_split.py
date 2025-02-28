import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def split_dataset(
    input_filepath, output_dir, val_percent, test_percent, random_seed=42
):
    """
    Split a parquet dataset into train, validation, and test sets.

    Parameters:
    -----------
    input_filepath : str
        Path to the input parquet file
    output_dir : str
        Directory to save the output files
    val_percent : float
        Percentage of data to use for validation set (0-1)
    test_percent : float
        Percentage of data to use for test set (0-1)
    random_seed : int
        Random seed for reproducibility
    """
    print(f"Loading dataset from {input_filepath}")
    df = pd.read_parquet(input_filepath)

    # Ensure the percentages are valid
    if val_percent + test_percent >= 1.0:
        raise ValueError("Sum of validation and test percentages must be less than 1.0")

    print(f"Dataset contains {len(df)} examples")

    # Calculate train percentage
    train_percent = 1.0 - (val_percent + test_percent)

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, test_size=test_percent, random_state=random_seed
    )

    # Second split: separate validation set from remaining data
    # Calculate new validation percentage relative to the remaining data
    adjusted_val_percent = val_percent / (train_percent + val_percent)

    train_df, val_df = train_test_split(
        train_val_df, test_size=adjusted_val_percent, random_state=random_seed
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save datasets
    train_output = os.path.join(output_dir, "train.parquet")
    val_output = os.path.join(output_dir, "val.parquet")
    test_output = os.path.join(output_dir, "test.parquet")

    train_df.to_parquet(train_output, index=False)
    val_df.to_parquet(val_output, index=False)
    test_df.to_parquet(test_output, index=False)

    # Print statistics
    print(f"Split complete:")
    print(f"  Train set: {len(train_df)} examples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation set: {len(val_df)} examples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test set: {len(test_df)} examples ({len(test_df)/len(df)*100:.1f}%)")
    print(f"Output saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a Manifold dataset into train, validation, and test sets"
    )
    parser.add_argument(
        "--input_filepath",
        type=str,
        required=True,
        help="Path to the input parquet file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="parquet_data",
        help="Directory to save the output files",
    )
    parser.add_argument(
        "--val_percent",
        type=float,
        default=0.1,
        help="Percentage of data to use for validation set (0-1)",
    )
    parser.add_argument(
        "--test_percent",
        type=float,
        default=0.1,
        help="Percentage of data to use for test set (0-1)",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Validate percentage inputs
    if not (0 < args.val_percent < 1):
        parser.error("Validation percentage must be between 0 and 1")
    if not (0 < args.test_percent < 1):
        parser.error("Test percentage must be between 0 and 1")
    if args.val_percent + args.test_percent >= 1.0:
        parser.error("Sum of validation and test percentages must be less than 1.0")

    split_dataset(
        input_filepath=args.input_filepath,
        output_dir=args.output_dir,
        val_percent=args.val_percent,
        test_percent=args.test_percent,
        random_seed=args.random_seed,
    )
