"""
main.py - Master Pipeline Runner for Fashion Product Classification
Unified CLI interface for training, evaluation, comparison, prediction, and GUI.
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from src.dataset import CLASS_NAMES, CLASS_DESCRIPTIONS, export_sample_images
from src.models import print_architecture_comparison
from src.predict import predict_fashion_item


def run_pipeline_demo():
    """
    Runs an end-to-end automated demonstration of the Fashion Product Classification system.
    """
    print("\n" + "="*80)
    print("      FASHION PRODUCT CLASSIFICATION - COMPLETE PIPELINE DEMO")
    print("="*80)
    print("Dataset: Fashion-MNIST (10 clothing & accessory categories)")
    print("Supported Architectures: Artificial Neural Network (ANN) & Convolutional Neural Network (CNN)")
    print("Libraries: TensorFlow, Keras, NumPy, Pillow\n")

    # 1. Ensure sample images exist
    samples_dir = os.path.join(PROJECT_ROOT, "data", "samples")
    print(f"[Step 1/4] Ensuring test images exist in: {samples_dir}")
    if not os.path.exists(samples_dir) or len(os.listdir(samples_dir)) < 10:
        export_sample_images(output_dir=samples_dir)
    else:
        print("           Sample images already prepared.")

    # 2. Architecture Comparison
    print("\n[Step 2/4] Displaying Deep Learning Architecture Comparison:")
    print_architecture_comparison()

    # 3. Model Benchmark Comparison
    print("\n[Step 3/4] Running Comparative Evaluation (ANN vs CNN):")
    from compare_models import compare_architectures
    compare_architectures(output_dir=os.path.join(PROJECT_ROOT, "outputs"))

    # 4. Multi-Category Inference Demo
    print("\n[Step 4/4] Performing Live Inference on Distinct Fashion Items:")
    demo_samples = [
        ("Bag", "sample_bag.png"),
        ("Shirt", "sample_shirt.png"),
        ("Coat", "sample_coat.png"),
        ("Trouser", "sample_trouser.png"),
        ("Sneaker", "sample_sneaker.png"),
    ]

    for label, fname in demo_samples:
        img_path = os.path.join(samples_dir, fname)
        if os.path.exists(img_path):
            results, _ = predict_fashion_item(img_path, top_k=2)
            top = results[0]
            print(f"  * Ground Truth: {label:<10} -> Top Prediction: {top['category']:<12} ({top['percentage']}) | {top['description']}")

    print("\n" + "="*80)
    print(" Demo Complete! To launch the interactive desktop GUI, run:")
    print("     python main.py --mode gui")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Fashion Product Classification System (Fashion-MNIST with TF/Keras)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode demo                     # Run complete automated pipeline demo
  python main.py --mode train --model cnn        # Train CNN model
  python main.py --mode train --model ann        # Train ANN model
  python main.py --mode evaluate --model cnn     # Evaluate model performance
  python main.py --mode compare                  # Compare ANN vs CNN performance
  python main.py --mode predict --image path.png # Predict single image category
  python main.py --mode gui                      # Launch interactive desktop app
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "train", "evaluate", "compare", "predict", "gui"],
        help="Pipeline mode to execute (default: demo)"
    )
    parser.add_argument("--model", type=str, default="cnn", choices=["ann", "cnn"], help="Architecture choice (ann or cnn)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--image", type=str, default=None, help="Path to image for prediction mode")
    parser.add_argument("--save-vis", type=str, default="outputs/prediction_result.png", help="Path to save prediction plot")

    args = parser.parse_args()

    if args.mode == "demo":
        run_pipeline_demo()

    elif args.mode == "train":
        from src.train import train_model
        train_model(
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            save_dir=os.path.join(PROJECT_ROOT, "models"),
            output_dir=os.path.join(PROJECT_ROOT, "outputs")
        )

    elif args.mode == "evaluate":
        from src.evaluate import evaluate_model
        model_path = os.path.join(PROJECT_ROOT, "models", f"fashion_{args.model}_best.keras")
        evaluate_model(model_path_or_name=model_path, output_dir=os.path.join(PROJECT_ROOT, "outputs"))

    elif args.mode == "compare":
        from compare_models import compare_architectures
        compare_architectures(output_dir=os.path.join(PROJECT_ROOT, "outputs"))

    elif args.mode == "predict":
        if not args.image:
            # Fallback to sample bag if no image provided
            default_sample = os.path.join(PROJECT_ROOT, "data", "samples", "sample_bag.png")
            if os.path.exists(default_sample):
                args.image = default_sample
                print(f"[main] No image specified, using default sample: {default_sample}")
            else:
                print("[ERROR] Please provide --image <path_to_image>")
                sys.exit(1)

        model_path = os.path.join(PROJECT_ROOT, "models", f"fashion_{args.model}_best.keras")
        results, _ = predict_fashion_item(
            image_input=args.image,
            model_path=model_path if os.path.exists(model_path) else None,
            save_vis=args.save_vis
        )
        print("\n" + "="*50)
        print(f"Prediction for: {os.path.basename(args.image)}")
        print("="*50)
        for rank, res in enumerate(results, 1):
            print(f"Top {rank}: {res['category']:<15} | Confidence: {res['percentage']} | {res['description']}")
        print("="*50 + "\n")

    elif args.mode == "gui":
        from app import launch_gui
        launch_gui()


if __name__ == "__main__":
    main()
