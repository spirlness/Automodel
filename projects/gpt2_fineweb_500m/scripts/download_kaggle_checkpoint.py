# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Monitor and download latest checkpoint from Kaggle kernel runs."""

import argparse
import sys
import time
from pathlib import Path

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("❌ Error: kaggle package not installed. Run `uv pip install kaggle`.")
    sys.exit(1)


def main() -> None:
    """CLI entrypoint to check status and download kernel output."""
    parser = argparse.ArgumentParser(description="Download latest checkpoint from Kaggle kernel")
    parser.add_argument(
        "--kernel",
        type=str,
        default="hwbwhzbsh/automodel-gpt2-fineweb-t4x2",
        help="Kernel identifier (owner/kernel-name)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "checkpoints" / "gpt2_kaggle"),
        help="Directory to save downloaded files",
    )
    parser.add_argument(
        "--poll",
        action="store_true",
        help="Continuously poll until kernel run completes, then download",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1200,
        help="Polling interval in seconds (default: 1200, i.e. 20 minutes)",
    )
    args = parser.parse_args()

    api = KaggleApi()
    api.authenticate()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Checking status for kernel: {args.kernel}")

    while True:
        status_obj = api.kernels_status(args.kernel)
        status_str = getattr(status_obj, "status", None) or str(status_obj)
        print(f"⏱️ [{time.strftime('%Y-%m-%d %H:%M:%S')}] Status: {status_str}")

        if "RUNNING" in str(status_str).upper() or "QUEUED" in str(status_str).upper():
            if not args.poll:
                print("\n⚠️ Note: The Kaggle kernel is currently RUNNING.")
                print("   Kaggle only exposes output artifacts via API once the run finishes (COMPLETE).")
                print("   Run with `--poll` flag to automatically wait and download when finished:")
                print(f"   python {sys.argv[0]} --poll\n")
                break
            else:
                print(f"   Waiting {args.interval}s before next check...")
                time.sleep(args.interval)
        elif "COMPLETE" in str(status_str).upper():
            print(f"\n🎉 Kernel execution COMPLETE! Downloading output to {out_path}...")
            api.kernels_output(args.kernel, path=str(out_path))
            print(f"✅ Successfully downloaded kernel outputs to: {out_path}")
            print("Downloaded files:")
            for p in out_path.glob("**/*"):
                if p.is_file():
                    print(f" - {p.relative_to(out_path)} ({p.stat().st_size / (1024 * 1024):.2f} MB)")
            break
        else:
            print(f"\n⚠️ Kernel status is: {status_str}")
            print(f"Attempting to download available outputs to {out_path}...")
            api.kernels_output(args.kernel, path=str(out_path))
            break


if __name__ == "__main__":
    main()
