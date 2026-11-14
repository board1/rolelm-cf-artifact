"""Reconstruct the saved CF adaptation from pinned base and unchanged adapter."""
import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    target = Path(args.output_dir)
    if target.exists() and any(target.iterdir()):
        raise RuntimeError('Output directory must be absent or empty')
    repo = 'HeAAAAA/RoleRM'
    revision = '52ae4bf5ba2616ec48834f8467ee1af88443161a'
    base = AutoModelForCausalLM.from_pretrained(
        repo, revision=revision, torch_dtype=torch.float32,
        device_map={'': 'cpu'}, trust_remote_code=True)
    adapted = PeftModel.from_pretrained(
        base, str(Path(__file__).resolve().parent / 'adapter'))
    merged = adapted.merge_and_unload(safe_merge=True)
    merged.to(dtype=torch.bfloat16)
    merged.save_pretrained(target, safe_serialization=True)
    AutoTokenizer.from_pretrained(repo, revision=revision).save_pretrained(target)

if __name__ == '__main__':
    main()
