from huggingface_hub import HfApi, create_repo, upload_folder

# === Edit these variables ===
folder_path = '/ccn2/u/khaiaw/Code/baselines/dinov3/babyview/outputs/grad_accum_1/ckpt/119999/huggingface'
repo_name = 'dinov3-vitl-babyview-gradaccum1'


# == Automatically set variables ===
repo_id = f"awwkl/{repo_name}"
create_repo(repo_id, exist_ok=True)


# === Upload ===
upload_folder(folder_path=folder_path, repo_id=repo_id)
