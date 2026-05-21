# HOPE Download Installation Guide

This note captures the steps already taken to get the BOP 2024 HOPE dataset download working in this workspace.

## Steps Already Taken

1. Started the project Docker container with `sudo docker start docker-gigapose-1`.
2. Installed or updated `huggingface_hub[cli]` inside the container.
3. Attempted to download the dataset with `python -m src.scripts.download_test_bop24 test_dataset_name=hope`.
4. Hit an empty split / conversion failure in `convert_imagewise_to_webdataset.py` because the input directory had no imagewise files.
5. Added checks so the conversion scripts fail early with a clear error when the input split is empty.
6. Fixed a bug in `convert_scenewise_to_imagewise.py` where the output directory was created with an invalid `Path(..., exist_ok=True)` call.
7. Logged into Hugging Face with a valid token after seeing the authentication failure.
8. Discovered that `bop-benchmark/datasets` is not the correct Hugging Face repository for HOPE.
9. Updated `download_test_bop24.py` to download from the dataset-specific repository `bop-benchmark/hope` instead of the aggregate repo.

## Installation Instructions

1. Activate the Gigapose environment inside the container.
2. Make sure Hugging Face CLI support is installed:

   ```bash
   pip install -U "huggingface_hub[cli]"
   ```

3. Log in to Hugging Face with a read token:

   ```bash
   hf auth login
   ```

   If needed, paste a token generated from your Hugging Face settings page.

4. Download the HOPE dataset:

   ```bash
   export DATASET_NAME=hope
   python -m src.scripts.download_test_bop24 test_dataset_name=$DATASET_NAME
   ```

5. The BOP24 downloader now fetches the top-level `*.zip` archives from dataset-specific Hugging Face repos such as `bop-benchmark/hope`, `bop-benchmark/handal`, and `bop-benchmark/hot3d`.

6. The extractor now accepts zip files stored either directly under the dataset folder or at the repo root, so the download layout can vary without breaking the flow.

7. If the script reports that the split is empty, verify that the Hugging Face account has access to the selected repo and that the token is still valid.

8. For BOP23 datasets, run:

   ```bash
   python -m src.scripts.download_test_bop23
   ```

   This script now also downloads from the dataset-specific Hugging Face repos and unpacks the dataset zips locally before converting them to WebDataset format.

9. If BOP23 conversion fails with an empty split, it usually means the downloaded repo contained no `*.zip` files for that dataset or the repo name is wrong.

## Notes

- The scripts no longer depend on the older aggregate Hugging Face repo path.
- The scripts raise a clear error when the expected archive files or extracted split files are missing, which usually indicates an auth or repo-layout problem.
- If you are running inside Docker and see a torch import error related to `iJIT_NotifyEvent`, the container may also need `LD_PRELOAD=/opt/conda/envs/gigapose/lib/libiomp5.so` set in the container environment.