#!/usr/bin/env bash
# Full GigaPose best-performance configuration on the custom realsense_cup dataset.
#
# This mirrors what the authors run in src/scripts/eval_bop.py (run_bop):
#   coarse -> MegaPose refine top-1 -> MegaPose refine top-5 (best).
# The MegaPose RGB refiner uses n_iterations=5 and the multi-hypothesis model
# (configs/model/refiner/rgb.yaml) by default.
#
# Run INSIDE the Docker container (entrypoint activates env gigapose-py310,
# WORKDIR=/workspace, repo + gigaPose_datasets are bind-mounted):
#   docker compose -f docker/docker-compose.yml run --rm gigapose bash run_full_cup_pipeline.sh
# or on the host: conda activate gigapose && bash run_full_cup_pipeline.sh
#
# Prereq (run once, on the host where FoundationPose/KITchen live):
#   python -m src.scripts.prepare_realsense_cup
#   python -m src.scripts.convert_scenewise_to_imagewise --input <ds>/scenewise --output <ds>/imagewise --nprocs 1
#   python -m src.scripts.convert_imagewise_to_webdataset --input <ds>/imagewise --output <ds>/test
#   python -m src.scripts.render_custom_templates custom_dataset_name=realsense_cup
set -euo pipefail

DS=realsense_cup
RUN=cup
RES=gigaPose_datasets/results/large_${RUN}

# 1. Coarse pose -> predictions/ (top-1) + ...MultiHypothesis.csv
python test.py test_dataset_name=$DS run_id=$RUN test_setting=detection

# 2. MegaPose refine, top-1 hypothesis -> refined_predictions/
python refine.py test_dataset_name=$DS run_id=$RUN test_setting=detection use_multiple=False

# 3. MegaPose refine, top-5 hypotheses (best variant) -> refined_multiple_predictions/
python refine.py test_dataset_name=$DS run_id=$RUN test_setting=detection use_multiple=True

# 4. Overlay the best (top-5 refined) result onto the RGB frames
python src/scripts/visualize_realsense_cup.py \
  "$RES/refined_multiple_predictions/large-pbrreal-rgb-mmodel_${DS}-test_${RUN}MultiHypothesis.csv"

echo "Done. Results in $RES/{predictions,refined_predictions,refined_multiple_predictions,viz}"
