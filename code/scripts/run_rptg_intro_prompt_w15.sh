#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
CODE_DIR="${ROOT}/code"
MODEL_PATH="${ROOT}/hf_cache"
OUTPUT_ROOT="${ROOT}/results/rptg_intro_prompt_w15_candidates"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PROMPT_IDS=(
  "glass_blue_hummingbird"
  "porcelain_blue_bird"
  "neon_blue_kingfisher"
  "white_marble_blue_bird"
  "blue_bird_chrome_studio"
  "tropical_blue_bird"
  "porcelain_teapot_roses"
  "glass_perfume_neon"
  "blue_ceramic_vase_gold"
  "white_sports_car_neon"
  "crystal_cake_display"
  "chrome_robot_toy"
  "marble_bathroom_flowers"
  "blue_sneaker_product"
)

SEEDS=(
  20270625
  20270626
  20270627
  20270628
  20270629
  20270630
  20270631
  20270632
  20270633
  20270634
  20270635
  20270636
  20270637
  20270638
)

PROMPTS=(
  "A small electric-blue hummingbird made of translucent glass, glowing under intense sunlight on a white marble surface, surrounded by saturated red flowers and shiny golden leaves, strong reflections, high contrast macro photography, ultra detailed"
  "A white porcelain bird statue with glossy cobalt-blue painted wings, standing on a mirror-like wet table under harsh studio spotlights, surrounded by bright red roses and golden ornaments, strong reflections, high contrast, realistic product photography, ultra detailed"
  "A vivid cobalt-blue kingfisher bird perched on a wet black branch, bright orange chest, glossy feathers, emerald forest background, sparkling water droplets, strong rim light, high contrast wildlife photography, ultra detailed"
  "A tiny blue songbird standing on a bright white marble fountain at noon, wet feathers, reflective water, saturated red flowers in the background, harsh sunlight, strong highlights, realistic telephoto wildlife photography"
  "A metallic blue bird sculpture with chrome feathers on a glossy white studio floor, surrounded by red glass spheres and golden ribbons, intense spotlights, mirror reflections, high contrast product photography, sharp focus"
  "A tiny electric-blue bird with iridescent feathers perched beside bright red berries and yellow flowers, rain droplets on feathers, saturated tropical garden, strong sunlight, macro wildlife photography, extremely detailed, sharp focus"
  "A glossy white porcelain teapot with cobalt-blue floral patterns on a mirror-like wet table, surrounded by bright red roses and golden teaspoons, harsh studio spotlights, strong reflections, high contrast product photography, ultra detailed"
  "A transparent glass perfume bottle filled with electric-blue liquid on a glossy white surface, surrounded by red silk ribbons and golden jewelry, neon reflections, intense studio lighting, sharp product photography, ultra detailed"
  "A cobalt-blue ceramic vase with white porcelain flowers and polished gold trim, placed on a reflective black table under harsh spotlights, saturated red background, strong highlights, luxury product photography, ultra detailed"
  "A glossy white sports car with cobalt-blue racing stripes parked on wet pavement at night, saturated neon signs reflected across the body, bright headlights, red and gold city lights, cinematic high contrast photography"
  "A crystal wedding cake display on a white marble table, blue sugar flowers, red strawberries, golden candles, mirror reflections, intense daylight from a window, high contrast realistic food photography, ultra detailed"
  "A small chrome robot toy with cobalt-blue glass eyes standing on a glossy white desk, surrounded by red toy blocks and golden coins, harsh studio lighting, strong reflections, realistic product photography, ultra detailed"
  "A luxurious white marble bathroom sink filled with cobalt-blue water, red rose petals, golden faucet, mirror reflections, bright sunlight, clean interior photography, high contrast, ultra detailed"
  "A cobalt-blue running sneaker on a glossy white acrylic platform, red laces, golden logo, wet reflective floor, harsh studio spotlight, high contrast commercial product photography, ultra detailed"
)

cd "${ROOT}"

for index in "${!PROMPT_IDS[@]}"; do
  PROMPT_ID="${PROMPT_IDS[$index]}"
  SEED="${SEEDS[$index]}"
  PROMPT="${PROMPTS[$index]}"
  OUTPUT_DIR="${OUTPUT_ROOT}/${PROMPT_ID}"

  echo "Running ${PROMPT_ID} with seed ${SEED}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 "${CODE_DIR}/generation/generate_rptg_intro_prompt.py" \
    --output_dir "${OUTPUT_DIR}" \
    --model_path "${MODEL_PATH}" \
    --prompt_id "${PROMPT_ID}" \
    --prompt "${PROMPT}" \
    --seed "${SEED}" \
    --preset compact \
    --cfg_scale 15 \
    --num_steps 20 \
    --shift 3.0 \
    --frequency_cutoff 0.25 \
    --high_radial_scale 0.25 \
    --radial_band low \
    --radial_direction positive_radial \
    --low_cfg_scale 1.0 \
    --adg_max_angle 1.0471975512 \
    --apg_r 1.5 \
    --apg_beta -0.2 \
    --fdg_kernel_size 5 \
    --fdg_sigma 1.0
done

echo "Intro prompt candidate outputs saved to ${OUTPUT_ROOT}"
