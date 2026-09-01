# Bridging the English-Arabic Medical Knowledge Gap

This repository provides the code for **TLoRA (Targeted Low-Rank Adaptation)**, described in the paper ["Bridging the English-Arabic Medical Knowledge Gap: Targeted Low-Rank Adaptation via Causal Layer Selection"](https://arxiv.org/abs/2608.00207v1), accepted to Findings of EMNLP 2026. We show that Arabic medical knowledge is present in intermediate LLM representations but fails to surface at the output, and use mechanistic interpretability methods to localize this failure to a specific layer window. TLoRA restricts LoRA adaptation to that window, outperforming full-network LoRA and other baselines on Arabic medical QA. We also introduce **AraClinicDialog**, a clinician-constructed Arabic medical dialogue benchmark in MSA with validated variants across four Arabic dialects.

## Installation

### 1. Clone the repo

```
git clone https://github.com/cha-y-mae/Adaptation.git
cd Adaptation
```

### 2. Set up the environment

Create a virtual environment (Python 3.10+ recommended) and install the requirements:

```
conda create -n adaptation python=3.10 && conda activate adaptation
pip install -r requirements.txt
```

<!-- TODO (Chaimae): confirm the minimum Python version this was actually developed/tested on. -->

### 3. Set your API keys

Several baselines call external APIs, so set the corresponding environment variable(s) before running a config that uses them, e.g.:

```
export OPENAI_API_KEY=<your-key>
export ANTHROPIC_API_KEY=<your-key>
```

Local HF models (Mistral, Llama, Jais, ALLaM, Med42, BiMediX, etc.) are pulled from the Hugging Face Hub on first use instead, run `huggingface-cli login` first if any are gated.

### 4. Set your cache directory

Update the `cache_dir` field in the config YAMLs under `configs/` to point to your own Hugging Face cache directory before running.

## Repository structure

```
Adaptation/
├── configs/      # experiment config files (model, layer window, learning rate, etc.)
├── datasets/     # dataset files 
├── diagnosis/    # tuned lens probing, causal activation patching, KL-divergence profiling scripts 
├── evals/        # evaluation scripts and metrics
├── models/       # model scripts for TLoRA and baselines
├── prompts/      # system prompts used for each task
├── scripts/      # entry-point scripts for running the pipeline
├── results/      # generated outputs 
└── requirements.txt
```

## Usage

### 1. Mechanistic diagnosis

Each diagnostic method has a separate script per backbone (`_mistral` / `_llama`):

**Tuned lens** 

```
python diagnosis/tuned_lens_mistral.py --mode train --model_path mistralai/Mistral-Small-3.2-24B-Instruct-2506 --train_csv <path/to/train.csv> --lens_dir ./tuned_lens_mistral
python diagnosis/tuned_lens_mistral.py --mode eval  --model_path mistralai/Mistral-Small-3.2-24B-Instruct-2506 --csv <path/to/eval.csv> --lens_dir ./tuned_lens_mistral --out_dir ./tuned_lens_mistral_out
```

**Causal activation patching:**

```
python diagnosis/activation_patching_mistral.py --csv <path/to/data.csv> --model_path mistralai/Mistral-Small-3.2-24B-Instruct-2506 --out_dir ./activation_patching_mistral_out
```

**KL-divergence profiling** 

```
python diagnosis/probe_kl_profile.py --data_file <path/to/data.(csv|json)> --output_dir <out_dir> --model_name mistralai/Mistral-Small-3.2-24B-Instruct-2506 --l_patch 24
```

### 2. Train TLoRA

```
python models/train_lora_targeted.py \
  --train_file datasets/train/train.json \
  --val_file datasets/train/val.json \
  --model_name mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --output_dir models/lora_targeted_l24_40 \
  --lora_mode targeted_l24
```

`--lora_mode` selects the adapted layer window: `targeted_l24` is the paper's winning window (L24–L40, §4.2), `full` trains all layers (the full-network LoRA baseline), and `targeted_l01`/`targeted_l14`/`targeted_l32`/`targeted_l35`/`targeted_l01_34` are the other windows swept in the ablations.

### 3. Evaluate

Evaluation is config-driven, each config in `configs/task1/` (MCQA), `configs/task2/` (answer generation), and `configs/task3/` (dialogue) specifies a model, dataset, and output paths:

```
python scripts/run_evaluation.py configs/task1/mistral.yaml
```

This loads the dataset, runs the model, saves predictions to `output.predictions_path`, and computes metrics (accuracy for MCQA; BLEU/ROUGE/BERTScore for generation and dialogue) into `output.metrics_path`, both set inside the config.

For task2/task3 outputs, LLM-as-judge scoring is a separate step:

```
python scripts/run_judge_standalone.py --predictions_csv <path/to/predictions.csv> --metrics_json <path/to/metrics.json> --task_type {answer_generation,dialogue_completion} --judge_prompt_file prompts/judge-task2.txt
```

To run a trained TLoRA adapter directly (outside the config pipeline), use the matching per-task inference script, e.g.:

```
python models/inference_lora_task1.py \
  --test_file <path/to/test.json> \
  --instruction_file prompts/task1.txt \
  --base_model mistralai/Mistral-Small-3.2-24B-Instruct-2506 \
  --adapter_path models/lora_targeted_l24_40 \
  --output_file results/predictions/task1/preds.csv \
  --metrics_file results/metrics/task1/metrics.json
```

## Reproducing results

All headline numbers reported in the paper (Tables 1–3 and appendix Tables S18–S24) can be reproduced by running the relevant config.

## AraClinicDialog

AraClinicDialog is our clinician-constructed Arabic medical dialogue benchmark. The MSA and dialect splits are at `datasets/task3`.

## Citing this work

```bibtex
@inproceedings{abouzahir2026bridging,
      title={Bridging the {E}nglish-{A}rabic Medical Knowledge Gap: Targeted Low-Rank Adaptation via Causal Layer Selection},
      author={Abouzahir, Chaimae and Khan, Musa and Ali-Hassan, Hala and Ma, Congbo and Saleh, Khaled and Sadqi, Yousra and Mallat, Jihad and Al-Eisawi, Walid and Habash, Nizar and Shamout, Farah E.},
      booktitle={Findings of the Association for Computational Linguistics: EMNLP 2026},
      year={2026},
      url={https://openreview.net/forum?id=GLWhomy55Q},
}
```
## Contact

Please direct any questions to ca2627@nyu.edu.
