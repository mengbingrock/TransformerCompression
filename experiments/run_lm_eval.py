# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import json
import logging
import os

import lm_eval
import torch
import wandb
from lm_eval import tasks
from lm_eval import utils as lm_eval_utils
from lm_eval.api.registry import ALL_TASKS
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks

from slicegpt import gpu_utils, hf_utils, utils
from slicegpt.config import config

TASK_METRIC_MAP = {
    "mmlu_abstract_algebra": "acc,none",
    "mmlu_business_ethics": "acc,none",
    "mmlu_college_computer_science": "acc,none",
    "mmlu_college_mathematics": "acc,none",
    "mmlu_conceptual_physics": "acc,none",
    "mmlu_formal_logic": "acc,none",
    "mmlu_machine_learning": "acc,none",
    "mmlu_miscellaneous": "acc,none",
    "mmlu_philosophy": "acc,none",
    "mmlu_global_facts": "acc,none",
    "arc_challenge": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "piqa": "acc_norm,none",
    "winogrande": "acc,none",
}


def eval_arg_parser(interactive: bool = True) -> argparse.Namespace:
    initialize_tasks()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-125m",
        help="Model to load",
    )
    path_group = parser.add_mutually_exclusive_group()
    path_group.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to load the model and tokenizer from (required for local models, not required for HF models)",
    )
    path_group.add_argument(
        "--sliced-model-path",
        type=str,
        help="Path to load the model to fine-tune (sliced) and tokenizer from",
        default=None,
    )
    parser.add_argument(
        "--sparsity", type=float, default=0.0, help="A measure of how much slicing is applied (in the range [0, 1))"
    )
    parser.add_argument(
        "--round-interval",
        type=int,
        default=8,
        help="Interval for rounding the weights (the best value may depend on your hardware)",
    )
    parser.add_argument('--hf-token', type=str, default=os.getenv('HF_TOKEN', None))
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluating with lm eval harness.")
    parser.add_argument(
        "--distribute-model",
        action="store_true",
        help="Use accelerate to put the model on multiple GPUs for evaluation. It is recommended to use it for models with 30B parameters and above.",
    )
    parser.add_argument('--wandb-project', type=str, default="slicegpt-lm-eval", help="wandb project name.")
    parser.add_argument('--no-wandb', action="store_true", help="Disable wandb.")
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande"],
        choices=lm_eval_utils.MultiChoice(tasks.ALL_TASKS),
    )
    parser.add_argument('--num-fewshot', type=int, default=0, help="Number of fewshots for all tasks.")
    parser.add_argument("--save-dir", type=str, default=".", help="Path to save the lm eval results")
    return parser.parse_args() if interactive else parser.parse_args('')


def process_eval_args(args: argparse.Namespace):
    logging.info(f'Parsed arguments:')
    for arg, argv in vars(args).items():
        logging.info(f'{arg} = {argv}')


def calculate_avg_accuracy(task_names: str, results: dict) -> float:
    n_tasks = len(task_names)
    acc_cumul = sum(result.get(TASK_METRIC_MAP[task]) for task, result in results.items() if 'mmlu' not in task)

    questions_per_mmlu_task = {
        task_name: lm_eval.tasks.get_task_dict([task_name])[task_name].dataset["test"].num_rows
        for task_name in task_names
        if 'mmlu' in task_name
    }

    if not questions_per_mmlu_task:
        return acc_cumul / n_tasks

    # Calculate average accuracy for mmlu tasks, weighted by number of questions in each task
    acc_mmlu = sum(
        result.get(TASK_METRIC_MAP[task]) * questions_per_mmlu_task[task]
        for task, result in results.items()
        if 'mmlu' in task
    )
    acc_mmlu_avg = acc_mmlu / sum(questions_per_mmlu_task.values())
    wandb.log({'acc_mmlu_avg': acc_mmlu_avg})

    return (acc_cumul + acc_mmlu_avg) / (n_tasks - len(questions_per_mmlu_task) + 1)


def eval_main(args: argparse.Namespace) -> None:
    logging.info("Running SliceGPT LM eval experiment.")

    logging.info(f"PyTorch device: {config.device}")
    logging.info(f"Number of available cuda devices: {torch.cuda.device_count()}")

    try:
        wandb.init(project=args.wandb_project, config=args, mode='disabled' if args.no_wandb else None)
    except wandb.UsageError as e:
        # wandb.init will throw an error if the user is not logged in and the process is running in a non-shell
        # environment, e.g. notebook, IDE, no-shell process, etc. In this case, we want to continue without wandb.
        logging.info(f'Failed to initialize wandb: {e}, continuing without wandb')
        wandb.init(project=args.wandb_project, mode='disabled')

    if args.sliced_model_path:
        # load the sliced model
        logging.info(f"Loading sliced {args.model} model from {args.sliced_model_path} with sparsity {args.sparsity}")
        model_adapter, tokenizer = hf_utils.load_sliced_model(
            args.model,
            args.sliced_model_path,
            sparsity=args.sparsity,
            token=args.hf_token,
            round_interval=args.round_interval,
        )
    else:
        # load the original model
        logging.info(f"Loading {args.model} model")
        model_adapter, tokenizer = hf_utils.get_model_and_tokenizer(args.model, args.model_path, token=args.hf_token)
    
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = 'left'
    #model_adapter.model.resize_token_embeddings(len(tokenizer))

    # the lm eval harness ties the weights, but this should not be done for sliced models unless the lm_head was sliced
    model_adapter.model.tie_weights = lambda: None

    if args.distribute_model:
        # distribute model across available GPUs
        gpu_utils.distribute_model(model_adapter)
    else:
        model_adapter.model.to(config.device)

    # change from here, we've already got model and tokenizer

    from datasets import load_dataset
    import math
    from rouge_score import rouge_scorer
    from sklearn.metrics import precision_recall_fscore_support
    
    def generate_predictions(model, tokenizer, input_text, masks):
        model.eval()
        generated_text = input_text

        model_inputs = tokenizer([generated_text], return_tensors="pt").to("cuda")

        # Base Sparse model prediction
        if masks == None:
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                model_output = model(
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    return_dict=True
                )
        # masked model prediction
        else:
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                model_output = model(
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    return_dict=True,
                    pruning_mask = masks,
                )

        logits = model_output.logits
        next_token_logits = logits[:, -1, :]
        probabilities = torch.softmax(next_token_logits, dim=-1)

        # Get next token predictions
        next_token_id = torch.argmax(probabilities, dim=-1)
        next_token = tokenizer.decode(next_token_id[0])
        print(next_token)
        return next_token 
    def evaluate_pubmedqa(model, tokenizer, masks, dataset):
        print("Evaluating on PubMedQA dataset...")
        true_labels = []
        pred_labels = []

        for i in range(len(dataset)):
            context = " ".join(dataset[i]['CONTEXTS'])
            question = dataset[i]['QUESTION']
            gold_label = dataset[i]['final_decision'].lower()

            input_text = (
                f"The abstract of a biomedical research article is '{context}'. "
                f"Here comes a question '{question}', and please answer the question with 'yes', 'no', or 'maybe'. "
                f"The answer is '"
            )

            prediction = generate_predictions(model, tokenizer, input_text, masks)

            # Map prediction to one of the labels
            prediction = prediction.lower()
            if "yes" in prediction:
                prediction = 'yes'
            elif 'maybe' in prediction or 'ma' in prediction:
                prediction = 'maybe'
            elif 'no' in prediction:
                prediction = 'no'
            else:
                prediction = 'unknown'  # For unexpected predictions

            true_labels.append(gold_label)
            pred_labels.append(prediction)

            print(f"Sample {i+1}/{len(dataset)} | Gold: {gold_label} | Prediction: {prediction}")
        
        # Calculate precision, recall, and F1 score for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, labels=['yes', 'no', 'maybe'], average=None, zero_division=0
        )

        # Calculate macro-F1 score
        macro_f1 = f1.mean()

        # Print per-class metrics
        for i, label in enumerate(['yes', 'no', 'maybe']):
            print(f"Class '{label}': Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1 Score: {f1[i]:.4f}, Support: {support[i]}")

        print(f"\nMacro-F1 Score: {macro_f1:.4f}")
    
    import re
    def extract_message(text):
        match = re.search(r'MESSAGE:(.*)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        else:
            return text.strip()


    def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k >0: keep only top k tokens with highest probability (top-k filtering).
                top_p >0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits


    def generate_text_custom(model, tokenizer, input_ids, max_length=50, masks=None, free=False, top_k=50, top_p=0.9, temperature=1.0):
        model.eval()
        generated = input_ids
        text = input_ids[0]

        with torch.no_grad():
            past_key_values = None  # Initialize past_key_values to None
            input_ids = generated  # Initial input

            for _ in range(max_length):
                if masks is None:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True)
                else:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        outputs = model(input_ids=input_ids, past_key_values=past_key_values, use_cache=True, pruning_mask=masks)

                # Get the logits for the last generated token
                next_token_logits = outputs.logits[0, -1, :]

                # Update past_key_values for the next iteration
                past_key_values = outputs.past_key_values

                # Apply temperature scaling
                next_token_logits = next_token_logits / temperature

                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)

                # Apply softmax to get probabilities
                next_token_probs = torch.softmax(filtered_logits, dim=-1)
            
                next_token_id = torch.multinomial(next_token_probs, num_samples=1)

                # Append the generated token to the sequence
                text = torch.cat((text, next_token_id), dim=0)

                # Update input_ids to only include the newly generated token for the next iteration
                input_ids = next_token_id.unsqueeze(0)

                # Check if the generated token is the EOS token
                if next_token_id.item() == tokenizer.eos_token_id:
                    break

        return text

    def generate_summary(model, tokenizer, input_text, masks, free=False, max_length=500):
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(config.device)

        generated_ids = generate_text_custom(
            model, tokenizer, input_ids, max_length=max_length, masks=masks, free=free  # 根据需要调整 max_length
        )

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 提取摘要
        if generated_text.startswith(input_text):
            generated_summary = generated_text[len(input_text):].strip()
        else:
            generated_summary = generated_text.strip()

        return generated_summary
    
    def evaluate_healthquestionsum(model, tokenizer, dataset, masks):
        print("Evaluating on HealthQuestionSum dataset...")
        from rouge_score import rouge_scorer
        references = []
        hypotheses = []

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        for i in range(len(dataset)):
            original_question = dataset[i]['CHQ']
            reference_summary = dataset[i]['Summary']

            question = extract_message(original_question)

            input_text = (
            f"A question posted by a patient is '{question}'."
            f"The summary of the patient's question is: '"
            )

            generated_summary = generate_summary(model, tokenizer, input_text, masks)

            references.append(reference_summary)
            hypotheses.append(generated_summary)

            print(f"Sample {i+1}/{len(dataset)}")
            print(f"Question: {question}")
            print(f"Reference Summary: {reference_summary}")
            print(f"Generated Summary: {generated_summary}")
            print("-" * 50)

        # Calculate ROUGE scores
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

        for ref, hyp in zip(references, hypotheses):
            scores = scorer.score(ref, hyp)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)

        # Calculate average scores
        for key in rouge_scores:
            avg_score = sum(rouge_scores[key]) / len(rouge_scores[key]) * 100  # Convert to percentage
            print(f"Average {key} F1 Score: {avg_score:.2f}%")

    def evaluate_mednli(model, tokenizer, masks, dataset):
        print("Evaluating on MedNLI dataset...")
        acc_count_base = 0

        for i in range(len(dataset)):
            sentence1 = dataset[i]["sentence1"]
            sentence2 = dataset[i]["sentence2"]
            gold_label = dataset[i]["gold_label"]

            input_text = (
                f"Premise is '{sentence1}', "
                f"and hypothesis is '{sentence2}'. "
                f"Their relationship is '"
            )
            
            prediction_base = generate_predictions(model, tokenizer, input_text, masks)
            print('base_prediction->', prediction_base)
            #generated_text = generate_summary(model, tokenizer, input_text, masks, True, max_length=10)
            #print(generated_text)
            
            if "contr" in prediction_base:
                prediction_base = "contradiction"
            elif "ent" in prediction_base:
                prediction_base = "entailment"
            elif "neut" in prediction_base:
                prediction_base = "neutral"
            else:
                prediction_base = None

            if prediction_base == gold_label:
                acc_count_base += 1


            print(f"Sample {i+1}/{len(dataset)} | Gold: {gold_label} | Base Prediction: {prediction_base}")

        print(f"Pruned Model Accuracy: {acc_count_base / len(dataset) * 100:.2f}%")

    
    
    def evaluate_billsum(model, tokenizer, masks):
        print("Evaluating on BillSum dataset...")
        dataset = load_dataset('json', data_files='nlp_dataset_collections/BillSum/billsum_test_200.jsonl', split='train')

        references = []
        hypotheses = []

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        for i in range(len(dataset)):
            example = dataset[i]
            reference_summary = example['summary']

            input_text = (
            f"A bill text is '{example['source']}'. "
            f"Please summart this bill."
            f"The summary of the bill is"
            )

            generated_summary = generate_summary(model, tokenizer, input_text, masks, False, 600)

            references.append(reference_summary)
            hypotheses.append(generated_summary)

            print(f"Sample {i+1}/{len(dataset)}")
            print(f"Question: {example['source']}\n")
            print(f"Reference Summary: {reference_summary}\n")
            print(f"Generated Summary: {generated_summary}\n")
            print("-" * 50)

        # Calculate ROUGE scores
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

        for ref, hyp in zip(references, hypotheses):
            scores = scorer.score(ref, hyp)
            for key in rouge_scores:
                rouge_scores[key].append(scores[key].fmeasure)

        # Calculate average scores
        for key in rouge_scores:
            avg_score = sum(rouge_scores[key]) / len(rouge_scores[key]) * 100  # Convert to percentage
            print(f"Average {key} F1 Score: {avg_score:.2f}%")
            
    def compute_perplexity(model, tokenizer, dataset, masks):
        total_loss = 0.0
        total_length = 0

        model.eval()
        for example in dataset:
            with torch.no_grad():
                inputs = tokenizer(
                    example['text'],
                    return_tensors='pt',
                    truncation=True,
                    #max_length=2048  # 根据需要调整 max_length
                ).to('cuda')

                if masks == None:
                    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                        outputs = model(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            labels=inputs['input_ids']
                        )
                else:
                    with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
                        outputs = model(
                            input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            labels=inputs['input_ids'],
                            pruning_mask = masks
                        )

                loss = outputs.loss
                # 乘以标记数获取总损失
                total_loss   += loss.item() * inputs['input_ids'].size(1)
                total_length += inputs['input_ids'].size(1)

        perplexity = math.exp(total_loss / total_length)
        return perplexity

    
    
    def evaluate_perplexity_on_multilegalpile(model, tokenizer, masks):
        print("Evaluating perplexity on MultiLegalPile Dataset...")

        dataset_file = 'nlp_dataset_collections/MultiLegalPile/multilegalpile_300.jsonl'
        dataset = load_dataset('json', data_files=dataset_file, split='train')
        perplexity = compute_perplexity(model, tokenizer, dataset, masks)
        print(f"Perplexity on Harrison dataset: {perplexity:.2f}")

    def evaluate_perplexity_on_harrison(model, tokenizer, masks):
        print("Evaluating perplexity on Harrison dataset...")

        # 直接从 harrison.jsonl 文件加载数据
        dataset_file = "nlp_dataset_collections/internalMed/internalMed_test.jsonl"  # 请替换为实际路径

        # 使用 datasets 库加载数据集
        dataset = load_dataset('json', data_files=dataset_file, split='train')

        # 计算困惑度
        perplexity = compute_perplexity(model, tokenizer, dataset, masks)
        print(f"Perplexity on Harrison dataset: {perplexity:.2f}")

        
    
    def evaluate_casehold(model, tokenizer, masks):
        #dataset_file = 'nlp_dataset_collections/CaseHold/casehold_train_clean_2000.jsonl'
        #dataset = load_dataset('json', data_files=dataset_file, split='train')
        dataset = load_dataset("casehold/casehold", "all")['test']

        true_labels = []
        pred_labels = []

        for i in range(200):
            citing_prompt = dataset[i]['citing_prompt']
            holding_statements = [
                dataset[i].get(f'holding_{i}', '') for i in range(5)
            ]
            label = dataset[i]['label']
            
            # 确定索引名称
            idx_mapping = {
                "0": "first",
                "1": "second",
                "2": "third",
                "3": "fourth",
                "4": "fifth"
            }
            idx = idx_mapping.get(str(label), None)
            if idx is None:
                raise ValueError("Label out of expected range.")

            # 根据模板格式化文本
            input_text = (
                f"A citing text consisting of the context and legal citation text is '{citing_prompt}'. "
                f"Holding statement 0 is '{holding_statements[0]}', "
                f"holding statement 1 is '{holding_statements[1]}', "
                f"holding statement 2 is '{holding_statements[2]}', "
                f"holding statement 3 is '{holding_statements[3]}', "
                f"and holding statement 4 is '{holding_statements[4]}'. "
                f"The correct answer is holding statement "
            )

            #prediction = generate_predictions(model, tokenizer, input_text, masks)
            prediction = generate_summary(model, tokenizer, input_text, masks)
            # Map prediction to one of the labels

            if '0' in prediction:
                prediction = '0'
            elif '1' in prediction:
                prediction = '1'
            elif '2' in prediction:
                prediction = '2'
            elif '3' in prediction:
                prediction = '3'
            elif '4' in prediction:
                prediction = '4'

            true_labels.append(label)
            pred_labels.append(prediction)

            print(f"Sample {i+1}/{len(dataset)} | Gold: {label} | Prediction: {prediction}")

        
        # Calculate precision, recall, and F1 score for each class
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, pred_labels, labels=['0', '1', '2', '3', '4'], average=None, zero_division=0
        )

        # Calculate macro-F1 score
        macro_f1 = f1.mean()

        # Print per-class metrics
        for i, label in enumerate(['0', '1', '2', '3', '4']):
            print(f"Class '{label}': Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1 Score: {f1[i]:.4f}, Support: {support[i]}")

        print(f"\nMacro-F1 Score: {macro_f1:.4f}")

    
    
    
    
    
    def evaluate_model_on_dataset(model, tokenizer, masks, dataset_name):
        if dataset_name.lower() == 'pubmedqa':
            dataset = load_dataset(
                "json",
                data_files="nlp_dataset_collections/PubMedQA/pubMedQA_test.jsonl"
            )["train"]
            evaluate_pubmedqa(model, tokenizer, masks, dataset)
        elif dataset_name.lower() == 'mednli':
            dataset = load_dataset(
                "json",
                data_files="nlp_dataset_collections/medNLI/mli_test_v1.jsonl"
            ).remove_columns(
                ["pairID", "sentence1_parse", "sentence1_binary_parse", "sentence2_parse", "sentence2_binary_parse"]
            )["train"]
            evaluate_mednli(model, tokenizer, masks, dataset)
        elif dataset_name.lower() == 'hqs':
            dataset = load_dataset(
                "json",
                data_files="nlp_dataset_collections/HQS/HQS_test.jsonl"
            )["train"]
            evaluate_healthquestionsum(model, tokenizer, dataset, masks)
        elif dataset_name.lower() == 'harrison':
            evaluate_perplexity_on_harrison(model, tokenizer, masks)
        elif dataset_name.lower() == 'multilegalpile':
            evaluate_perplexity_on_multilegalpile(model, tokenizer, masks)
        elif dataset_name.lower() == 'casehold':
            evaluate_casehold(model, tokenizer, masks)
        elif dataset_name.lower() == 'billsum':
            evaluate_billsum(model, tokenizer, masks)
        else:
            print(f"Dataset '{dataset_name}' is not supported.")
            return
    
    # medical
    '''evaluate_model_on_dataset(model_adapter.model, tokenizer, None, "mednli")
    evaluate_model_on_dataset(model_adapter.model, tokenizer, None, "pubMedQA")
    evaluate_model_on_dataset(model_adapter.model, tokenizer, None, "hqs")
    evaluate_model_on_dataset(model_adapter.model, tokenizer, None, "harrison")'''
    
    #legal
    
    #evaluate_model_on_dataset(model_adapter.model, tokenizer, None, 'multilegalpile')
    evaluate_model_on_dataset(model_adapter.model, tokenizer, None, 'casehold')
    input('check casehold')
    evaluate_model_on_dataset(model_adapter.model, tokenizer, None, 'billsum')

    input('check billsum')

    
    ### LM Eval Harness ###
    hflm = HFLM(pretrained=model_adapter.model, tokenizer=tokenizer, batch_size=args.batch_size)

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)

    logging.info(f"Selected Tasks: {task_names}")

    for task in task_names:
        if task not in TASK_METRIC_MAP:
            raise NotImplementedError(
                f"Please specify the metric to use for {task} in TASK_METRIC_MAP. Available info {TASK_METRIC_MAP}"
            )

    results = lm_eval.simple_evaluate(hflm, tasks=task_names, num_fewshot=args.num_fewshot, batch_size=args.batch_size)[
        'results'
    ]

    logging.info(results)
    wandb.log(results)

    with open(f"{args.save_dir}/full_results_{args.num_fewshot}_shot.json", "w") as f:
        json.dump(results, f)

    metric_vals = {task: round(result.get(TASK_METRIC_MAP[task]), 4) for task, result in results.items()}
    acc_avg = calculate_avg_accuracy(task_names, results)
    metric_vals['average'] = round(acc_avg, 4)
    with open(f"{args.save_dir}/{args.num_fewshot}_shot_task_results.json", "w") as f:
        json.dump(metric_vals, f)

    wandb.log({'acc_avg': acc_avg})

    logging.info(json.dumps(metric_vals, indent=4))
    logging.info(f"Average accuracy across tasks: {acc_avg}")


if __name__ == "__main__":
    # Use the logger from lm_eval, adding a file handler to write the log to file
    logging = lm_eval_utils.eval_logger
    logging.addHandler(utils.create_file_handler(log_dir="log"))

    os.environ["WANDB__SERVICE_WAIT"] = "300"

    eval_args = eval_arg_parser()
    process_eval_args(eval_args)
    eval_main(eval_args)
