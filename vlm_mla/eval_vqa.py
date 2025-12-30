"""
VQAv2 Evaluation Script for Vision-Language Models
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Import your MLA loader
try:
    from .load import load_llava_with_optional_mla
except:
    from load import load_llava_with_optional_mla


@dataclass
class EvalConfig:
    """Evaluation configuration for VQAv2"""
    model_path: str
    split: str = "validation"
    max_samples: Optional[int] = None
    batch_size: int = 8
    max_new_tokens: int = 32
    device: str = "cuda"
    dtype: str = "bfloat16"
    output_dir: str = "./eval_results"
    save_predictions: bool = True
    analyze_errors: bool = True
    num_workers: int = 4
    

@dataclass
class VQAResult:
    """Single VQA prediction result"""
    question_id: str
    question: str
    prediction: str
    prediction_short: str
    ground_truth: List[str]
    accuracy: float
    is_correct: bool
    

class VQAEvaluator:
    """VQAv2 evaluator"""
    
    # VQAv2 dataset configuration
    DATASET_PATH = "lmms-lab/VQAv2"
    VALID_SPLITS = ["validation", "test"]
    
    def __init__(self, config: EvalConfig):
        self.config = config
        self.results: List[VQAResult] = []
        
        # Setup output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Load model
        print(f"Loading model from {config.model_path}...")
        self.model, self.processor = load_llava_with_optional_mla(
            model_name_or_path=config.model_path,
            torch_dtype=self._get_dtype(config.dtype),
            device=config.device,
            trust_remote_code=True,
        )
        
        tokenizer = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        if "<image>" not in tokenizer.vocab:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
        self.model.resize_token_embeddings(len(tokenizer))
        
        self.model.eval()
        
        # Validate split
        if config.split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split: {config.split}. Must be one of {self.VALID_SPLITS}")
    
    @staticmethod
    def _get_dtype(name: str) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(name, torch.bfloat16)
    
    @staticmethod
    def _normalize_answer(text: str) -> str:
        """Normalize answer for comparison"""
        text = text.strip().lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        return text
    
    @staticmethod
    def _postprocess_answer(text: str) -> str:
        """Extract short answer from model generation"""
        text = text.strip()
        
        if not text:
            return "unknown"
        
        # Take first line
        text = text.split('\n')[0].strip()
        
        # Remove common prefixes
        text = re.sub(r"^(assistant|ASSISTANT|answer|Answer)\s*[:：]?\s*", "", text)
        
        # Extract "the answer is X" pattern
        match = re.search(r"(?:the answer is|answer is)\s+(.+?)(?:[.!?]|$)", text, re.IGNORECASE)
        if match:
            text = match.group(1).strip()
        
        # Take text before punctuation
        text = re.split(r"[.!?;]", text)[0].strip()
        
        # Limit length
        words = text.split()
        if len(words) > 5:
            text = " ".join(words[:5])
        
        return text if text else "unknown"
    
    def _build_prompt(self, question: str) -> str:
        """Build prompt for the model using Vicuna format"""
        return f"USER: <image>\n{question}\nAnswer the question using a single word or short phrase. ASSISTANT:"
    
    
    def _extract_vqa_fields(self, example: Dict) -> Tuple[Any, str, str, List[str]]:
        """Extract image, question_id, question, and answers from dataset example"""
        # Image
        image = example.get("image") or example.get("img")
        if image is None:
            raise ValueError(f"No image field found in example keys: {list(example.keys())}")
        
        # Question ID
        qid = example.get("question_id") or example.get("id") or example.get("idx", "unknown")
        qid = str(qid)
        
        # Question
        question = example.get("question") or example.get("questions")
        if question is None:
            raise ValueError(f"No question field found in example keys: {list(example.keys())}")
        question = str(question)
        
        # Answers
        answers = []
        if "answers" in example:
            ans = example["answers"]
            if isinstance(ans, list):
                if ans and isinstance(ans[0], dict):
                    answers = [str(a.get("answer", "")) for a in ans if "answer" in a]
                else:
                    answers = [str(a) for a in ans]
            elif isinstance(ans, dict) and "text" in ans:
                answers = [str(a) for a in ans["text"]]
        elif "multiple_choice_answer" in example:
            answers = [str(example["multiple_choice_answer"])]
        elif "answer" in example:
            ans = example["answer"]
            answers = [str(ans)] if not isinstance(ans, list) else [str(a) for a in ans]
        
        if not answers:
            answers = [""]
        
        return image, qid, question, answers
    
    def _vqa_accuracy(self, pred: str, answers: List[str]) -> float:
        """
        VQAv2-style soft accuracy: min(#humans_said_pred / 3, 1.0)
        """
        pred_norm = self._normalize_answer(pred)
        answers_norm = [self._normalize_answer(a) for a in answers]
        
        # VQAv2 soft accuracy
        counter = Counter(answers_norm)
        count = counter.get(pred_norm, 0)
        return min(count / 3.0, 1.0)
    
    @torch.no_grad()
    def _generate_batch(
        self, 
        images: List[Image.Image], 
        questions: List[str]
    ) -> List[str]:
        """Generate answers for a batch of images and questions"""
        prompts = [self._build_prompt(q) for q in questions]
        
        # Process inputs
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )
        
        # Decode
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = outputs[:, input_len:]
        
        predictions = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return [pred.strip() for pred in predictions]
    
    def evaluate(self) -> Dict[str, Any]:
        """Run evaluation on the dataset"""
        print(f"\n{'='*60}")
        print(f"Evaluating on VQAv2")
        print(f"Dataset: {self.DATASET_PATH}")
        print(f"Split: {self.config.split}")
        print(f"{'='*60}\n")
        
        # Load VQAv2 dataset
        load_kwargs = {
            "path": self.DATASET_PATH,
            "split": self.config.split,
        }

        try:
            dataset = load_dataset(**load_kwargs)
        except Exception as e:
            print(f"Load failed: {e}")
            print("Retrying with trust_remote_code=True...")
            load_kwargs["trust_remote_code"] = True
            dataset = load_dataset(**load_kwargs)
        
        
        # Limit samples if specified
        if self.config.max_samples is not None:
            dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
        
        print(f"Total samples: {len(dataset)}")
        
        # Evaluate in batches
        batch_images = []
        batch_questions = []
        batch_qids = []
        batch_answers = []
        
        all_results = []
        
        for example in tqdm(dataset, desc="Evaluating"):
            try:
                image, qid, question, answers = self._extract_vqa_fields(example)
                
                batch_images.append(image)
                batch_questions.append(question)
                batch_qids.append(qid)
                batch_answers.append(answers)
                
                # Process batch when full
                if len(batch_images) >= self.config.batch_size:
                    predictions = self._generate_batch(batch_images, batch_questions)
                    
                    for i in range(len(predictions)):
                        pred_short = self._postprocess_answer(predictions[i])
                        acc = self._vqa_accuracy(pred_short, batch_answers[i])
                        
                        result = VQAResult(
                            question_id=batch_qids[i],
                            question=batch_questions[i],
                            prediction=predictions[i],
                            prediction_short=pred_short,
                            ground_truth=batch_answers[i],
                            accuracy=acc,
                            is_correct=(acc >= 0.3),  # Threshold for "correct"
                        )
                        all_results.append(result)
                    
                    # Clear batch
                    batch_images = []
                    batch_questions = []
                    batch_qids = []
                    batch_answers = []
                    
            except Exception as e:
                print(f"\nError processing example: {e}")
                continue
        
        # Process remaining batch
        if batch_images:
            predictions = self._generate_batch(batch_images, batch_questions)
            for i in range(len(predictions)):
                pred_short = self._postprocess_answer(predictions[i])
                acc = self._vqa_accuracy(pred_short, batch_answers[i])
                
                result = VQAResult(
                    question_id=batch_qids[i],
                    question=batch_questions[i],
                    prediction=predictions[i],
                    prediction_short=pred_short,
                    ground_truth=batch_answers[i],
                    accuracy=acc,
                    is_correct=(acc >= 0.3),
                )
                all_results.append(result)
        
        self.results = all_results
        
        # Compute metrics
        metrics = self._compute_metrics()
        
        # Save results
        if self.config.save_predictions:
            self._save_results(metrics)
        
        # Analyze errors
        if self.config.analyze_errors:
            self._analyze_errors()
        
        return metrics
    
    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute evaluation metrics"""
        if not self.results:
            return {}
        
        total = len(self.results)
        accuracies = [r.accuracy for r in self.results]
        
        metrics = {
            "dataset": "vqav2",
            "split": self.config.split,
            "total_samples": total,
            "accuracy": sum(accuracies) / total,
            "exact_match": sum(1 for r in self.results if r.is_correct) / total,
        }
        
        # Per-question-type analysis (if available)
        # This would require additional metadata from the dataset
        
        return metrics
    
    def _save_results(self, metrics: Dict[str, Any]):
        """Save evaluation results to files"""
        base_name = f"vqav2_{self.config.split}"
        
        # Save metrics
        metrics_path = os.path.join(self.config.output_dir, f"{base_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\n✓ Saved metrics to {metrics_path}")
        
        # Save detailed predictions
        predictions_path = os.path.join(self.config.output_dir, f"{base_name}_predictions.jsonl")
        with open(predictions_path, 'w') as f:
            for r in self.results:
                f.write(json.dumps({
                    "question_id": r.question_id,
                    "question": r.question,
                    "prediction": r.prediction,
                    "prediction_short": r.prediction_short,
                    "ground_truth": r.ground_truth,
                    "accuracy": r.accuracy,
                    "is_correct": r.is_correct,
                }) + '\n')
        print(f"✓ Saved predictions to {predictions_path}")
    
    def _analyze_errors(self):
        """Analyze common error patterns"""
        errors = [r for r in self.results if not r.is_correct]
        
        if not errors:
            print("\nNo errors found!")
            return
        
        print(f"\n{'='*60}")
        print(f"ERROR ANALYSIS")
        print(f"{'='*60}")
        print(f"Total errors: {len(errors)} / {len(self.results)} ({len(errors)/len(self.results)*100:.1f}%)")
        
        # Show a few error examples
        print(f"\nSample errors:")
        for i, err in enumerate(errors[:5], 1):
            print(f"\n{i}. Question: {err.question}")
            print(f"   Predicted: {err.prediction_short}")
            print(f"   Ground truth: {err.ground_truth[:3]}")
        
        # Save full error report
        error_path = os.path.join(
            self.config.output_dir,
            f"vqav2_{self.config.split}_errors.jsonl"
        )
        with open(error_path, 'w') as f:
            for err in errors:
                f.write(json.dumps({
                    "question_id": err.question_id,
                    "question": err.question,
                    "prediction": err.prediction_short,
                    "ground_truth": err.ground_truth,
                }) + '\n')
        print(f"\n✓ Saved error analysis to {error_path}")
    
    def print_results(self, metrics: Dict[str, Any]):
        """Pretty print evaluation results"""
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Dataset: {metrics['dataset'].upper()}")
        print(f"Split: {metrics['split']}")
        print(f"Total samples: {metrics['total_samples']}")
        print(f"\n{'Accuracy':<20} {metrics['accuracy']*100:>6.2f}%")
        print(f"{'Exact Match':<20} {metrics['exact_match']*100:>6.2f}%")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="VQAv2 Evaluation for Vision-Language Models")
    
    # Model args
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    
    # Dataset args
    parser.add_argument("--split", default="validation", choices=["validation", "test"], 
                       help="Dataset split (validation or test)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    
    # Generation args
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for inference")
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Max tokens to generate")
    
    # Output args
    parser.add_argument("--output_dir", default="./eval_results", help="Output directory")
    parser.add_argument("--save_predictions", action="store_true", default=True)
    parser.add_argument("--analyze_errors", action="store_true", default=True)
    
    args = parser.parse_args()
    
    # Create config
    config = EvalConfig(
        model_path=args.model_path,
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
        analyze_errors=args.analyze_errors,
    )
    
    # Run evaluation
    evaluator = VQAEvaluator(config)
    metrics = evaluator.evaluate()
    evaluator.print_results(metrics)


if __name__ == "__main__":
    main()