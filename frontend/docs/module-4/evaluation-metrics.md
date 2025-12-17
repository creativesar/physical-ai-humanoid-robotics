---
sidebar_position: 7
title: "Evaluation Metrics"
---

# Evaluation Metrics for VLA Systems

## Introduction to VLA Evaluation

Evaluating Vision-Language-Action (VLA) systems in humanoid robotics requires a comprehensive set of metrics that assess performance across multiple dimensions: perception accuracy, language understanding, action execution, safety, efficiency, and human-robot interaction quality. Unlike traditional AI systems that focus on single-modal performance, VLA systems must be evaluated holistically to ensure they can effectively integrate vision, language, and action in real-world scenarios.

## Multi-Dimensional Evaluation Framework

### 1. Performance Metrics Hierarchy

VLA systems require evaluation across multiple levels of abstraction:

```python
# Comprehensive VLA evaluation framework
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
import json

@dataclass
class VLAMetric:
    """Structure for VLA metric results"""
    name: str
    value: float
    confidence: float
    timestamp: float
    metadata: Dict[str, Any]

class VLAEvaluationFramework:
    """Comprehensive evaluation framework for VLA systems"""
    def __init__(self):
        self.metrics = {}
        self.evaluation_history = []
        self.benchmark_suites = self.initialize_benchmarks()
        self.performance_trackers = {
            'vision': VisionEvaluator(),
            'language': LanguageEvaluator(),
            'action': ActionEvaluator(),
            'integration': IntegrationEvaluator()
        }

    def evaluate_comprehensive(self, vla_model, test_dataset) -> Dict[str, any]:
        """Comprehensive evaluation across all VLA components"""
        results = {}

        # Evaluate vision component
        vision_results = self.performance_trackers['vision'].evaluate(vla_model, test_dataset)
        results['vision'] = vision_results

        # Evaluate language component
        language_results = self.performance_trackers['language'].evaluate(vla_model, test_dataset)
        results['language'] = language_results

        # Evaluate action component
        action_results = self.performance_trackers['action'].evaluate(vla_model, test_dataset)
        results['action'] = action_results

        # Evaluate integration
        integration_results = self.performance_trackers['integration'].evaluate(vla_model, test_dataset)
        results['integration'] = integration_results

        # Calculate overall VLA score
        results['overall'] = self.calculate_overall_vla_score(results)

        # Generate comprehensive report
        results['report'] = self.generate_evaluation_report(results)

        return results

    def calculate_overall_vla_score(self, component_results: Dict[str, any]) -> Dict[str, float]:
        """Calculate overall VLA performance score"""
        # Weighted combination of component scores
        weights = {
            'vision': 0.25,
            'language': 0.25,
            'action': 0.30,
            'integration': 0.20
        }

        overall_score = 0.0
        for component, weight in weights.items():
            if component in component_results:
                # Get primary metric for component (accuracy, F1, etc.)
                primary_metric = self.get_primary_metric(component_results[component])
                overall_score += primary_metric * weight

        return {
            'overall_score': overall_score,
            'component_weights': weights,
            'weighted_scores': {
                comp: self.get_primary_metric(results[comp]) * weights[comp]
                for comp in weights.keys() if comp in results
            }
        }

    def get_primary_metric(self, component_results: Dict[str, any]) -> float:
        """Get primary metric from component results"""
        if 'accuracy' in component_results:
            return component_results['accuracy']
        elif 'f1_score' in component_results:
            return component_results['f1_score']
        elif 'mean_error' in component_results:
            # Invert for accuracy-like metric (lower error = higher score)
            return 1.0 / (1.0 + component_results['mean_error'])
        elif 'success_rate' in component_results:
            return component_results['success_rate']
        else:
            return 0.5  # Default neutral score

    def generate_evaluation_report(self, results: Dict[str, any]) -> str:
        """Generate comprehensive evaluation report"""
        report = f"""
# VLA System Evaluation Report

## Executive Summary
- Overall VLA Score: {results['overall']['overall_score']:.3f}
- Evaluation Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}
- Test Dataset Size: {self.get_dataset_size()}

## Component Performance

### Vision Component
- Accuracy: {results['vision'].get('accuracy', 0):.3f}
- Object Detection F1: {results['vision'].get('object_detection_f1', 0):.3f}
- Semantic Segmentation IoU: {results['vision'].get('segmentation_iou', 0):.3f}

### Language Component
- Command Understanding: {results['language'].get('command_accuracy', 0):.3f}
- Semantic Parsing: {results['language'].get('semantic_f1', 0):.3f}
- Natural Language Generation: {results['language'].get('generation_bleu', 0):.3f}

### Action Component
- Task Success Rate: {results['action'].get('success_rate', 0):.3f}
- Action Accuracy: {results['action'].get('action_accuracy', 0):.3f}
- Execution Time: {results['action'].get('avg_execution_time', 0):.3f}s

### Integration Component
- Vision-Language Alignment: {results['integration'].get('alignment_score', 0):.3f}
- Task Completion Rate: {results['integration'].get('task_completion_rate', 0):.3f}
- Human Satisfaction: {results['integration'].get('human_satisfaction', 0):.3f}

## Recommendations
{self.generate_recommendations(results)}
        """

        return report.strip()

    def generate_recommendations(self, results: Dict[str, any]) -> str:
        """Generate improvement recommendations based on results"""
        recommendations = []

        # Vision component recommendations
        if results['vision'].get('accuracy', 0) < 0.8:
            recommendations.append("- Improve vision component accuracy (current: {:.3f})".format(
                results['vision'].get('accuracy', 0)))

        # Language component recommendations
        if results['language'].get('command_accuracy', 0) < 0.85:
            recommendations.append("- Enhance language understanding capabilities (current: {:.3f})".format(
                results['language'].get('command_accuracy', 0)))

        # Action component recommendations
        if results['action'].get('success_rate', 0) < 0.75:
            recommendations.append("- Optimize action execution success rate (current: {:.3f})".format(
                results['action'].get('success_rate', 0)))

        # Integration recommendations
        if results['integration'].get('task_completion_rate', 0) < 0.8:
            recommendations.append("- Improve multi-modal integration (current: {:.3f})".format(
                results['integration'].get('task_completion_rate', 0)))

        if not recommendations:
            recommendations.append("- System performance is satisfactory across all components")

        return "\n".join(recommendations)

    def get_dataset_size(self) -> int:
        """Get size of evaluation dataset"""
        # This would interface with the actual dataset
        # For now, return a placeholder
        return 1000

class VisionEvaluator:
    """Evaluator for vision component of VLA systems"""
    def __init__(self):
        self.object_detection_metrics = ObjectDetectionEvaluator()
        self.segmentation_metrics = SegmentationEvaluator()
        self.feature_matching_metrics = FeatureMatchingEvaluator()

    def evaluate(self, vla_model, test_dataset) -> Dict[str, any]:
        """Evaluate vision component performance"""
        results = {}

        # Object detection evaluation
        detection_results = self.object_detection_metrics.evaluate(vla_model, test_dataset)
        results.update(detection_results)

        # Semantic segmentation evaluation
        segmentation_results = self.segmentation_metrics.evaluate(vla_model, test_dataset)
        results.update(segmentation_results)

        # Feature matching evaluation
        matching_results = self.feature_matching_metrics.evaluate(vla_model, test_dataset)
        results.update(matching_results)

        # Calculate overall vision score
        results['overall_vision_score'] = self.calculate_vision_score(results)

        return results

    def calculate_vision_score(self, results: Dict[str, any]) -> float:
        """Calculate overall vision component score"""
        score = 0.0
        weight_sum = 0.0

        if 'object_detection_f1' in results:
            score += results['object_detection_f1'] * 0.4
            weight_sum += 0.4

        if 'segmentation_iou' in results:
            score += results['segmentation_iou'] * 0.3
            weight_sum += 0.3

        if 'feature_matching_accuracy' in results:
            score += results['feature_matching_accuracy'] * 0.3
            weight_sum += 0.3

        return score / weight_sum if weight_sum > 0 else 0.0

class ObjectDetectionEvaluator:
    """Evaluate object detection capabilities"""
    def evaluate(self, vla_model, test_dataset) -> Dict[str, any]:
        """Evaluate object detection performance"""
        all_predictions = []
        all_ground_truths = []

        for sample in test_dataset:
            # Get vision-only predictions
            image = sample['image']
            with torch.no_grad():
                detections = vla_model.vision_component.detect_objects(image)

            all_predictions.append(detections)
            all_ground_truths.append(sample['ground_truth_objects'])

        # Calculate detection metrics
        precision, recall, f1_score, _ = self.calculate_detection_metrics(
            all_predictions, all_ground_truths
        )

        # Calculate mAP (mean Average Precision)
        map_score = self.calculate_mean_average_precision(all_predictions, all_ground_truths)

        return {
            'object_detection_precision': precision,
            'object_detection_recall': recall,
            'object_detection_f1': f1_score,
            'object_detection_map': map_score,
            'object_detection_accuracy': self.calculate_detection_accuracy(all_predictions, all_ground_truths)
        }

    def calculate_detection_metrics(self, predictions: List[Dict[str, any]],
                                  ground_truths: List[Dict[str, any]]) -> Tuple[float, float, float, float]:
        """Calculate object detection metrics"""
        # This would implement COCO-style evaluation
        # For now, return mock values
        return 0.85, 0.82, 0.83, 0.90  # precision, recall, f1, accuracy

    def calculate_mean_average_precision(self, predictions: List[Dict[str, any]],
                                       ground_truths: List[Dict[str, any]]) -> float:
        """Calculate mean Average Precision"""
        # Implement mAP calculation (would use COCO evaluation tools in practice)
        return 0.78  # Mock mAP score

    def calculate_detection_accuracy(self, predictions: List[Dict[str, any]],
                                   ground_truths: List[Dict[str, any]]) -> float:
        """Calculate detection accuracy"""
        correct_detections = 0
        total_detections = 0

        for pred, gt in zip(predictions, ground_truths):
            # Count correct detections based on IoU threshold
            for pred_obj in pred.get('objects', []):
                for gt_obj in gt.get('objects', []):
                    if (pred_obj['class'] == gt_obj['class'] and
                        self.calculate_iou(pred_obj['bbox'], gt_obj['bbox']) > 0.5):
                        correct_detections += 1
                        break
                total_detections += 1

        return correct_detections / total_detections if total_detections > 0 else 0.0

    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

        # Calculate union
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0.0

class SegmentationEvaluator:
    """Evaluate semantic segmentation capabilities"""
    def evaluate(self, vla_model, test_dataset) -> Dict[str, any]:
        """Evaluate segmentation performance"""
        iou_scores = []
        pixel_accuracies = []

        for sample in test_dataset:
            image = sample['image']
            ground_truth_seg = sample['ground_truth_segmentation']

            with torch.no_grad():
                predicted_seg = vla_model.vision_component.segment_image(image)

            # Calculate IoU
            iou = self.calculate_segmentation_iou(predicted_seg, ground_truth_seg)
            iou_scores.append(iou)

            # Calculate pixel accuracy
            pixel_acc = self.calculate_pixel_accuracy(predicted_seg, ground_truth_seg)
            pixel_accuracies.append(pixel_acc)

        mean_iou = np.mean(iou_scores) if iou_scores else 0.0
        mean_pixel_accuracy = np.mean(pixel_accuracies) if pixel_accuracies else 0.0

        return {
            'segmentation_iou': mean_iou,
            'segmentation_pixel_accuracy': mean_pixel_accuracy,
            'segmentation_dice_score': self.calculate_dice_score(iou_scores),
            'segmentation_per_class_iou': self.calculate_per_class_iou(test_dataset, vla_model)
        }

    def calculate_segmentation_iou(self, pred_seg: torch.Tensor,
                                 gt_seg: torch.Tensor) -> float:
        """Calculate segmentation IoU"""
        # Convert to numpy for calculation
        pred_np = pred_seg.cpu().numpy()
        gt_np = gt_seg.cpu().numpy()

        intersection = np.logical_and(pred_np, gt_np).sum()
        union = np.logical_or(pred_np, gt_np).sum()

        return intersection / union if union > 0 else 0.0

    def calculate_pixel_accuracy(self, pred_seg: torch.Tensor,
                               gt_seg: torch.Tensor) -> float:
        """Calculate pixel-level accuracy"""
        pred_np = pred_seg.cpu().numpy()
        gt_np = gt_seg.cpu().numpy()

        correct_pixels = np.equal(pred_np, gt_np).sum()
        total_pixels = pred_np.size

        return correct_pixels / total_pixels if total_pixels > 0 else 0.0

    def calculate_dice_score(self, iou_scores: List[float]) -> float:
        """Calculate Dice coefficient from IoU scores"""
        # Dice = 2 * IoU / (IoU + 1)
        dice_scores = [2 * iou / (iou + 1) for iou in iou_scores if iou >= 0]
        return np.mean(dice_scores) if dice_scores else 0.0

    def calculate_per_class_iou(self, test_dataset, vla_model) -> Dict[str, float]:
        """Calculate IoU for each semantic class"""
        # This would calculate IoU for each object class
        # For now, return mock values
        return {
            'person': 0.85,
            'chair': 0.78,
            'table': 0.82,
            'cabinet': 0.75,
            'door': 0.80,
            'floor': 0.95,
            'wall': 0.90
        }

class LanguageEvaluator:
    """Evaluator for language component of VLA systems"""
    def __init__(self):
        self.command_understanding = CommandUnderstandingEvaluator()
        self.semantic_parsing = SemanticParsingEvaluator()
        self.natural_language_generation = NLGEvaluator()

    def evaluate(self, vla_model, test_dataset) -> Dict[str, any]:
        """Evaluate language component performance"""
        results = {}

        # Command understanding evaluation
        command_results = self.command_understanding.evaluate(vla_model, test_dataset)
        results.update(command_results)

        # Semantic parsing evaluation
        parsing_results = self.semantic_parsing.evaluate(vla_model, test_dataset)
        results.update(parsing_results)

        # Natural language generation evaluation
        generation_results = self.natural_language_generation.evaluate(vla_model, test_dataset)
        results.update(generation_results)

        # Calculate overall language score
        results['overall_language_score'] = self.calculate_language_score(results)

        return results

    def calculate_language_score(self, results: Dict[str, any]) -> float:
        """Calculate overall language component score"""
        score = 0.0
        weight_sum = 0.0

        if 'command_accuracy' in results:
            score += results['command_accuracy'] * 0.4
            weight_sum += 0.4

        if 'semantic_f1' in results:
            score += results['semantic_f1'] * 0.3
            weight_sum += 0.3

        if 'generation_bleu' in results:
            score += results['generation_bleu'] * 0.3
            weight_sum += 0.3

        return score / weight_sum if weight_sum > 0 else 0.0

class CommandUnderstandingEvaluator:
    """Evaluate command understanding capabilities"""
    def evaluate(self, vla_model, test_dataset) -> Dict[str, any]:
        """Evaluate command understanding performance"""
        correct_understandings = 0
        total_commands = 0
        command_types = {}

        for sample in test_dataset:
            if 'command' in sample and 'intended_action' in sample:
                command = sample['command']
                intended_action = sample['intended_action']

                with torch.no_grad():
                    parsed_intent = vla_model.language_component.parse_command(command)

                # Check if intent matches intended action
                if self.intent_matches_action(parsed_intent, intended_action):
                    correct_understandings += 1

                # Track command types
                cmd_type = self.classify_command_type(command)
                if cmd_type not in command_types:
                    command_types[cmd_type] = {'correct': 0, 'total': 0}
                command_types[cmd_type]['total'] += 1
                if self.intent_matches_action(parsed_intent, intended_action):
                    command_types[cmd_type]['correct'] += 1

                total_commands += 1

        overall_accuracy = correct_understandings / total_commands if total_commands > 0 else 0.0

        # Calculate per-type accuracy
        per_type_accuracy = {
            cmd_type: cmd_data['correct'] / cmd_data['total']
            for cmd_type, cmd_data in command_types.items()
        }

        return {
            'command_accuracy': overall_accuracy,
            'command_per_type_accuracy': per_type_accuracy,
            'command_total_evaluated': total_commands,
            'command_correct_understood': correct_understandings
        }

    def intent_matches_action(self, parsed_intent: Dict[str, any],
                             intended_action: Dict[str, any]) -> bool:
        """Check if parsed intent matches intended action"""
        # This would implement sophisticated intent matching
        # For now, use simple string matching
        parsed_action = parsed_intent.get('action', '').lower()
        intended_action_str = str(intended_action).lower()

        return parsed_action in intended_action_str or intended_action_str in parsed_action

    def classify_command_type(self, command: str) -> str:
        """Classify command into types"""
        command_lower = command.lower()

        if any(word in command_lower for word in ['pick', 'grasp', 'take', 'grab']):
            return 'manipulation'
        elif any(word in command_lower for word in ['go', 'move', 'navigate', 'walk']):
            return 'navigation'
        elif any(word in command_lower for word in ['follow', 'come', 'after']):
            return 'following'
        elif any(word in command_lower for word in ['stop', 'wait', 'pause']):
            return 'control'
        elif any(word in command_lower for word in ['tell', 'describe', 'explain']):
            return 'information'
        else:
            return 'other'

    def calculate_command_complexity(self, command: str) -> float:
        """Calculate command complexity (0.0-1.0)"""
        # More complex commands have more words, objects, and relationships
        words = command.split()
        complexity = min(1.0, len(words) / 10.0)  # Normalize by 10 words

        # Consider number of objects mentioned
        object_count = self.count_mentioned_objects(command)
        complexity = max(complexity, object_count * 0.2)

        return complexity

    def count_mentioned_objects(self, command: str) -> int:
        """Count objects mentioned in command"""
        common_objects = [
            'cup', 'bottle', 'box', 'chair', 'table', 'door', 'person',
            'robot', 'computer', 'phone', 'book', 'paper', 'pen'
        ]

        count = 0
        command_lower = command.lower()
        for obj in common_objects:
            if obj in command_lower:
                count += 1

        return count

class SemanticParsingEvaluator:
    """Evaluate semantic parsing capabilities"""
    def evaluate(self, vla_model, test_dataset) -> Dict[str, any]:
        """Evaluate semantic parsing performance"""
        all_parsed = []
        all_ground_truth = []

        for sample in test_dataset:
            if 'command' in sample and 'semantic_annotation' in sample:
                command = sample['command']
                ground_truth_semantic = sample['semantic_annotation']

                with torch.no_grad():
                    parsed_semantic = vla_model.language_component.parse_semantic(command)

                all_parsed.append(parsed_semantic)
                all_ground_truth.append(ground_truth_semantic)

        # Calculate semantic parsing metrics
        precision, recall, f1_score, support = precision_recall_fscore_support(
            y_true=self.flatten_semantic_annotations(all_ground_truth),
            y_pred=self.flatten_semantic_annotations(all_parsed),
            average='weighted',
            zero_division=0
        )

        return {
            'semantic_precision': precision,
            'semantic_recall': recall,
            'semantic_f1': f1_score,
            'semantic_accuracy': self.calculate_semantic_accuracy(all_parsed, all_ground_truth),
            'semantic_entity_recognition': self.evaluate_entity_recognition(all_parsed, all_ground_truth)
        }

    def flatten_semantic_annotations(self, semantic_annotations: List[Dict[str, any]]) -> List[str]:
        """Flatten semantic annotations for sklearn compatibility"""
        flattened = []
        for annotation in semantic_annotations:
            # Extract semantic elements (entities, relations, actions)
            for key, value in annotation.items():
                if isinstance(value, list):
                    flattened.extend([f"{key}_{item}" for item in value])
                else:
                    flattened.append(f"{key}_{value}")
        return flattened

    def calculate_semantic_accuracy(self, parsed: List[Dict[str, any]],
                                  ground_truth: List[Dict[str, any]]) -> float:
        """Calculate semantic parsing accuracy"""
        correct = 0
        total = 0

        for p, gt in zip(parsed, ground_truth):
            if self.semantic_structures_match(p, gt):
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.0

    def semantic_structures_match(self, parsed: Dict[str, any],
                                 ground_truth: Dict[str, any]) -> bool:
        """Check if semantic structures match"""
        # Compare key components of semantic structure
        return (
            parsed.get('action') == ground_truth.get('action') and
            parsed.get('object') == ground_truth.get('object') and
            parsed.get('location') == ground_truth.get('location')
        )

    def evaluate_entity_recognition(self, parsed: List[Dict[str, any]],
                                  ground_truth: List[Dict[str, any]]) -> Dict[str, float]:
        """Evaluate entity recognition performance"""
        entity_types = ['object', 'location', 'person', 'action', 'modifier']

        entity_metrics = {}
        for entity_type in entity_types:
            true_entities = [gt.get(entity_type, '') for gt in ground_truth]
            pred_entities = [p.get(entity_type, '') for p in parsed]

            # Calculate entity recognition metrics
            correct = sum(1 for t, p in zip(true_entities, pred_entities) if t == p)
            total = len(true_entities)

            entity_metrics[entity_type] = correct / total if total > 0 else 0.0

        return entity_metrics

class NLGEvaluator:
    """Evaluate natural language generation capabilities"""
    def evaluate(self, vla_model, test_dataset) -> Dict[str, any]:
        """Evaluate NLG performance"""
        generated_texts = []
        reference_texts = []

        for sample in test_dataset:
            if 'input_context' in sample and 'reference_response' in sample:
                context = sample['input_context']
                reference = sample['reference_response']

                with torch.no_grad():
                    generated = vla_model.language_component.generate_response(context)

                generated_texts.append(generated)
                reference_texts.append(reference)

        # Calculate NLG metrics
        bleu_score = self.calculate_bleu_score(generated_texts, reference_texts)
        rouge_scores = self.calculate_rouge_scores(generated_texts, reference_texts)
        distinct_scores = self.calculate_distinct_scores(generated_texts)

        return {
            'generation_bleu': bleu_score,
            'generation_rouge': rouge_scores,
            'generation_distinct': distinct_scores,
            'generation_coherence': self.calculate_coherence_score(generated_texts),
            'generation_fluency': self.estimate_fluency_score(generated_texts)
        }

    def calculate_bleu_score(self, generated: List[str], reference: List[str]) -> float:
        """Calculate BLEU score"""
        try:
            from nltk.translate.bleu_score import sentence_bleu
            bleu_scores = []
            for gen, ref in zip(generated, reference):
                gen_tokens = gen.split()
                ref_tokens = [ref.split()]  # List of reference sentences
                bleu = sentence_bleu(ref_tokens, gen_tokens)
                bleu_scores.append(bleu)
            return sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
        except ImportError:
            print("NLTK not available, using mock BLEU score")
            return 0.65  # Mock score

    def calculate_rouge_scores(self, generated: List[str], reference: List[str]) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

            rouge_results = {'rouge1': [], 'rouge2': [], 'rougeL': []}

            for gen, ref in zip(generated, reference):
                scores = scorer.score(ref, gen)
                for metric, score in scores.items():
                    rouge_results[metric].append(score.fmeasure)

            return {
                metric: sum(scores) / len(scores) if scores else 0.0
                for metric, scores in rouge_results.items()
            }
        except ImportError:
            print("rouge_score not available, using mock ROUGE scores")
            return {'rouge1': 0.55, 'rouge2': 0.35, 'rougeL': 0.52}

    def calculate_distinct_scores(self, generated_texts: List[str]) -> Dict[str, float]:
        """Calculate distinct n-gram scores (diversity metric)"""
        all_unigrams = set()
        all_bigrams = set()
        total_unigrams = 0
        total_bigrams = 0

        for text in generated_texts:
            tokens = text.split()
            # Unigrams
            all_unigrams.update(tokens)
            total_unigrams += len(tokens)

            # Bigrams
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
            all_bigrams.update(bigrams)
            total_bigrams += len(bigrams)

        distinct_1 = len(all_unigrams) / total_unigrams if total_unigrams > 0 else 0.0
        distinct_2 = len(all_bigrams) / total_bigrams if total_bigrams > 0 else 0.0

        return {
            'distinct_1': distinct_1,
            'distinct_2': distinct_2,
            'vocabulary_diversity': (distinct_1 + distinct_2) / 2
        }

    def calculate_coherence_score(self, generated_texts: List[str]) -> float:
        """Calculate coherence score using simple heuristics"""
        coherence_scores = []

        for text in generated_texts:
            # Calculate coherence based on sentence structure and flow
            sentences = text.split('.')
            if len(sentences) < 2:
                coherence_scores.append(0.5)  # Neutral score for single sentences
                continue

            # Check for pronoun usage (indicative of coherence)
            pronouns = ['he', 'she', 'it', 'they', 'we', 'him', 'her', 'them']
            pronoun_count = sum(1 for word in text.lower().split() if word in pronouns)

            # Check for discourse markers
            markers = ['however', 'therefore', 'because', 'since', 'while', 'although']
            marker_count = sum(1 for word in text.lower().split() if word in markers)

            coherence_score = min(1.0, (pronoun_count + marker_count * 2) / len(sentences))
            coherence_scores.append(coherence_score)

        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5

    def estimate_fluency_score(self, generated_texts: List[str]) -> float:
        """Estimate fluency score"""
        # Simple fluency estimation based on grammar and structure
        fluency_scores = []

        for text in generated_texts:
            # Calculate basic fluency metrics
            words = text.split()
            avg_word_length = np.mean([len(word) for word in words]) if words else 0

            # Count unusual word lengths (potential spelling errors)
            unusual_words = sum(1 for word in words if len(word) > 20 or len(word) < 2)
            unusual_ratio = unusual_words / len(words) if words else 0

            # Estimate fluency
            fluency = max(0.0, 1.0 - unusual_ratio - abs(avg_word_length - 5) / 10.0)
            fluency_scores.append(fluency)

        return sum(fluency_scores) / len(fluency_scores) if fluency_scores else 0.5
```

### 2. Action Execution Metrics

```python
# Action execution evaluation for humanoid robots
class ActionEvaluator:
    """Evaluator for action component of VLA systems"""
    def __init__(self):
        self.trajectory_evaluator = TrajectoryEvaluator()
        self.manipulation_evaluator = ManipulationEvaluator()
        self.navigation_evaluator = NavigationEvaluator()
        self.safety_evaluator = SafetyEvaluator()

    def evaluate(self, vla_model, test_dataset) -> Dict[str, any]:
        """Evaluate action component performance"""
        results = {}

        # Trajectory execution evaluation
        trajectory_results = self.trajectory_evaluator.evaluate(vla_model, test_dataset)
        results.update(trajectory_results)

        # Manipulation evaluation
        manipulation_results = self.manipulation_evaluator.evaluate(vla_model, test_dataset)
        results.update(manipulation_results)

        # Navigation evaluation
        navigation_results = self.navigation_evaluator.evaluate(vla_model, test_dataset)
        results.update(navigation_results)

        # Safety evaluation
        safety_results = self.safety_evaluator.evaluate(vla_model, test_dataset)
        results.update(safety_results)

        # Calculate overall action score
        results['overall_action_score'] = self.calculate_action_score(results)

        return results

    def calculate_action_score(self, results: Dict[str, any]) -> float:
        """Calculate overall action component score"""
        score = 0.0
        weight_sum = 0.0

        if 'trajectory_success_rate' in results:
            score += results['trajectory_success_rate'] * 0.3
            weight_sum += 0.3

        if 'manipulation_success_rate' in results:
            score += results['manipulation_success_rate'] * 0.3
            weight_sum += 0.3

        if 'navigation_success_rate' in results:
            score += results['navigation_success_rate'] * 0.25
            weight_sum += 0.25

        if 'safety_compliance_rate' in results:
            score += results['safety_compliance_rate'] * 0.15
            weight_sum += 0.15

        return score / weight_sum if weight_sum > 0 else 0.0

class TrajectoryEvaluator:
    """Evaluate trajectory execution performance"""
    def evaluate(self, vla_model, test_dataset) -> Dict[str, any]:
        """Evaluate trajectory execution"""
        success_count = 0
        total_count = 0
        execution_times = []
        trajectory_errors = []

        for sample in test_dataset:
            if 'trajectory_goal' in sample and 'initial_state' in sample:
                initial_state = sample['initial_state']
                goal_state = sample['trajectory_goal']

                start_time = time.time()
                with torch.no_grad():
                    executed_trajectory = vla_model.action_component.execute_trajectory(
                        initial_state, goal_state
                    )
                execution_time = time.time() - start_time

                # Evaluate trajectory success
                success = self.evaluate_trajectory_success(executed_trajectory, goal_state)
                if success:
                    success_count += 1

                execution_times.append(execution_time)
                trajectory_errors.append(self.calculate_trajectory_error(executed_trajectory, goal_state))

                total_count += 1

        success_rate = success_count / total_count if total_count > 0 else 0.0
        avg_execution_time = np.mean(execution_times) if execution_times else 0.0
        avg_error = np.mean(trajectory_errors) if trajectory_errors else float('inf')

        return {
            'trajectory_success_rate': success_rate,
            'trajectory_avg_execution_time': avg_execution_time,
            'trajectory_avg_error': avg_error,
            'trajectory_total_evaluated': total_count,
            'trajectory_smoothness': self.calculate_trajectory_smoothness(executed_trajectory),
            'trajectory_efficiency': self.calculate_trajectory_efficiency(executed_trajectory, goal_state)
        }

    def evaluate_trajectory_success(self, executed_trajectory: List[Dict[str, any]],
                                  goal_state: Dict[str, any]) -> bool:
        """Evaluate if trajectory successfully reached goal"""
        if not executed_trajectory:
            return False

        final_state = executed_trajectory[-1]
        goal_position = goal_state.get('position', [0, 0, 0])
        final_position = final_state.get('position', [0, 0, 0])

        # Calculate distance to goal
        distance = np.linalg.norm(np.array(goal_position) - np.array(final_position))

        # Success if within tolerance
        tolerance = goal_state.get('tolerance', 0.1)  # Default 10cm tolerance
        return distance <= tolerance

    def calculate_trajectory_error(self, trajectory: List[Dict[str, any]],
                                 goal_state: Dict[str, any]) -> float:
        """Calculate trajectory error"""
        if not trajectory:
            return float('inf')

        final_state = trajectory[-1]
        goal_position = goal_state.get('position', [0, 0, 0])
        final_position = final_state.get('position', [0, 0, 0])

        return np.linalg.norm(np.array(goal_position) - np.array(final_position))

    def calculate_trajectory_smoothness(self, trajectory: List[Dict[str, any]]) -> float:
        """Calculate trajectory smoothness"""
        if len(trajectory) < 3:
            return 1.0  # Perfect smoothness for short trajectories

        # Calculate jerk (third derivative) to measure smoothness
        positions = np.array([state['position'] for state in trajectory if 'position' in state])

        if len(positions) < 3:
            return 1.0

        # Calculate velocities
        velocities = np.diff(positions, axis=0)

        # Calculate accelerations
        accelerations = np.diff(velocities, axis=0)

        # Calculate jerk
        jerks = np.diff(accelerations, axis=0)

        # Average jerk magnitude (lower is smoother)
        avg_jerk = np.mean(np.linalg.norm(jerks, axis=1))

        # Convert to smoothness score (higher = smoother)
        smoothness_score = 1.0 / (1.0 + avg_jerk)

        return smoothness_score

    def calculate_trajectory_efficiency(self, trajectory: List[Dict[str, any]],
                                      goal_state: Dict[str, any]) -> float:
        """Calculate trajectory efficiency (path length vs direct distance)"""
        if len(trajectory) < 2 or not trajectory:
            return 0.0

        # Calculate actual path length
        positions = [state['position'] for state in trajectory if 'position' in state]
        path_length = 0.0
        for i in range(1, len(positions)):
            path_length += np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))

        # Calculate direct distance to goal
        start_pos = np.array(positions[0])
        goal_pos = np.array(goal_state.get('position', positions[-1]))

        direct_distance = np.linalg.norm(goal_pos - start_pos)

        # Efficiency = direct distance / path length (lower ratio = less efficient)
        if path_length > 0:
            efficiency = direct_distance / path_length
            # Normalize to 0-1 scale where 1.0 = perfectly efficient (straight line)
            return min(1.0, efficiency * 2.0)  # Multiply by 2 to account for realistic deviations
        else:
            return 0.0

class ManipulationEvaluator:
    """Evaluate manipulation task performance"""
    def __init__(self):
        self.grasp_evaluator = GraspEvaluator()
        self.place_evaluator = PlaceEvaluator()
        self.transport_evaluator = TransportEvaluator()

    def evaluate(self, vla_model, test_dataset) -> Dict[str, any]:
        """Evaluate manipulation performance"""
        results = {}

        # Grasp evaluation
        grasp_results = self.grasp_evaluator.evaluate(vla_model, test_dataset)
        results.update(grasp_results)

        # Place evaluation
        place_results = self.place_evaluator.evaluate(vla_model, test_dataset)
        results.update(place_results)

        # Transport evaluation
        transport_results = self.transport_evaluator.evaluate(vla_model, test_dataset)
        results.update(transport_results)

        # Calculate overall manipulation score
        manipulation_score = self.calculate_manipulation_score(results)
        results['manipulation_overall_score'] = manipulation_score

        return results

    def calculate_manipulation_score(self, results: Dict[str, any]) -> float:
        """Calculate overall manipulation score"""
        score = 0.0
        weight_sum = 0.0

        if 'grasp_success_rate' in results:
            score += results['grasp_success_rate'] * 0.4
            weight_sum += 0.4

        if 'place_success_rate' in results:
            score += results['place_success_rate'] * 0.3
            weight_sum += 0.3

        if 'transport_success_rate' in results:
            score += results['transport_success_rate'] * 0.3
            weight_sum += 0.3

        return score / weight_sum if weight_sum > 0 else 0.0

class GraspEvaluator:
    """Evaluate grasp execution performance"""
    def evaluate(self, vla_model, test_dataset) -> Dict[str, any]:
        """Evaluate grasp performance"""
        grasp_successes = 0
        total_grasps = 0
        grasp_times = []
        grasp_qualities = []

        for sample in test_dataset:
            if 'grasp_task' in sample:
                task = sample['grasp_task']

                start_time = time.time()
                with torch.no_grad():
                    grasp_result = vla_model.action_component.execute_grasp(task)
                execution_time = time.time() - start_time

                success = self.evaluate_grasp_success(grasp_result, task)
                if success:
                    grasp_successes += 1

                quality = self.evaluate_grasp_quality(grasp_result, task)

                grasp_times.append(execution_time)
                grasp_qualities.append(quality)
                total_grasps += 1

        success_rate = grasp_successes / total_grasps if total_grasps > 0 else 0.0
        avg_time = np.mean(grasp_times) if grasp_times else 0.0
        avg_quality = np.mean(grasp_qualities) if grasp_qualities else 0.0

        return {
            'grasp_success_rate': success_rate,
            'grasp_avg_time': avg_time,
            'grasp_avg_quality': avg_quality,
            'grasp_total_evaluated': total_grasps,
            'grasp_success_count': grasp_successes,
            'grasp_force_efficiency': self.calculate_force_efficiency(grasp_result),
            'grasp_stability': self.calculate_stability_score(grasp_result)
        }

    def evaluate_grasp_success(self, grasp_result: Dict[str, any],
                              grasp_task: Dict[str, any]) -> bool:
        """Evaluate if grasp was successful"""
        # Check if object was successfully grasped
        success = grasp_result.get('grasp_successful', False)

        # Additional checks based on task requirements
        if success and 'required_object' in grasp_task:
            grabbed_object = grasp_result.get('grabbed_object')
            required_object = grasp_task['required_object']
            success = success and (grabbed_object == required_object)

        return success

    def evaluate_grasp_quality(self, grasp_result: Dict[str, any],
                              grasp_task: Dict[str, any]) -> float:
        """Evaluate grasp quality"""
        quality_score = 0.0

        # Stability factor
        stability = grasp_result.get('stability_score', 0.5)
        quality_score += stability * 0.4

        # Force efficiency (not using excessive force)
        applied_force = grasp_result.get('applied_force', 10.0)
        max_allowed_force = grasp_task.get('max_safe_force', 20.0)
        force_efficiency = max(0.0, 1.0 - (applied_force / max_allowed_force))
        quality_score += force_efficiency * 0.3

        # Grasp appropriateness (right grasp type for object)
        grasp_type = grasp_result.get('grasp_type', 'unknown')
        object_type = grasp_task.get('object_type', 'unknown')
        appropriateness = self.evaluate_grasp_appropriateness(grasp_type, object_type)
        quality_score += appropriateness * 0.3

        return min(1.0, quality_score)

    def evaluate_grasp_appropriateness(self, grasp_type: str, object_type: str) -> float:
        """Evaluate if grasp type is appropriate for object type"""
        appropriate_grasps = {
            'small_object': ['precision_pinch', 'tripod'],
            'large_object': ['power_grasp', 'cylindrical'],
            'fragile_object': ['precision_pinch', 'lateral'],
            'heavy_object': ['power_grasp', 'spherical']
        }

        if object_type in appropriate_grasps:
            if grasp_type in appropriate_grasps[object_type]:
                return 1.0  # Perfect match
            else:
                return 0.5  # Partial match
        else:
            return 0.7  # Default for unknown object types

    def calculate_force_efficiency(self, grasp_result: Dict[str, any]) -> float:
        """Calculate force efficiency of grasp"""
        applied_force = grasp_result.get('applied_force', 10.0)
        required_force = grasp_result.get('required_force', 5.0)

        # Efficiency is higher when applied force is close to required force
        # but not exceeding it unnecessarily
        if applied_force <= required_force:
            # Perfect if force is just enough
            return 1.0
        else:
            # Efficiency decreases as excess force increases
            excess_ratio = (applied_force - required_force) / required_force
            return max(0.0, 1.0 - excess_ratio)

    def calculate_stability_score(self, grasp_result: Dict[str, any]) -> float:
        """Calculate grasp stability score"""
        # Stability based on force distribution, contact points, etc.
        contact_points = grasp_result.get('contact_points', [])
        force_distribution = grasp_result.get('force_distribution', {})

        # More contact points generally mean more stability
        stability_from_contacts = min(1.0, len(contact_points) / 4.0)  # Normalize to 4 contacts

        # Balanced force distribution means stability
        if force_distribution:
            forces = list(force_distribution.values())
            force_variance = np.var(forces) if len(forces) > 1 else 0.0
            stability_from_balance = max(0.0, 1.0 - force_variance)  # Lower variance = higher stability
        else:
            stability_from_balance = 0.5  # Default

        # Combined stability score
        stability_score = 0.6 * stability_from_contacts + 0.4 * stability_from_balance

        return stability_score

class NavigationEvaluator:
    """Evaluate navigation task performance"""
    def evaluate(self, vla_model, test_dataset) -> Dict[str, any]:
        """Evaluate navigation performance"""
        success_count = 0
        total_navigations = 0
        path_efficiencies = []
        navigation_times = []
        collision_counts = []

        for sample in test_dataset:
            if 'navigation_task' in sample:
                task = sample['navigation_task']

                start_time = time.time()
                with torch.no_grad():
                    navigation_result = vla_model.action_component.navigate_to(task)
                execution_time = time.time() - start_time

                success = self.evaluate_navigation_success(navigation_result, task)
                if success:
                    success_count += 1

                efficiency = self.calculate_path_efficiency(navigation_result, task)
                collisions = self.count_collisions(navigation_result)

                path_efficiencies.append(efficiency)
                navigation_times.append(execution_time)
                collision_counts.append(collisions)

                total_navigations += 1

        success_rate = success_count / total_navigations if total_navigations > 0 else 0.0
        avg_efficiency = np.mean(path_efficiencies) if path_efficiencies else 0.0
        avg_time = np.mean(navigation_times) if navigation_times else 0.0
        avg_collisions = np.mean(collision_counts) if collision_counts else 0.0

        return {
            'navigation_success_rate': success_rate,
            'navigation_avg_time': avg_time,
            'navigation_avg_efficiency': avg_efficiency,
            'navigation_avg_collisions': avg_collisions,
            'navigation_total_evaluated': total_navigations,
            'navigation_success_count': success_count,
            'navigation_safety_score': self.calculate_safety_score(collision_counts),
            'navigation_adaptability': self.calculate_adaptability_score(navigation_result)
        }

    def evaluate_navigation_success(self, nav_result: Dict[str, any],
                                   nav_task: Dict[str, any]) -> bool:
        """Evaluate navigation success"""
        # Check if destination was reached within tolerance
        destination_reached = nav_result.get('destination_reached', False)
        if not destination_reached:
            return False

        # Check if arrival was within time constraints
        arrival_time = nav_result.get('arrival_time', float('inf'))
        max_time = nav_task.get('max_time', float('inf'))
        if arrival_time > max_time:
            return False

        # Check if path was collision-free
        collisions = nav_result.get('collision_count', 0)
        max_collisions = nav_task.get('max_collisions', 0)
        if collisions > max_collisions:
            return False

        return True

    def calculate_path_efficiency(self, nav_result: Dict[str, any],
                                 nav_task: Dict[str, any]) -> float:
        """Calculate path efficiency"""
        actual_path_length = nav_result.get('path_length', 0.0)
        optimal_path_length = nav_task.get('optimal_path_length', actual_path_length)

        if optimal_path_length > 0:
            # Efficiency = optimal_length / actual_length (1.0 = perfectly efficient)
            efficiency = optimal_path_length / actual_path_length
            return min(1.0, efficiency)
        else:
            return 0.0

    def count_collisions(self, nav_result: Dict[str, any]) -> int:
        """Count collisions during navigation"""
        return nav_result.get('collision_count', 0)

    def calculate_safety_score(self, collision_counts: List[int]) -> float:
        """Calculate safety score based on collision data"""
        if not collision_counts:
            return 1.0

        avg_collisions = np.mean(collision_counts)
        # Lower collision count = higher safety score
        safety_score = max(0.0, 1.0 - (avg_collisions * 0.2))  # Each collision reduces score by 0.2

        return safety_score

    def calculate_adaptability_score(self, nav_result: Dict[str, any]) -> float:
        """Calculate how well robot adapted to dynamic obstacles"""
        dynamic_obstacles_encountered = nav_result.get('dynamic_obstacles', 0)
        dynamic_obstacles_handled = nav_result.get('dynamic_obstacles_handled', 0)

        if dynamic_obstacles_encountered > 0:
            adaptability = dynamic_obstacles_handled / dynamic_obstacles_encountered
            return min(1.0, adaptability)
        else:
            # If no dynamic obstacles, return neutral score
            return 0.7

class SafetyEvaluator:
    """Evaluate safety performance of VLA actions"""
    def evaluate(self, vla_model, test_dataset) -> Dict[str, any]:
        """Evaluate safety performance"""
        safety_violations = 0
        total_evaluations = 0
        safety_scores = []

        for sample in test_dataset:
            if 'safety_scenario' in sample:
                scenario = sample['safety_scenario']

                with torch.no_grad():
                    safety_analysis = vla_model.evaluate_safety(scenario)

                violations = self.count_safety_violations(safety_analysis, scenario)
                safety_score = self.calculate_safety_score_from_analysis(safety_analysis)

                safety_violations += violations
                safety_scores.append(safety_score)
                total_evaluations += 1

        compliance_rate = 1.0 - (safety_violations / total_evaluations) if total_evaluations > 0 else 1.0
        avg_safety_score = np.mean(safety_scores) if safety_scores else 1.0

        return {
            'safety_compliance_rate': compliance_rate,
            'safety_avg_score': avg_safety_score,
            'safety_violations_count': safety_violations,
            'safety_total_evaluated': total_evaluations,
            'safety_collision_avoidance': self.evaluate_collision_avoidance(safety_analysis),
            'safety_emergency_response': self.evaluate_emergency_response(safety_analysis),
            'safety_human_awareness': self.evaluate_human_awareness(safety_analysis)
        }

    def count_safety_violations(self, safety_analysis: Dict[str, any],
                               scenario: Dict[str, any]) -> int:
        """Count safety violations in analysis"""
        violations = 0

        # Collision violations
        if safety_analysis.get('collision_risk', 0) > 0.5:
            violations += 1

        # Personal space violations
        if safety_analysis.get('personal_space_violation', False):
            violations += 1

        # Force limit violations
        if safety_analysis.get('force_limit_exceeded', False):
            violations += 1

        # Speed limit violations
        if safety_analysis.get('speed_limit_exceeded', False):
            violations += 1

        return violations

    def calculate_safety_score_from_analysis(self, safety_analysis: Dict[str, any]) -> float:
        """Calculate safety score from safety analysis"""
        score = 1.0

        # Penalize for various risk factors
        score -= safety_analysis.get('collision_risk', 0) * 0.3
        score -= safety_analysis.get('force_risk', 0) * 0.2
        score -= safety_analysis.get('velocity_risk', 0) * 0.2
        score -= safety_analysis.get('balance_risk', 0) * 0.3

        return max(0.0, score)  # Clamp to 0-1 range

    def evaluate_collision_avoidance(self, safety_analysis: Dict[str, any]) -> float:
        """Evaluate collision avoidance performance"""
        # This would analyze collision avoidance metrics
        # For now, return a mock evaluation
        return safety_analysis.get('collision_avoidance_success_rate', 0.95)

    def evaluate_emergency_response(self, safety_analysis: Dict[str, any]) -> float:
        """Evaluate emergency response performance"""
        # This would analyze emergency stop and response metrics
        return safety_analysis.get('emergency_response_success_rate', 0.98)

    def evaluate_human_awareness(self, safety_analysis: Dict[str, any]) -> float:
        """Evaluate human awareness and safety"""
        # This would analyze human detection and awareness metrics
        return safety_analysis.get('human_awareness_score', 0.92)
```

## Integration and End-to-End Evaluation

### 1. Multi-Modal Integration Metrics

```python
# Multi-modal integration evaluation
class IntegrationEvaluator:
    """Evaluator for multi-modal integration in VLA systems"""
    def __init__(self):
        self.alignment_evaluator = AlignmentEvaluator()
        self.temporal_coherence_evaluator = TemporalCoherenceEvaluator()
        self.cross_modal_consistency_evaluator = CrossModalConsistencyEvaluator()
        self.task_completion_evaluator = TaskCompletionEvaluator()

    def evaluate(self, vla_model, test_dataset) -> Dict[str, any]:
        """Evaluate integration performance"""
        results = {}

        # Alignment evaluation
        alignment_results = self.alignment_evaluator.evaluate(vla_model, test_dataset)
        results.update(alignment_results)

        # Temporal coherence evaluation
        temporal_results = self.temporal_coherence_evaluator.evaluate(vla_model, test_dataset)
        results.update(temporal_results)

        # Cross-modal consistency evaluation
        consistency_results = self.cross_modal_consistency_evaluator.evaluate(vla_model, test_dataset)
        results.update(consistency_results)

        # Task completion evaluation
        task_results = self.task_completion_evaluator.evaluate(vla_model, test_dataset)
        results.update(task_results)

        # Calculate overall integration score
        results['overall_integration_score'] = self.calculate_integration_score(results)

        return results

    def calculate_integration_score(self, results: Dict[str, any]) -> float:
        """Calculate overall integration score"""
        score = 0.0
        weight_sum = 0.0

        if 'vision_language_alignment' in results:
            score += results['vision_language_alignment'] * 0.3
            weight_sum += 0.3

        if 'temporal_coherence' in results:
            score += results['temporal_coherence'] * 0.25
            weight_sum += 0.25

        if 'cross_modal_consistency' in results:
            score += results['cross_modal_consistency'] * 0.25
            weight_sum += 0.25

        if 'task_completion_rate' in results:
            score += results['task_completion_rate'] * 0.2
            weight_sum += 0.2

        return score / weight_sum if weight_sum > 0 else 0.0

class AlignmentEvaluator:
    """Evaluate alignment between modalities"""
    def evaluate(self, vla_model, test_dataset) -> Dict[str, any]:
        """Evaluate vision-language alignment"""
        alignment_scores = []

        for sample in test_dataset:
            if 'image' in sample and 'command' in sample and 'expected_action' in sample:
                image = sample['image']
                command = sample['command']
                expected_action = sample['expected_action']

                with torch.no_grad():
                    # Get vision and language features separately
                    vision_features = vla_model.vision_component.encode(image)
                    language_features = vla_model.language_component.encode(command)

                    # Get integrated features
                    integrated_features = vla_model.integrate_modalities(
                        vision_features, language_features
                    )

                    # Predict action from integrated features
                    predicted_action = vla_model.action_component.decode(integrated_features)

                # Calculate alignment score
                alignment_score = self.calculate_alignment(
                    vision_features, language_features, predicted_action, expected_action
                )
                alignment_scores.append(alignment_score)

        avg_alignment = np.mean(alignment_scores) if alignment_scores else 0.0

        return {
            'vision_language_alignment': avg_alignment,
            'alignment_std_dev': np.std(alignment_scores) if alignment_scores else 0.0,
            'alignment_min': min(alignment_scores) if alignment_scores else 0.0,
            'alignment_max': max(alignment_scores) if alignment_scores else 0.0,
            'alignment_samples': len(alignment_scores)
        }

    def calculate_alignment(self, vision_features: torch.Tensor,
                           language_features: torch.Tensor,
                           predicted_action: torch.Tensor,
                           expected_action: torch.Tensor) -> float:
        """Calculate alignment between vision, language, and action"""
        # Method 1: Cosine similarity between vision and language features
        vision_lang_similarity = torch.cosine_similarity(
            vision_features.flatten(), language_features.flatten(), dim=0
        ).item()

        # Method 2: Action prediction accuracy
        action_accuracy = 1.0 - torch.mean(
            torch.abs(predicted_action - expected_action)
        ).item()

        # Method 3: Cross-modal consistency (how well integrated features represent both modalities)
        integrated_vision_similarity = torch.cosine_similarity(
            integrated_features.flatten(), vision_features.flatten(), dim=0
        ).item()
        integrated_language_similarity = torch.cosine_similarity(
            integrated_features.flatten(), language_features.flatten(), dim=0
        ).item()
        cross_modal_consistency = (integrated_vision_similarity + integrated_language_similarity) / 2.0

        # Combined alignment score
        alignment_score = (
            0.4 * vision_lang_similarity +
            0.4 * action_accuracy +
            0.2 * cross_modal_consistency
        )

        return alignment_score

class TemporalCoherenceEvaluator:
    """Evaluate temporal coherence in VLA systems"""
    def evaluate(self, vla_model, test_dataset) -> Dict[str, any]:
        """Evaluate temporal coherence across sequences"""
        coherence_scores = []
        sequence_lengths = []

        for sample in test_dataset:
            if 'action_sequence' in sample and 'command_sequence' in sample:
                command_sequence = sample['command_sequence']
                expected_action_sequence = sample['action_sequence']

                predicted_sequence = []
                for command in command_sequence:
                    with torch.no_grad():
                        action = vla_model.predict_action(command)
                        predicted_sequence.append(action)

                coherence_score = self.calculate_sequence_coherence(
                    predicted_sequence, expected_action_sequence
                )
                coherence_scores.append(coherence_score)
                sequence_lengths.append(len(command_sequence))

        avg_coherence = np.mean(coherence_scores) if coherence_scores else 0.0
        avg_length = np.mean(sequence_lengths) if sequence_lengths else 0.0

        return {
            'temporal_coherence': avg_coherence,
            'avg_sequence_length': avg_length,
            'coherence_std_dev': np.std(coherence_scores) if coherence_scores else 0.0,
            'coherence_samples': len(coherence_scores),
            'sequence_stability': self.calculate_sequence_stability(predicted_sequence)
        }

    def calculate_sequence_coherence(self, predicted_sequence: List[torch.Tensor],
                                   expected_sequence: List[torch.Tensor]) -> float:
        """Calculate coherence of action sequence"""
        if len(predicted_sequence) != len(expected_sequence):
            return 0.0

        # Calculate smoothness of predicted sequence
        predicted_smoothness = self.calculate_sequence_smoothness(predicted_sequence)

        # Calculate similarity to expected sequence
        sequence_similarity = self.calculate_sequence_similarity(predicted_sequence, expected_sequence)

        # Combined coherence score
        coherence_score = 0.5 * predicted_smoothness + 0.5 * sequence_similarity

        return coherence_score

    def calculate_sequence_smoothness(self, action_sequence: List[torch.Tensor]) -> float:
        """Calculate smoothness of action sequence"""
        if len(action_sequence) < 2:
            return 1.0

        # Calculate velocity and acceleration smoothness
        velocities = []
        for i in range(1, len(action_sequence)):
            velocity = torch.norm(action_sequence[i] - action_sequence[i-1])
            velocities.append(velocity.item())

        if len(velocities) < 2:
            return 1.0

        accelerations = []
        for i in range(1, len(velocities)):
            acceleration = abs(velocities[i] - velocities[i-1])
            accelerations.append(acceleration)

        # Smoothness = inverse of acceleration variance (higher variance = less smooth)
        if accelerations:
            smoothness = 1.0 / (1.0 + np.var(accelerations))
        else:
            smoothness = 1.0

        return smoothness

    def calculate_sequence_similarity(self, pred_sequence: List[torch.Tensor],
                                    exp_sequence: List[torch.Tensor]) -> float:
        """Calculate similarity between predicted and expected sequences"""
        if len(pred_sequence) != len(exp_sequence):
            return 0.0

        similarities = []
        for pred, exp in zip(pred_sequence, exp_sequence):
            similarity = torch.cosine_similarity(
                pred.flatten(), exp.flatten(), dim=0
            ).item()
            similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def calculate_sequence_stability(self, action_sequence: List[torch.Tensor]) -> float:
        """Calculate stability of action sequence"""
        if len(action_sequence) < 2:
            return 1.0

        # Calculate variation in action space
        actions_tensor = torch.stack(action_sequence)
        variation = torch.std(actions_tensor, dim=0).mean().item()

        # Lower variation = higher stability
        stability = 1.0 / (1.0 + variation)

        return stability

class CrossModalConsistencyEvaluator:
    """Evaluate consistency across modalities"""
    def evaluate(self, vla_model, test_dataset) -> Dict[str, any]:
        """Evaluate cross-modal consistency"""
        consistency_scores = []

        for sample in test_dataset:
            if all(key in sample for key in ['image', 'command', 'expected_outcome']):
                image = sample['image']
                command = sample['command']
                expected_outcome = sample['expected_outcome']

                with torch.no_grad():
                    # Get predictions using different modality combinations
                    vision_only_pred = vla_model.predict_with_vision_only(image)
                    language_only_pred = vla_model.predict_with_language_only(command)
                    multimodal_pred = vla_model.predict_with_multimodal(image, command)

                # Calculate consistency between modalities
                consistency_score = self.calculate_cross_modal_consistency(
                    vision_only_pred, language_only_pred, multimodal_pred, expected_outcome
                )
                consistency_scores.append(consistency_score)

        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0.0

        return {
            'cross_modal_consistency': avg_consistency,
            'consistency_std_dev': np.std(consistency_scores) if consistency_scores else 0.0,
            'consistency_samples': len(consistency_scores),
            'modality_agreement': self.calculate_modality_agreement(consistency_scores)
        }

    def calculate_cross_modal_consistency(self, vision_pred: torch.Tensor,
                                        language_pred: torch.Tensor,
                                        multimodal_pred: torch.Tensor,
                                        expected: torch.Tensor) -> float:
        """Calculate consistency between different modality predictions"""
        # Vision-language agreement
        vision_language_agreement = torch.cosine_similarity(
            vision_pred.flatten(), language_pred.flatten(), dim=0
        ).item()

        # Multimodal consistency with expected
        multimodal_expected_similarity = torch.cosine_similarity(
            multimodal_pred.flatten(), expected.flatten(), dim=0
        ).item()

        # How well multimodal combines vision and language
        vision_multimodal_similarity = torch.cosine_similarity(
            vision_pred.flatten(), multimodal_pred.flatten(), dim=0
        ).item()
        language_multimodal_similarity = torch.cosine_similarity(
            language_pred.flatten(), multimodal_pred.flatten(), dim=0
        ).item()

        # Combined consistency score
        consistency_score = (
            0.3 * vision_language_agreement +
            0.4 * multimodal_expected_similarity +
            0.3 * (vision_multimodal_similarity + language_multimodal_similarity) / 2.0
        )

        return consistency_score

    def calculate_modality_agreement(self, consistency_scores: List[float]) -> float:
        """Calculate agreement between modalities"""
        if not consistency_scores:
            return 0.0

        # High consistency scores indicate good modality agreement
        agreement = np.mean(consistency_scores)
        return agreement

class TaskCompletionEvaluator:
    """Evaluate task completion in VLA systems"""
    def evaluate(self, vla_model, test_dataset) -> Dict[str, any]:
        """Evaluate task completion performance"""
        completion_results = []

        for sample in test_dataset:
            if 'task_description' in sample and 'environment_state' in sample:
                task_desc = sample['task_description']
                env_state = sample['environment_state']

                with torch.no_grad():
                    task_result = vla_model.execute_task(task_desc, env_state)

                completion_results.append(task_result)

        # Calculate task completion metrics
        success_rate = self.calculate_success_rate(completion_results)
        efficiency = self.calculate_efficiency(completion_results)
        robustness = self.calculate_robustness(completion_results)

        return {
            'task_completion_rate': success_rate,
            'task_efficiency': efficiency,
            'task_robustness': robustness,
            'tasks_evaluated': len(completion_results),
            'task_success_count': sum(1 for r in completion_results if r.get('success', False)),
            'task_avg_time': np.mean([r.get('execution_time', 0) for r in completion_results]),
            'task_avg_attempts': np.mean([r.get('attempts', 1) for r in completion_results])
        }

    def calculate_success_rate(self, task_results: List[Dict[str, any]]) -> float:
        """Calculate task completion success rate"""
        if not task_results:
            return 0.0

        successful_tasks = sum(1 for result in task_results if result.get('success', False))
        return successful_tasks / len(task_results)

    def calculate_efficiency(self, task_results: List[Dict[str, any]]) -> float:
        """Calculate task execution efficiency"""
        if not task_results:
            return 0.0

        # Efficiency = success rate / average execution time (normalized)
        success_rate = self.calculate_success_rate(task_results)
        avg_time = np.mean([r.get('execution_time', 1.0) for r in task_results])

        if avg_time > 0:
            efficiency = success_rate / avg_time
            # Normalize to 0-1 scale (assume max reasonable time is 100s)
            return min(1.0, efficiency / 10.0)  # Arbitrary normalization
        else:
            return success_rate

    def calculate_robustness(self, task_results: List[Dict[str, any]]) -> float:
        """Calculate task execution robustness"""
        if not task_results:
            return 0.0

        # Robustness = success rate / (1 + average retries needed)
        total_attempts = sum(r.get('attempts', 1) for r in task_results)
        total_tasks = len(task_results)

        if total_tasks > 0:
            avg_attempts = total_attempts / total_tasks
            success_rate = self.calculate_success_rate(task_results)

            # Robustness penalizes high number of attempts
            robustness = success_rate / avg_attempts
            return min(1.0, robustness)
        else:
            return 0.0
```

## Human-Centered Evaluation Metrics

### 1. Human Satisfaction and Acceptance

```python
# Human-centered evaluation metrics
class HumanSatisfactionEvaluator:
    """Evaluate human satisfaction with VLA system interactions"""
    def __init__(self):
        self.acceptance_metrics = AcceptanceMetrics()
        self.satisfaction_survey = SatisfactionSurvey()
        self.behavior_analysis = BehaviorAnalysis()

    def evaluate_human_interaction(self, vla_model, interaction_logs) -> Dict[str, any]:
        """Evaluate human satisfaction and acceptance"""
        results = {}

        # Analyze interaction logs
        interaction_analysis = self.behavior_analysis.analyze_interactions(interaction_logs)

        # Calculate acceptance metrics
        acceptance_results = self.acceptance_metrics.calculate_acceptance(interaction_analysis)
        results.update(acceptance_results)

        # Process satisfaction surveys (if available)
        if 'satisfaction_data' in interaction_logs:
            survey_results = self.satisfaction_survey.analyze_surveys(
                interaction_logs['satisfaction_data']
            )
            results.update(survey_results)

        # Calculate overall human-centered score
        results['human_centered_score'] = self.calculate_human_centered_score(results)

        return results

    def calculate_human_centered_score(self, results: Dict[str, any]) -> float:
        """Calculate overall human-centered evaluation score"""
        score = 0.0
        weight_sum = 0.0

        if 'acceptance_rate' in results:
            score += results['acceptance_rate'] * 0.4
            weight_sum += 0.4

        if 'satisfaction_score' in results:
            score += results['satisfaction_score'] * 0.3
            weight_sum += 0.3

        if 'interaction_quality' in results:
            score += results['interaction_quality'] * 0.3
            weight_sum += 0.3

        return score / weight_sum if weight_sum > 0 else 0.0

class AcceptanceMetrics:
    """Calculate robot acceptance metrics"""
    def calculate_acceptance(self, interaction_analysis: Dict[str, any]) -> Dict[str, float]:
        """Calculate robot acceptance metrics"""
        # Time spent interacting
        interaction_time = interaction_analysis.get('total_interaction_time', 0)
        positive_interactions = interaction_analysis.get('positive_interactions', 0)
        negative_interactions = interaction_analysis.get('negative_interactions', 0)
        total_interactions = positive_interactions + negative_interactions

        acceptance_rate = positive_interactions / total_interactions if total_interactions > 0 else 0.0

        # Interaction duration quality
        avg_interaction_duration = interaction_time / total_interactions if total_interactions > 0 else 0

        # Acceptance score based on multiple factors
        duration_score = min(1.0, avg_interaction_duration / 60.0)  # Normalize by 60 seconds
        frequency_score = min(1.0, positive_interactions / 10.0)   # Normalize by 10 interactions

        overall_acceptance = (
            0.4 * acceptance_rate +
            0.3 * duration_score +
            0.3 * frequency_score
        )

        return {
            'acceptance_rate': acceptance_rate,
            'overall_acceptance_score': overall_acceptance,
            'interaction_duration_score': duration_score,
            'interaction_frequency_score': frequency_score,
            'positive_interaction_ratio': positive_interactions / total_interactions if total_interactions > 0 else 0.0
        }

class SatisfactionSurvey:
    """Process satisfaction survey data"""
    def analyze_surveys(self, survey_data: List[Dict[str, any]]) -> Dict[str, float]:
        """Analyze satisfaction survey responses"""
        if not survey_data:
            return {'satisfaction_score': 0.5, 'survey_count': 0}

        # Survey questions typically include:
        # 1. Overall satisfaction (1-5 scale)
        # 2. Ease of use (1-5 scale)
        # 3. Naturalness of interaction (1-5 scale)
        # 4. Helpfulness (1-5 scale)
        # 5. Safety perception (1-5 scale)

        overall_satisfaction = np.mean([s.get('overall_satisfaction', 3) for s in survey_data]) / 5.0
        ease_of_use = np.mean([s.get('ease_of_use', 3) for s in survey_data]) / 5.0
        naturalness = np.mean([s.get('naturalness', 3) for s in survey_data]) / 5.0
        helpfulness = np.mean([s.get('helpfulness', 3) for s in survey_data]) / 5.0
        safety_perception = np.mean([s.get('safety_perception', 3) for s in survey_data]) / 5.0

        # Weighted satisfaction score
        satisfaction_score = (
            0.25 * overall_satisfaction +
            0.2 * ease_of_use +
            0.2 * naturalness +
            0.2 * helpfulness +
            0.15 * safety_perception
        )

        return {
            'satisfaction_score': satisfaction_score,
            'overall_satisfaction': overall_satisfaction,
            'ease_of_use_score': ease_of_use,
            'naturalness_score': naturalness,
            'helpfulness_score': helpfulness,
            'safety_perception_score': safety_perception,
            'survey_count': len(survey_data)
        }

class BehaviorAnalysis:
    """Analyze human behavior during interactions"""
    def analyze_interactions(self, interaction_logs: List[Dict[str, any]]) -> Dict[str, any]:
        """Analyze interaction patterns and behaviors"""
        analysis = {
            'total_interaction_time': 0.0,
            'positive_interactions': 0,
            'negative_interactions': 0,
            'neutral_interactions': 0,
            'interaction_patterns': [],
            'response_times': [],
            'task_completion_rates': [],
            'safety_incidents': 0
        }

        for log in interaction_logs:
            # Analyze interaction sentiment
            sentiment = log.get('sentiment', 'neutral')
            if sentiment == 'positive':
                analysis['positive_interactions'] += 1
            elif sentiment == 'negative':
                analysis['negative_interactions'] += 1
            else:
                analysis['neutral_interactions'] += 1

            # Track interaction time
            duration = log.get('duration', 0)
            analysis['total_interaction_time'] += duration

            # Track response times
            response_time = log.get('response_time', 0)
            if response_time > 0:
                analysis['response_times'].append(response_time)

            # Track task completion
            if 'task_success' in log:
                analysis['task_completion_rates'].append(1 if log['task_success'] else 0)

            # Track safety incidents
            if log.get('safety_violation', False):
                analysis['safety_incidents'] += 1

        # Calculate derived metrics
        analysis['average_response_time'] = np.mean(analysis['response_times']) if analysis['response_times'] else float('inf')
        analysis['average_task_completion_rate'] = np.mean(analysis['task_completion_rates']) if analysis['task_completion_rates'] else 0.0

        return analysis

# Benchmark evaluation suites
class VLAEvaluationSuite:
    """Comprehensive VLA evaluation suite"""
    def __init__(self):
        self.benchmarks = {
            'vlbench': VLBenchEvaluator(),
            'robotics_bench': RoboticsBenchEvaluator(),
            'hri_bench': HRIBenchEvaluator(),
            'safety_bench': SafetyBenchEvaluator()
        }

    def run_comprehensive_evaluation(self, vla_model, test_datasets) -> Dict[str, any]:
        """Run comprehensive evaluation using multiple benchmarks"""
        results = {}

        for benchmark_name, evaluator in self.benchmarks.items():
            print(f"Running {benchmark_name} evaluation...")
            benchmark_results = evaluator.evaluate(vla_model, test_datasets[benchmark_name])
            results[benchmark_name] = benchmark_results

        # Calculate composite score
        results['composite_score'] = self.calculate_composite_score(results)

        # Generate executive summary
        results['executive_summary'] = self.generate_executive_summary(results)

        return results

    def calculate_composite_score(self, benchmark_results: Dict[str, any]) -> float:
        """Calculate composite evaluation score"""
        # Weight different benchmarks based on importance
        weights = {
            'vlbench': 0.3,      # Vision-language benchmarks
            'robotics_bench': 0.3,  # Robotics-specific benchmarks
            'hri_bench': 0.25,    # Human-robot interaction benchmarks
            'safety_bench': 0.15   # Safety benchmarks
        }

        composite_score = 0.0
        total_weight = 0.0

        for bench_name, weight in weights.items():
            if bench_name in benchmark_results:
                bench_score = self.extract_benchmark_score(benchmark_results[bench_name])
                composite_score += bench_score * weight
                total_weight += weight

        return composite_score / total_weight if total_weight > 0 else 0.0

    def extract_benchmark_score(self, benchmark_result: Dict[str, any]) -> float:
        """Extract primary score from benchmark result"""
        # Look for common score fields
        for field in ['overall_score', 'accuracy', 'success_rate', 'f1_score']:
            if field in benchmark_result:
                return benchmark_result[field]

        # If no primary score found, return 0.5 as neutral
        return 0.5

    def generate_executive_summary(self, results: Dict[str, any]) -> str:
        """Generate executive summary of evaluation results"""
        summary = f"""
# VLA System Evaluation Summary

## Overall Performance
- **Composite Score**: {results['composite_score']:.3f}
- **Evaluation Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Total Benchmarks**: {len(results) - 2}  # Excluding composite and summary

## Benchmark Results
"""
        for bench_name, bench_results in results.items():
            if bench_name not in ['composite_score', 'executive_summary']:
                primary_score = self.extract_benchmark_score(bench_results)
                summary += f"- **{bench_name.upper()}**: {primary_score:.3f}\n"

        summary += f"""
## Key Strengths
{self.identify_strengths(results)}

## Areas for Improvement
{self.identify_weaknesses(results)}

## Recommendations
{self.generate_recommendations(results)}
        """

        return summary.strip()

    def identify_strengths(self, results: Dict[str, any]) -> str:
        """Identify system strengths"""
        strengths = []

        for bench_name, bench_results in results.items():
            if bench_name not in ['composite_score', 'executive_summary']:
                score = self.extract_benchmark_score(bench_results)
                if score > 0.8:
                    strengths.append(f"- {bench_name} performance: {score:.3f}")

        if not strengths:
            strengths.append("- System shows balanced performance across all benchmarks")

        return "\n".join(strengths)

    def identify_weaknesses(self, results: Dict[str, any]) -> str:
        """Identify system weaknesses"""
        weaknesses = []

        for bench_name, bench_results in results.items():
            if bench_name not in ['composite_score', 'executive_summary']:
                score = self.extract_benchmark_score(bench_results)
                if score < 0.6:
                    weaknesses.append(f"- {bench_name} performance: {score:.3f} (needs improvement)")

        if not weaknesses:
            weaknesses.append("- No significant weaknesses identified (all benchmarks ≥ 0.6)")

        return "\n".join(weaknesses)

    def generate_recommendations(self, results: Dict[str, any]) -> str:
        """Generate improvement recommendations"""
        recommendations = []

        for bench_name, bench_results in results.items():
            if bench_name not in ['composite_score', 'executive_summary']:
                score = self.extract_benchmark_score(bench_results)
                if score < 0.7:
                    if bench_name == 'vlbench':
                        recommendations.append("- Focus on vision-language integration improvements")
                    elif bench_name == 'robotics_bench':
                        recommendations.append("- Improve robotics-specific task execution")
                    elif bench_name == 'hri_bench':
                        recommendations.append("- Enhance human-robot interaction capabilities")
                    elif bench_name == 'safety_bench':
                        recommendations.append("- Strengthen safety mechanisms and protocols")

        if not recommendations:
            recommendations.append("- System performance is strong across all evaluation areas")

        return "\n".join(recommendations)

class VLBenchEvaluator:
    """Evaluator for vision-language benchmarks"""
    def evaluate(self, vla_model, dataset) -> Dict[str, any]:
        """Evaluate on vision-language tasks"""
        # Implementation would include tasks like:
        # - Visual question answering
        # - Object detection with language grounding
        # - Visual reasoning with text
        # - Multimodal understanding tasks

        # For now, return mock results
        return {
            'overall_accuracy': 0.85,
            'vision_language_alignment': 0.82,
            'grounding_accuracy': 0.79,
            'reasoning_score': 0.87,
            'samples_evaluated': len(dataset)
        }

class RoboticsBenchEvaluator:
    """Evaluator for robotics-specific benchmarks"""
    def evaluate(self, vla_model, dataset) -> Dict[str, any]:
        """Evaluate on robotics tasks"""
        # Implementation would include tasks like:
        # - Manipulation tasks
        # - Navigation tasks
        # - Object interaction tasks
        # - Multi-step task execution

        # For now, return mock results
        return {
            'manipulation_success_rate': 0.78,
            'navigation_success_rate': 0.84,
            'task_completion_rate': 0.81,
            'action_accuracy': 0.86,
            'samples_evaluated': len(dataset)
        }

class HRIBenchEvaluator:
    """Evaluator for human-robot interaction benchmarks"""
    def evaluate(self, vla_model, dataset) -> Dict[str, any]:
        """Evaluate on HRI tasks"""
        # Implementation would include tasks like:
        # - Natural language interaction
        # - Social cue recognition
        # - Collaborative task execution
        # - Human intention understanding

        # For now, return mock results
        return {
            'language_understanding': 0.89,
            'social_cue_recognition': 0.83,
            'collaboration_success': 0.76,
            'interaction_naturalness': 0.85,
            'samples_evaluated': len(dataset)
        }

class SafetyBenchEvaluator:
    """Evaluator for safety benchmarks"""
    def evaluate(self, vla_model, dataset) -> Dict[str, any]:
        """Evaluate on safety tasks"""
        # Implementation would include tasks like:
        # - Collision avoidance
        # - Safe navigation
        # - Emergency response
        # - Human safety protocols

        # For now, return mock results
        return {
            'collision_avoidance_rate': 0.98,
            'safety_compliance': 0.95,
            'emergency_response_success': 0.97,
            'human_safety_score': 0.94,
            'samples_evaluated': len(dataset)
        }
```

## Evaluation Dashboard and Reporting

### 1. Real-time Evaluation Dashboard

```python
# Real-time evaluation dashboard
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import numpy as np
import json
import time
from collections import deque
import seaborn as sns

class VLAEvaluationDashboard:
    """Real-time evaluation dashboard for VLA systems"""
    def __init__(self):
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 12))
        self.fig.suptitle('VLA System Real-time Evaluation Dashboard', fontsize=16)

        # Data buffers for plotting
        self.performance_history = {
            'vision_accuracy': deque(maxlen=100),
            'language_accuracy': deque(maxlen=100),
            'action_success': deque(maxlen=100),
            'integration_score': deque(maxlen=100),
            'human_satisfaction': deque(maxlen=100),
            'safety_compliance': deque(maxlen=100)
        }

        # Initialize plots
        self.initialize_plots()

        # Animation
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plots, interval=1000, blit=False
        )

    def initialize_plots(self):
        """Initialize dashboard plots"""
        # Vision accuracy over time
        self.axes[0, 0].set_title('Vision Component Accuracy')
        self.axes[0, 0].set_ylabel('Accuracy')
        self.axes[0, 0].set_ylim(0, 1)
        self.vision_line, = self.axes[0, 0].plot([], [], 'b-', label='Vision Accuracy')

        # Language accuracy over time
        self.axes[0, 1].set_title('Language Component Accuracy')
        self.axes[0, 1].set_ylabel('Accuracy')
        self.axes[0, 1].set_ylim(0, 1)
        self.language_line, = self.axes[0, 1].plot([], [], 'g-', label='Language Accuracy')

        # Action success rate over time
        self.axes[0, 2].set_title('Action Success Rate')
        self.axes[0, 2].set_ylabel('Success Rate')
        self.axes[0, 2].set_ylim(0, 1)
        self.action_line, = self.axes[0, 2].plot([], [], 'r-', label='Action Success')

        # Integration score over time
        self.axes[1, 0].set_title('Integration Score')
        self.axes[1, 0].set_ylabel('Score')
        self.axes[1, 0].set_ylim(0, 1)
        self.integration_line, = self.axes[1, 0].plot([], [], 'm-', label='Integration Score')

        # Human satisfaction over time
        self.axes[1, 1].set_title('Human Satisfaction')
        self.axes[1, 1].set_ylabel('Satisfaction')
        self.axes[1, 1].set_ylim(0, 1)
        self.satisfaction_line, = self.axes[1, 1].plot([], [], 'c-', label='Satisfaction')

        # Safety compliance over time
        self.axes[1, 2].set_title('Safety Compliance')
        self.axes[1, 2].set_ylabel('Compliance')
        self.axes[1, 2].set_ylim(0, 1)
        self.safety_line, = self.axes[1, 2].plot([], [], 'y-', label='Safety Compliance')

        # Add legends
        for ax in self.axes.flat:
            ax.legend()

    def update_plots(self, frame):
        """Update plots with new data"""
        # Get current evaluation results (mock data for demonstration)
        current_results = self.get_current_evaluation_results()

        # Update data buffers
        self.performance_history['vision_accuracy'].append(current_results['vision_accuracy'])
        self.performance_history['language_accuracy'].append(current_results['language_accuracy'])
        self.performance_history['action_success'].append(current_results['action_success_rate'])
        self.performance_history['integration_score'].append(current_results['integration_score'])
        self.performance_history['human_satisfaction'].append(current_results['human_satisfaction'])
        self.performance_history['safety_compliance'].append(current_results['safety_compliance'])

        # Update all plots
        x_data = range(len(self.performance_history['vision_accuracy']))

        self.vision_line.set_data(x_data, list(self.performance_history['vision_accuracy']))
        self.language_line.set_data(x_data, list(self.performance_history['language_accuracy']))
        self.action_line.set_data(x_data, list(self.performance_history['action_success']))
        self.integration_line.set_data(x_data, list(self.performance_history['integration_score']))
        self.satisfaction_line.set_data(x_data, list(self.performance_history['human_satisfaction']))
        self.safety_line.set_data(x_data, list(self.performance_history['safety_compliance']))

        # Adjust x-axis limits
        for ax in self.axes.flat:
            ax.set_xlim(0, max(10, len(x_data)))

        return [self.vision_line, self.language_line, self.action_line,
                self.integration_line, self.satisfaction_line, self.safety_line]

    def get_current_evaluation_results(self) -> Dict[str, float]:
        """Get current evaluation results (mock implementation)"""
        # This would interface with real evaluation system
        # For now, return realistic mock values
        return {
            'vision_accuracy': np.random.normal(0.85, 0.05),
            'language_accuracy': np.random.normal(0.88, 0.04),
            'action_success_rate': np.random.normal(0.82, 0.06),
            'integration_score': np.random.normal(0.84, 0.05),
            'human_satisfaction': np.random.normal(0.80, 0.07),
            'safety_compliance': np.random.normal(0.95, 0.03)
        }

    def add_evaluation_result(self, result: Dict[str, any]):
        """Add evaluation result to dashboard"""
        # Update performance history with new result
        if 'vision_accuracy' in result:
            self.performance_history['vision_accuracy'].append(result['vision_accuracy'])
        if 'language_accuracy' in result:
            self.performance_history['language_accuracy'].append(result['language_accuracy'])
        if 'action_success_rate' in result:
            self.performance_history['action_success'].append(result['action_success_rate'])
        if 'integration_score' in result:
            self.performance_history['integration_score'].append(result['integration_score'])
        if 'human_satisfaction' in result:
            self.performance_history['human_satisfaction'].append(result['human_satisfaction'])
        if 'safety_compliance' in result:
            self.performance_history['safety_compliance'].append(result['safety_compliance'])

    def save_dashboard_report(self, filename: str):
        """Save dashboard report with current data"""
        report_data = {
            'timestamp': time.time(),
            'performance_history': {
                key: list(value) for key, value in self.performance_history.items()
            },
            'current_performance': {
                key: value[-1] if value else 0.0
                for key, value in self.performance_history.items()
            }
        }

        with open(filename, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"Dashboard report saved to {filename}")

    def start_dashboard(self):
        """Start the evaluation dashboard"""
        plt.show()

class EvaluationReportGenerator:
    """Generate comprehensive evaluation reports"""
    def __init__(self):
        self.template = self.load_report_template()

    def load_report_template(self) -> str:
        """Load report template"""
        return """
# VLA System Evaluation Report

## Executive Summary
**Overall Performance Score:** {overall_score}
**Evaluation Period:** {start_date} - {end_date}
**Total Samples Evaluated:** {total_samples}

## Component Performance
### Vision Component
- **Accuracy:** {vision_accuracy}
- **Object Detection F1:** {object_detection_f1}
- **Segmentation IoU:** {segmentation_iou}

### Language Component
- **Command Understanding:** {command_accuracy}
- **Semantic Parsing F1:** {semantic_f1}
- **Natural Language Generation:** {generation_bleu}

### Action Component
- **Task Success Rate:** {task_success_rate}
- **Action Accuracy:** {action_accuracy}
- **Execution Time:** {execution_time}s

### Integration Component
- **Vision-Language Alignment:** {alignment_score}
- **Multi-Modal Consistency:** {consistency_score}
- **Human Satisfaction:** {human_satisfaction}

## Safety and Compliance
- **Collision Avoidance Rate:** {collision_avoidance_rate}
- **Safety Compliance:** {safety_compliance}
- **Emergency Response Success:** {emergency_response_success}

## Recommendations
{suggestions}

## Detailed Metrics
{detailed_metrics}
        """

    def generate_report(self, evaluation_results: Dict[str, any]) -> str:
        """Generate evaluation report"""
        report_data = {
            'overall_score': evaluation_results.get('overall_score', 0.0),
            'start_date': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(evaluation_results.get('start_time', time.time()))),
            'end_date': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(evaluation_results.get('end_time', time.time()))),
            'total_samples': evaluation_results.get('total_samples', 0),
            'vision_accuracy': evaluation_results.get('vision', {}).get('accuracy', 0.0),
            'object_detection_f1': evaluation_results.get('vision', {}).get('object_detection_f1', 0.0),
            'segmentation_iou': evaluation_results.get('vision', {}).get('segmentation_iou', 0.0),
            'command_accuracy': evaluation_results.get('language', {}).get('command_accuracy', 0.0),
            'semantic_f1': evaluation_results.get('language', {}).get('semantic_f1', 0.0),
            'generation_bleu': evaluation_results.get('language', {}).get('generation_bleu', 0.0),
            'task_success_rate': evaluation_results.get('action', {}).get('success_rate', 0.0),
            'action_accuracy': evaluation_results.get('action', {}).get('accuracy', 0.0),
            'execution_time': evaluation_results.get('action', {}).get('avg_execution_time', 0.0),
            'alignment_score': evaluation_results.get('integration', {}).get('alignment_score', 0.0),
            'consistency_score': evaluation_results.get('integration', {}).get('consistency_score', 0.0),
            'human_satisfaction': evaluation_results.get('integration', {}).get('human_satisfaction', 0.0),
            'collision_avoidance_rate': evaluation_results.get('safety', {}).get('collision_avoidance_rate', 0.0),
            'safety_compliance': evaluation_results.get('safety', {}).get('compliance_rate', 0.0),
            'emergency_response_success': evaluation_results.get('safety', {}).get('emergency_success_rate', 0.0),
            'suggestions': self.generate_suggestions(evaluation_results),
            'detailed_metrics': self.format_detailed_metrics(evaluation_results)
        }

        return self.template.format(**report_data)

    def generate_suggestions(self, results: Dict[str, any]) -> str:
        """Generate improvement suggestions based on results"""
        suggestions = []

        # Vision component suggestions
        vision_acc = results.get('vision', {}).get('accuracy', 0.0)
        if vision_acc < 0.8:
            suggestions.append("- Vision component needs improvement (current: {:.3f})".format(vision_acc))

        # Language component suggestions
        lang_acc = results.get('language', {}).get('command_accuracy', 0.0)
        if lang_acc < 0.85:
            suggestions.append("- Language understanding needs enhancement (current: {:.3f})".format(lang_acc))

        # Action component suggestions
        action_success = results.get('action', {}).get('success_rate', 0.0)
        if action_success < 0.75:
            suggestions.append("- Action execution success rate needs improvement (current: {:.3f})".format(action_success))

        # Safety suggestions
        safety_compliance = results.get('safety', {}).get('compliance_rate', 0.0)
        if safety_compliance < 0.95:
            suggestions.append("- Safety compliance needs strengthening (current: {:.3f})".format(safety_compliance))

        if not suggestions:
            suggestions.append("- System performance is satisfactory across all components")

        return "\n".join([f"- {suggestion}" for suggestion in suggestions])

    def format_detailed_metrics(self, results: Dict[str, any]) -> str:
        """Format detailed metrics for report"""
        detailed = []

        for component, metrics in results.items():
            if isinstance(metrics, dict):
                detailed.append(f"### {component.title()}")
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        detailed.append(f"- **{metric_name}:** {metric_value:.3f}")
                    else:
                        detailed.append(f"- **{metric_name}:** {metric_value}")
                detailed.append("")  # Empty line for spacing

        return "\n".join(detailed)

    def save_report(self, report: str, filename: str):
        """Save report to file"""
        with open(filename, 'w') as f:
            f.write(report)
        print(f"Evaluation report saved to {filename}")
```

## Continuous Evaluation and Adaptation

### 1. Online Learning and Adaptation

```python
# Online learning and adaptation for VLA systems
class OnlineAdaptationSystem:
    """System for continuous learning and adaptation"""
    def __init__(self, base_model, evaluation_threshold=0.7):
        self.base_model = base_model
        self.evaluation_threshold = evaluation_threshold
        self.performance_history = deque(maxlen=100)
        self.adaptation_history = []
        self.feedback_collector = FeedbackCollector()

    def evaluate_and_adapt(self, current_performance: float, feedback: Dict[str, any]):
        """Evaluate performance and trigger adaptation if needed"""
        self.performance_history.append(current_performance)

        # Check if performance has degraded
        if self.performance_degraded():
            # Trigger adaptation
            adaptation_result = self.trigger_adaptation(feedback)
            self.adaptation_history.append(adaptation_result)

            # Log adaptation
            print(f"Adaptation triggered due to performance degradation. New performance: {adaptation_result['new_performance']}")

    def performance_degraded(self) -> bool:
        """Check if performance has degraded beyond threshold"""
        if len(self.performance_history) < 10:
            return False  # Need more data

        # Calculate rolling average
        recent_avg = np.mean(list(self.performance_history)[-10:])
        historical_avg = np.mean(list(self.performance_history)[:-10]) if len(self.performance_history) > 10 else recent_avg

        # Performance is degraded if recent average is significantly worse than historical
        degradation_threshold = 0.05  # 5% performance drop
        return (historical_avg - recent_avg) > degradation_threshold

    def trigger_adaptation(self, feedback: Dict[str, any]) -> Dict[str, any]:
        """Trigger adaptation based on feedback"""
        # Determine adaptation strategy based on feedback
        adaptation_strategy = self.select_adaptation_strategy(feedback)

        # Apply adaptation
        if adaptation_strategy == 'fine_tuning':
            adaptation_result = self.fine_tune_model(feedback['training_data'])
        elif adaptation_strategy == 'hyperparameter_adjustment':
            adaptation_result = self.adjust_hyperparameters(feedback['performance_data'])
        elif adaptation_strategy == 'architecture_modification':
            adaptation_result = self.modify_architecture(feedback['error_analysis'])
        else:
            adaptation_result = self.incremental_learning(feedback['new_examples'])

        # Evaluate new performance
        new_performance = self.assess_performance(adaptation_result['updated_model'])

        return {
            'strategy': adaptation_strategy,
            'adaptation_success': True,
            'new_performance': new_performance,
            'adaptation_time': time.time() - adaptation_result['start_time'],
            'model_version': adaptation_result.get('model_version', 'unknown')
        }

    def select_adaptation_strategy(self, feedback: Dict[str, any]) -> str:
        """Select appropriate adaptation strategy based on feedback"""
        error_analysis = feedback.get('error_analysis', {})
        performance_data = feedback.get('performance_data', {})

        # Determine adaptation type based on error patterns
        if error_analysis.get('systematic_bias', False):
            return 'fine_tuning'
        elif performance_data.get('variance_high', False):
            return 'hyperparameter_adjustment'
        elif error_analysis.get('architectural_limitations', False):
            return 'architecture_modification'
        else:
            return 'incremental_learning'

    def fine_tune_model(self, training_data: List[Dict[str, any]]) -> Dict[str, any]:
        """Fine-tune model with new training data"""
        start_time = time.time()

        # Prepare training data
        train_loader = self.prepare_training_data(training_data)

        # Fine-tune model
        self.base_model.train()
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=1e-5)

        for epoch in range(5):  # Few epochs for fine-tuning
            for batch in train_loader:
                optimizer.zero_grad()

                images = batch['images'].cuda()
                commands = batch['commands'].cuda()
                actions = batch['actions'].cuda()

                with torch.cuda.amp.autocast():
                    pred_actions = self.base_model(images, commands)
                    loss = torch.nn.functional.mse_loss(pred_actions, actions)

                loss.backward()
                optimizer.step()

        return {
            'updated_model': self.base_model,
            'start_time': start_time,
            'model_version': f"{self.get_current_version()}_ft_{int(time.time())}"
        }

    def adjust_hyperparameters(self, performance_data: Dict[str, any]) -> Dict[str, any]:
        """Adjust model hyperparameters based on performance data"""
        start_time = time.time()

        # Analyze performance data to determine hyperparameter adjustments
        if performance_data.get('overfitting_detected', False):
            # Increase regularization
            self.increase_regularization()
        elif performance_data.get('underfitting_detected', False):
            # Decrease regularization, increase capacity
            self.decrease_regularization()
            self.increase_model_capacity()

        # Adjust learning rate based on convergence patterns
        if performance_data.get('convergence_slow', False):
            self.increase_learning_rate()
        elif performance_data.get('oscillating', False):
            self.decrease_learning_rate()

        return {
            'updated_model': self.base_model,
            'start_time': start_time,
            'model_version': f"{self.get_current_version()}_hp_{int(time.time())}"
        }

    def increase_regularization(self):
        """Increase model regularization"""
        # Add more dropout, increase weight decay, etc.
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.p = min(0.5, module.p + 0.1)  # Increase dropout rate

    def decrease_regularization(self):
        """Decrease model regularization"""
        for module in self.base_model.modules():
            if isinstance(module, nn.Dropout):
                module.p = max(0.1, module.p - 0.1)  # Decrease dropout rate

    def increase_model_capacity(self):
        """Increase model capacity"""
        # This would involve architectural changes
        # For now, just log the need for capacity increase
        print("Model capacity increase needed - consider adding layers or neurons")

    def increase_learning_rate(self):
        """Increase learning rate"""
        # Find and update optimizer learning rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = min(1e-3, param_group['lr'] * 1.1)  # Increase by 10%

    def decrease_learning_rate(self):
        """Decrease learning rate"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(1e-6, param_group['lr'] * 0.9)  # Decrease by 10%

    def get_current_version(self) -> str:
        """Get current model version"""
        # This would interface with model versioning system
        return "v1.0.0"

    def incremental_learning(self, new_examples: List[Dict[str, any]]) -> Dict[str, any]:
        """Perform incremental learning with new examples"""
        start_time = time.time()

        # Add new examples to experience replay buffer
        for example in new_examples:
            self.experience_buffer.add(example)

        # Perform incremental training on new examples
        if len(self.experience_buffer) > 100:  # Minimum buffer size
            # Sample from buffer and train
            batch = self.experience_buffer.sample(32)
            self.train_on_batch(batch)

        return {
            'updated_model': self.base_model,
            'start_time': start_time,
            'model_version': f"{self.get_current_version()}_il_{int(time.time())}",
            'examples_learned': len(new_examples)
        }

    def train_on_batch(self, batch: List[Dict[str, any]]):
        """Train model on a batch of examples"""
        self.base_model.train()
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=1e-4)

        images = torch.stack([ex['images'] for ex in batch]).cuda()
        commands = torch.stack([ex['commands'] for ex in batch]).cuda()
        actions = torch.stack([ex['actions'] for ex in batch]).cuda()

        optimizer.zero_grad()
        pred_actions = self.base_model(images, commands)
        loss = torch.nn.functional.mse_loss(pred_actions, actions)
        loss.backward()
        optimizer.step()

    def assess_performance(self, model) -> float:
        """Assess performance of updated model"""
        # This would evaluate the model on validation data
        # For now, return a mock assessment
        return np.random.normal(0.85, 0.05)  # Mock performance score

    def prepare_training_data(self, raw_data: List[Dict[str, any]]) -> DataLoader:
        """Prepare training data for fine-tuning"""
        # This would convert raw data to appropriate format
        # For now, return a mock data loader
        pass

class FeedbackCollector:
    """Collect feedback for continuous adaptation"""
    def __init__(self):
        self.feedback_buffer = deque(maxlen=1000)
        self.feedback_analyzer = FeedbackAnalyzer()

    def collect_feedback(self, interaction_result: Dict[str, any]):
        """Collect feedback from interaction"""
        feedback = {
            'timestamp': time.time(),
            'interaction_result': interaction_result,
            'user_rating': interaction_result.get('user_satisfaction', 0.5),
            'task_success': interaction_result.get('task_completed', False),
            'error_analysis': self.analyze_interaction_errors(interaction_result),
            'performance_indicators': self.extract_performance_indicators(interaction_result)
        }

        self.feedback_buffer.append(feedback)

    def analyze_interaction_errors(self, result: Dict[str, any]) -> Dict[str, any]:
        """Analyze errors in interaction"""
        error_analysis = {
            'command_misunderstanding': result.get('command_error', False),
            'action_failure': result.get('action_error', False),
            'timing_issue': result.get('timing_error', False),
            'safety_violation': result.get('safety_error', False),
            'communication_breakdown': result.get('communication_error', False)
        }

        return error_analysis

    def extract_performance_indicators(self, result: Dict[str, any]) -> Dict[str, float]:
        """Extract performance indicators from result"""
        indicators = {
            'response_time': result.get('response_time', 1.0),
            'accuracy': result.get('accuracy', 0.5),
            'efficiency': result.get('efficiency', 0.5),
            'safety_score': result.get('safety_score', 1.0),
            'user_satisfaction': result.get('user_satisfaction', 0.5)
        }

        return indicators

    def get_adaptation_feedback(self) -> Dict[str, any]:
        """Get feedback for adaptation decisions"""
        if not self.feedback_buffer:
            return {}

        # Analyze recent feedback for adaptation triggers
        recent_feedback = list(self.feedback_buffer)[-50:]  # Last 50 interactions

        performance_trend = self.analyze_performance_trend(recent_feedback)
        error_patterns = self.identify_error_patterns(recent_feedback)
        user_satisfaction_trend = self.analyze_satisfaction_trend(recent_feedback)

        return {
            'performance_trend': performance_trend,
            'error_patterns': error_patterns,
            'satisfaction_trend': user_satisfaction_trend,
            'training_data': self.extract_training_examples(recent_feedback),
            'error_analysis': self.aggregate_error_analysis(recent_feedback)
        }

    def analyze_performance_trend(self, feedback_list: List[Dict[str, any]]) -> str:
        """Analyze performance trend"""
        if len(feedback_list) < 10:
            return 'insufficient_data'

        recent_performance = [f['performance_indicators'].get('accuracy', 0.5) for f in feedback_list[-10:]]
        earlier_performance = [f['performance_indicators'].get('accuracy', 0.5) for f in feedback_list[:10]]

        recent_avg = np.mean(recent_performance)
        earlier_avg = np.mean(earlier_performance)

        if recent_avg > earlier_avg + 0.05:
            return 'improving'
        elif recent_avg < earlier_avg - 0.05:
            return 'degrading'
        else:
            return 'stable'

    def identify_error_patterns(self, feedback_list: List[Dict[str, any]]) -> Dict[str, int]:
        """Identify common error patterns"""
        error_counts = {
            'command_misunderstanding': 0,
            'action_failure': 0,
            'timing_issue': 0,
            'safety_violation': 0,
            'communication_breakdown': 0
        }

        for feedback in feedback_list:
            error_analysis = feedback.get('error_analysis', {})
            for error_type, occurred in error_analysis.items():
                if occurred:
                    error_counts[error_type] += 1

        return error_counts

    def analyze_satisfaction_trend(self, feedback_list: List[Dict[str, any]]) -> str:
        """Analyze user satisfaction trend"""
        if len(feedback_list) < 10:
            return 'insufficient_data'

        recent_satisfaction = [f['user_rating'] for f in feedback_list[-10:]]
        earlier_satisfaction = [f['user_rating'] for f in feedback_list[:10]]

        recent_avg = np.mean(recent_satisfaction)
        earlier_avg = np.mean(earlier_satisfaction)

        if recent_avg > earlier_avg + 0.1:
            return 'improving'
        elif recent_avg < earlier_avg - 0.1:
            return 'degrading'
        else:
            return 'stable'

    def extract_training_examples(self, feedback_list: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Extract examples for training from feedback"""
        training_examples = []

        for feedback in feedback_list:
            if feedback.get('task_success', False) and feedback.get('user_rating', 0) > 0.7:
                # Positive example
                interaction_result = feedback['interaction_result']
                training_examples.append({
                    'images': interaction_result.get('input_images'),
                    'commands': interaction_result.get('input_command'),
                    'actions': interaction_result.get('executed_actions'),
                    'success': True
                })
            elif not feedback.get('task_success', True) and feedback.get('user_rating', 1) < 0.3:
                # Negative example for learning from mistakes
                interaction_result = feedback['interaction_result']
                training_examples.append({
                    'images': interaction_result.get('input_images'),
                    'commands': interaction_result.get('input_command'),
                    'actions': interaction_result.get('executed_actions'),
                    'success': False,
                    'correction': interaction_result.get('expected_actions')  # If available
                })

        return training_examples

    def aggregate_error_analysis(self, feedback_list: List[Dict[str, any]]) -> Dict[str, any]:
        """Aggregate error analysis across feedback"""
        if not feedback_list:
            return {}

        # Count different error types
        error_type_counts = {}
        for feedback in feedback_list:
            error_analysis = feedback.get('error_analysis', {})
            for error_type, occurred in error_analysis.items():
                if occurred:
                    error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1

        # Calculate error frequencies
        total_feedback = len(feedback_list)
        error_frequencies = {
            error_type: count / total_feedback
            for error_type, count in error_type_counts.items()
        }

        # Identify most common errors
        most_common_errors = sorted(
            error_frequencies.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Top 3 errors

        return {
            'error_frequencies': error_frequencies,
            'most_common_errors': most_common_errors,
            'total_interactions': total_feedback,
            'error_rate': sum(error_type_counts.values()) / total_feedback if total_feedback > 0 else 0.0
        }
```

## Next Steps

In the next section, we'll explore deployment strategies and operational considerations for VLA systems in real-world humanoid robotics applications, including edge deployment, real-time performance optimization, and field validation techniques.