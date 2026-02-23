"""Evaluation metrics for speech segmentation."""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """Metrics for evaluating speech segmentation performance."""
    
    def __init__(self, config):
        """Initialize metrics calculator.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.collar_tolerance = config.evaluation.collar_tolerance
        
    def compute_metrics(self, 
                       predictions: np.ndarray, 
                       ground_truth: np.ndarray,
                       sample_rate: int) -> Dict[str, float]:
        """Compute comprehensive segmentation metrics.
        
        Args:
            predictions: Predicted segmentation (binary array)
            ground_truth: Ground truth segmentation (binary array)
            sample_rate: Sample rate of audio
            
        Returns:
            Dict containing various metrics
        """
        metrics = {}
        
        # Frame-level metrics
        frame_metrics = self._compute_frame_metrics(predictions, ground_truth)
        metrics.update(frame_metrics)
        
        # Event-level metrics
        event_metrics = self._compute_event_metrics(predictions, ground_truth, sample_rate)
        metrics.update(event_metrics)
        
        # DER (Diarization Error Rate)
        der = self._compute_der(predictions, ground_truth, sample_rate)
        metrics['der'] = der
        
        # JER (Jaccard Error Rate)
        jer = self._compute_jer(predictions, ground_truth, sample_rate)
        metrics['jer'] = jer
        
        return metrics
    
    def _compute_frame_metrics(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """Compute frame-level metrics.
        
        Args:
            predictions: Predicted segmentation
            ground_truth: Ground truth segmentation
            
        Returns:
            Dict containing frame-level metrics
        """
        # Ensure same length
        min_len = min(len(predictions), len(ground_truth))
        predictions = predictions[:min_len]
        ground_truth = ground_truth[:min_len]
        
        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth, predictions, average='binary', zero_division=0
        )
        
        # Accuracy
        accuracy = np.mean(predictions == ground_truth)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(ground_truth, predictions, labels=[0, 1]).ravel()
        
        # Specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'frame_precision': precision,
            'frame_recall': recall,
            'frame_f1': f1,
            'frame_accuracy': accuracy,
            'frame_specificity': specificity,
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
    
    def _compute_event_metrics(self, predictions: np.ndarray, ground_truth: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Compute event-level metrics with collar tolerance.
        
        Args:
            predictions: Predicted segmentation
            ground_truth: Ground truth segmentation
            sample_rate: Sample rate of audio
            
        Returns:
            Dict containing event-level metrics
        """
        # Convert to time segments
        pred_segments = self._array_to_segments(predictions, sample_rate)
        gt_segments = self._array_to_segments(ground_truth, sample_rate)
        
        # Compute collar tolerance in frames
        collar_frames = int(self.collar_tolerance * sample_rate / self.config.data.hop_length)
        
        # Match segments with collar tolerance
        matched_pred, matched_gt, unmatched_pred, unmatched_gt = self._match_segments(
            pred_segments, gt_segments, self.collar_tolerance
        )
        
        # Event-level precision, recall, F1
        total_pred = len(pred_segments)
        total_gt = len(gt_segments)
        matched = len(matched_pred)
        
        precision = matched / total_pred if total_pred > 0 else 0
        recall = matched / total_gt if total_gt > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'event_precision': precision,
            'event_recall': recall,
            'event_f1': f1,
            'total_predicted_events': total_pred,
            'total_ground_truth_events': total_gt,
            'matched_events': matched,
            'unmatched_predictions': len(unmatched_pred),
            'unmatched_ground_truth': len(unmatched_gt)
        }
    
    def _compute_der(self, predictions: np.ndarray, ground_truth: np.ndarray, sample_rate: int) -> float:
        """Compute Diarization Error Rate (DER).
        
        Args:
            predictions: Predicted segmentation
            ground_truth: Ground truth segmentation
            sample_rate: Sample rate of audio
            
        Returns:
            DER value
        """
        # Convert to time segments
        pred_segments = self._array_to_segments(predictions, sample_rate)
        gt_segments = self._array_to_segments(ground_truth, sample_rate)
        
        # Calculate total duration
        total_duration = len(predictions) * self.config.data.hop_length / sample_rate
        
        # Calculate overlap durations
        overlap_duration = 0
        for pred_start, pred_end in pred_segments:
            for gt_start, gt_end in gt_segments:
                overlap_start = max(pred_start, gt_start)
                overlap_end = min(pred_end, gt_end)
                if overlap_start < overlap_end:
                    overlap_duration += overlap_end - overlap_start
        
        # Calculate DER
        pred_duration = sum(end - start for start, end in pred_segments)
        gt_duration = sum(end - start for start, end in gt_segments)
        
        if total_duration == 0:
            return 1.0
        
        der = (pred_duration + gt_duration - 2 * overlap_duration) / total_duration
        return der
    
    def _compute_jer(self, predictions: np.ndarray, ground_truth: np.ndarray, sample_rate: int) -> float:
        """Compute Jaccard Error Rate (JER).
        
        Args:
            predictions: Predicted segmentation
            ground_truth: Ground truth segmentation
            sample_rate: Sample rate of audio
            
        Returns:
            JER value
        """
        # Convert to time segments
        pred_segments = self._array_to_segments(predictions, sample_rate)
        gt_segments = self._array_to_segments(ground_truth, sample_rate)
        
        # Calculate intersection and union
        intersection_duration = 0
        union_duration = 0
        
        # Calculate intersection
        for pred_start, pred_end in pred_segments:
            for gt_start, gt_end in gt_segments:
                overlap_start = max(pred_start, gt_start)
                overlap_end = min(pred_end, gt_end)
                if overlap_start < overlap_end:
                    intersection_duration += overlap_end - overlap_start
        
        # Calculate union
        all_segments = pred_segments + gt_segments
        all_segments.sort()
        
        union_duration = 0
        if all_segments:
            current_start, current_end = all_segments[0]
            for start, end in all_segments[1:]:
                if start <= current_end:
                    current_end = max(current_end, end)
                else:
                    union_duration += current_end - current_start
                    current_start, current_end = start, end
            union_duration += current_end - current_start
        
        # Calculate JER
        if union_duration == 0:
            return 1.0 if len(pred_segments) > 0 or len(gt_segments) > 0 else 0.0
        
        jer = 1 - (intersection_duration / union_duration)
        return jer
    
    def _array_to_segments(self, array: np.ndarray, sample_rate: int) -> List[Tuple[float, float]]:
        """Convert binary array to time segments.
        
        Args:
            array: Binary segmentation array
            sample_rate: Sample rate of audio
            
        Returns:
            List of (start_time, end_time) tuples
        """
        segments = []
        start = None
        
        for i, value in enumerate(array):
            if value == 1 and start is None:
                start = i
            elif value == 0 and start is not None:
                end_time = (i - 1) * self.config.data.hop_length / sample_rate
                start_time = start * self.config.data.hop_length / sample_rate
                segments.append((start_time, end_time))
                start = None
        
        if start is not None:
            end_time = (len(array) - 1) * self.config.data.hop_length / sample_rate
            start_time = start * self.config.data.hop_length / sample_rate
            segments.append((start_time, end_time))
        
        return segments
    
    def _match_segments(self, pred_segments: List[Tuple[float, float]], 
                       gt_segments: List[Tuple[float, float]], 
                       tolerance: float) -> Tuple[List, List, List, List]:
        """Match predicted and ground truth segments with tolerance.
        
        Args:
            pred_segments: Predicted segments
            gt_segments: Ground truth segments
            tolerance: Time tolerance for matching
            
        Returns:
            Tuple of (matched_pred, matched_gt, unmatched_pred, unmatched_gt)
        """
        matched_pred = []
        matched_gt = []
        unmatched_pred = list(pred_segments)
        unmatched_gt = list(gt_segments)
        
        for pred_seg in pred_segments:
            for gt_seg in unmatched_gt:
                # Check if segments overlap within tolerance
                pred_start, pred_end = pred_seg
                gt_start, gt_end = gt_seg
                
                # Calculate overlap
                overlap_start = max(pred_start, gt_start)
                overlap_end = min(pred_end, gt_end)
                
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    pred_duration = pred_end - pred_start
                    gt_duration = gt_end - gt_start
                    
                    # Check if overlap is significant
                    if overlap_duration >= tolerance:
                        matched_pred.append(pred_seg)
                        matched_gt.append(gt_seg)
                        unmatched_pred.remove(pred_seg)
                        unmatched_gt.remove(gt_seg)
                        break
        
        return matched_pred, matched_gt, unmatched_pred, unmatched_gt


class MetricsAggregator:
    """Aggregator for computing average metrics across multiple samples."""
    
    def __init__(self):
        """Initialize metrics aggregator."""
        self.metrics_history = []
    
    def add_metrics(self, metrics: Dict[str, float]) -> None:
        """Add metrics for a single sample.
        
        Args:
            metrics: Metrics dictionary
        """
        self.metrics_history.append(metrics)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average metrics across all samples.
        
        Returns:
            Dict containing average metrics
        """
        if not self.metrics_history:
            return {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in self.metrics_history:
            all_keys.update(metrics.keys())
        
        # Calculate averages
        average_metrics = {}
        for key in all_keys:
            values = [metrics.get(key, 0) for metrics in self.metrics_history]
            average_metrics[key] = np.mean(values)
        
        return average_metrics
    
    def get_std_metrics(self) -> Dict[str, float]:
        """Get standard deviation of metrics across all samples.
        
        Returns:
            Dict containing standard deviation of metrics
        """
        if not self.metrics_history:
            return {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in self.metrics_history:
            all_keys.update(metrics.keys())
        
        # Calculate standard deviations
        std_metrics = {}
        for key in all_keys:
            values = [metrics.get(key, 0) for metrics in self.metrics_history]
            std_metrics[key] = np.std(values)
        
        return std_metrics
    
    def reset(self) -> None:
        """Reset metrics history."""
        self.metrics_history = []
