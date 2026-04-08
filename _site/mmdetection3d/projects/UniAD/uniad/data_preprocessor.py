"""Queue-aware data preprocessor for temporal training."""

import torch
from mmdet3d.registry import MODELS
from mmdet3d.models import Det3DDataPreprocessor


class TemporalQueue:
    """Opaque container for multi-frame queue data.

    Wraps a list of frame dicts so pseudo_collate doesn't recurse into it.
    Each frame dict has 'inputs' and 'data_samples' from Pack3DDetInputs.
    """

    def __init__(self, frames):
        self.frames = frames  # list of Q frame dicts


@MODELS.register_module()
class QueueDet3DDataPreprocessor(Det3DDataPreprocessor):
    """Extends Det3DDataPreprocessor with temporal queue support.

    During training with temporal queue:
    - Extracts TemporalQueue from data dict
    - Processes each frame through parent's forward independently
    - Returns inputs_queue / data_samples_queue for model

    During validation/test or single-frame training:
    - Falls back to standard Det3DDataPreprocessor behavior
    """

    def forward(self, data, training=False):
        # Check for temporal queue
        temporal_queue = None
        if isinstance(data, dict) and 'temporal_queue' in data:
            tq_list = data.pop('temporal_queue')
            if isinstance(tq_list, list):
                temporal_queue = tq_list[0]
            else:
                temporal_queue = tq_list

        if temporal_queue is not None and training:
            return self._process_queue(temporal_queue, training)

        # Normal single-frame processing
        return super().forward(data, training)

    def _process_queue(self, temporal_queue, training):
        """Process each queue frame through parent's preprocessing."""
        frames = temporal_queue.frames

        inputs_queue = []
        data_samples_queue = []

        for frame_data in frames:
            # Build single-item batch format matching pseudo_collate output
            single_batch = {
                'inputs': {'points': [frame_data['inputs']['points']]},
                'data_samples': [frame_data['data_samples']],
            }
            processed = super().forward(single_batch, training)
            inputs_queue.append(processed['inputs'])
            data_samples_queue.append(processed['data_samples'][0])

        return {
            'inputs_queue': inputs_queue,
            'data_samples_queue': data_samples_queue,
        }
