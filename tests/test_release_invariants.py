from __future__ import annotations

from epilepsy_eeg.paths import MODEL_ORDER_TASK1, MODEL_ORDER_TASK2


def test_model_orders_are_frozen() -> None:
    assert MODEL_ORDER_TASK1 == ("eegwavenet_sc", "physio_sequence_no_aux", "physio_sequence_stats")
    assert MODEL_ORDER_TASK2 == ("eegwavenet_sc", "physio_tf", "physio_stats", "eegnetv4", "eegconformer")
