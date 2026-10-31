"""Declared dose, uneven windows and fractional epoch bounds."""
from hephaestus.training.batch_schedule import training_windows, cursor_at_step, step_limit


def test_uneven_epoch_flushes_once_without_wrapping():
    assert list(training_windows(3, 2, 10, 1)) == [(0, 1), (2,)]
    assert list(training_windows(3, 2, 10, 1.5)) == [(0, 1), (2,), (0, 1)]
    assert list(training_windows(3, 2, 10, 0.1)) == [(0,)]
    assert list(training_windows(3, 2, 3, None)) == [(0, 1), (2,), (0, 1)]
    assert step_limit(100, 1, 100, 0.07) == 7
    assert step_limit(100, 2, 5, 1e308) == 5


def test_resume_cursor_matches_consumed_batches_for_every_window():
    for count in (1, 3, 5):
        for accumulation in (1, 2, 7):
            for epochs in (None, 0.1, 1, 1.5, 2):
                windows = list(training_windows(count, accumulation, 8, epochs))
                assert len(windows) == step_limit(count, accumulation, 8, epochs)
                for step in range(len(windows) + 1):
                    assert cursor_at_step(count, accumulation, step, epochs) == sum(map(len, windows[:step]))
