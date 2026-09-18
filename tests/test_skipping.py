from skipgs import SkipController


def test_skipping_starts_after_warmup_and_roundtrips_state():
    controller = SkipController(start_iter=10, warmup=1, min_bwd_ratio=0)
    assert not controller.decide(0, 1.0, 9)
    assert not controller.decide(0, 1.0, 10)
    assert controller.decide(0, 0.5, 11)
    restored = SkipController(start_iter=10, warmup=1, min_bwd_ratio=0)
    restored.load_state_dict(controller.state_dict())
    assert restored.summary() == controller.summary()
    assert restored.decide(0, 0.4, 12) == controller.decide(0, 0.4, 12)
