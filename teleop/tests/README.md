# teleop/tests

Pytest suites for the parts of `teleop/` that can be tested without
robotics dependencies. The first test added here (`test_fsm.py`) sets the
pattern for future tests of FSM-shaped or pure-logic teleop code.

## Running

From the `xr_teleoperate` checkout root:

```bash
python -m pytest teleop/tests/ -v
```

Or a single file:

```bash
python -m pytest teleop/tests/test_fsm.py -v
```

## Why these tests run on a clean Mac

The teleop entry point (`teleop/teleop_hand_and_arm.py`) imports DDS, Vuer,
multiprocessing-aware logging, and a few hardware controllers at module load
time. None of those install cleanly on a developer Mac, so `import
teleop.teleop_hand_and_arm` fails with `ModuleNotFoundError` before any test
code runs.

To make the FSM testable, `on_press` and the FSM state globals were extracted
into `teleop/fsm.py`, which imports only Python stdlib. That module is what
`test_fsm.py` exercises. Tests that need the heavy imports belong in a
separate file with `pytest.importorskip` guards or container-only marks —
don't move them next to the FSM tests.

## Adding new tests

When testing other module-level state machines or pure-logic helpers:

1. If the unit under test currently lives in a module that can't import on
   Mac, extract it into its own module under `teleop/` first (mirror the
   `teleop/fsm.py` shape).
2. Add the test file under `teleop/tests/test_<unit>.py`.
3. Use `pytest.fixture(autouse=True)` to reset module globals between tests
   — see `_reset_fsm_state` in `test_fsm.py`.
4. Capture log output via `caplog`; set the module's logger level explicitly
   so info-level messages aren't dropped.
