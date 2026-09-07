# Wheat Threshold Experiment

The opt-in `AgentConfig.wheat_harvest_threshold` flag enables a fixed wheat
minimum eligibility threshold of 3 units. It defaults to `False`, so existing
executor behavior and fertilizer forecasts are unchanged unless the flag is
selected.

With the flag enabled, subthreshold wheat remains planted while useful growth
is mechanically possible. Watering remains available and existing
water-before-harvest dependencies are preserved. Yield 3 or 4 is ordinary
productive harvest work, not a priority boost. Expiry and the final actionable
turn can permit a subthreshold harvest; manager-directed DIG/replacement and
feed purchasing remain independent of the threshold.

The threshold-aware fertilizer forecast models untreated and treated wheat
separately, respecting held yield, current watering, the three-day fertilizer
duration, the crop cap, and the remaining horizon. Each modeled path stops at
its assumed harvest threshold. This is an ideal-upkeep approximation; actual
collection can occur later because the scheduler may lack capacity.

The evaluator retains `baseline` and `care`, and adds:

| Arm | Care | Fertilizer | Wheat threshold |
| --- | --- | --- | --- |
| `fertilizer` | off | on | off |
| `fertilizer_wheat3` | off | on | on |
| `combined` | on | on | off |
| `combined_wheat3` | on | on | on |

Threshold comparisons are recorded explicitly as
`fertilizer_wheat3 -> fertilizer` and `combined_wheat3 -> combined`. The
four-arm Kaggle handoff runs 16 games with the requested seeds, both seats,
master seed 25, official engine, replay capture, executor traces, audit, and a
downloadable ZIP. Run `notebooks/wheat_threshold_capture_audit.ipynb` only
after the checkpoint paths and Kaggle Secret are available.

Local tests use repository fixtures and one scripted fast-engine lifecycle
check. Fast-engine evidence is diagnostic only. No trained checkpoint,
official Kaggle gameplay, performance improvement, or gameplay conclusion is
claimed locally.
