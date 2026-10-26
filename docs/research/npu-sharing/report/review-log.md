# Presentation revision log

## 2026-10-21: v0.2 review draft

The user authorized outline revision and deck generation. The v0.1 outlines are
preserved in `archive/`. The deck `npu-sharing-2026-10-21-v02-draft.pptx` contains 18
main slides and two appendix slides. It awaits user review.

The three core mechanisms remain the organizing structure. New material covers R025,
R027, inference graphs, equal-TP and multi-replica validation. Performance charts and
tables are editable. Existing mechanism diagrams are embedded raster illustrations.
Per-slide speaker notes identify source records. No universal non-regression, isolated
migration benefit or training-quality equivalence claim is made.

### Megatron integration boundary

Failed workspace:
`/root/areal-benchmarks/coverage-20261021-131754-0700/20261021-134022-0700_math_megatron_c16_b6_a9428e`.

Fixed completed. Shared failed at the first weight publication with
`Cannot change replica epoch with outstanding leases`. R027 omitted the shared snapshot
when the next batch was ready. Megatron incorrectly selected its generic publication
path based on snapshot creation rather than fixed-receiver destination. Generic resume
reopened admission before the trainer advanced the version.

The repair selects fixed-only publication through `npu_share_fixed`, controls snapshot
creation independently, joins old snapshot readers and leaves admission reopening with
the trainer. Thirty tests in `test_megatron_shared_publication.py` and
`test_megatron_shared_routing.py` passed. NPU end-to-end confirmation remains pending.
The repair has no performance result yet.

### Review status

Awaiting feedback. No commit or push requested. Historical experiment outcomes and
frozen sources remain unchanged.

## v0.3 — Huawei template styling

Applied the user-supplied template: mountain cover, red headings, gray tables and Huawei
master footer. Retained all 20 research slides and evidence boundaries. Charts and
tables remain editable. All slides were rendered in PowerPoint and visually inspected;
package, layout and native-chart validation passed. No commit or push.
