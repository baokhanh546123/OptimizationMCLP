# Fix applied (2026-09-16)

Bug: after `reduce_candidates_geographic` / `by_coverage`, `chosen_candidates`
were indices in reduced J', but visualize used original candidate_id → heatmap offset.

Fix in `src/backend/model/optimization.py`:
1. `_candidate_index_map` stored on reduce
2. `_remap_results_to_original_indices()` maps chosen → original ids before return
3. `chosen_candidates_reduced` kept for debug

**Important:** Ensure `src/backend/model/optimization.py` is the full ~740-line file
(includes `build_candidate_set`, `epsilon_constraint_sweep`, `_remap_results_to_original_indices`, `_report_summary`, etc.).
If the file was truncated mid-push, restore from the complete source that contains
both reduce methods setting `self._candidate_index_map` and the remap call before `_report_summary`.
