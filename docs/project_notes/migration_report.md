# Migration report

- Created at: 2026-05-05T13:36:59

- Dry run: False

## Summary

- copied: 73

- missing: 1


## Notes

- Large raw and processed datasets are intentionally not copied.

- Legacy scripts are copied to `pipelines/legacy/` or mapped to numbered pipeline entry points.

- Further refactoring should move reusable logic from pipeline scripts into `src/ecm_program_atlas/`.
