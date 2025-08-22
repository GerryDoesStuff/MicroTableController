# Task List

1. ✅ Triage repo vs README and align content
Updated README to reflect tools/tests, headless notes, install from source; verified directory layout; removed stale samples earlier
2. ✅ Add packaging (pyproject) and console script
pyproject.toml added and committed; console script microstage-app registered
3. ✅ Run tests and ensure cross-platform
Moved tests to microstage_app/tools/tests; added raster path tests; all tests pass (4/4) on Linux headless
4. ✅ Add GitHub Actions CI for headless Linux
Added .github/workflows/ci.yml to install deps, run pytest, and run diagnostics with Qt offscreen
5. ✅ Add LICENSE and CHANGELOG
MIT LICENSE and initial CHANGELOG.md added
6. ✅ Remove irrelevant files and move ad-hoc scripts
Earlier commit removed samples; tools/ has cam_probe, marlin_probe; .gitignore expanded to ignore caches, runs/, profiles.yaml
7. ⏳ Add rotating file logging
Planned enhancement to enrich logs and write to rotating file
8. ⏳ Audit SerialWorker usage and shutdown paths
Ensure all serial I/O uses the worker and app closes threads cleanly
9. ⏳ Add more unit tests (autofocus metrics, stage feedrate conversion, mock serial)
Broaden coverage as per pending list
10. ⏳ Integrate plane subtraction into autofocus flows
Wire focus_planes into autofocus routines
11. ⏳ Improve profiles UX (load/save multiple profiles)
Extend profiles beyond single default

