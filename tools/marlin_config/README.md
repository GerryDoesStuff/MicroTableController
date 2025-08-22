
MicroStageController Marlin Config Snippet

- Drop Configuration_snippet.h contents into your Marlin Configuration.h and merge.
- Ensure CUSTOM_MACHINE_NAME and MACHINE_UUID match the values expected by the app.
- Compute DEFAULT_AXIS_STEPS_PER_UNIT from your current M92 values using:
  X *= 860.0; Y *= 420.0; Z *= 43.4/1.15 ≈ 37.73913
- After flashing, tune with M92 adjustments, then bake into firmware.
