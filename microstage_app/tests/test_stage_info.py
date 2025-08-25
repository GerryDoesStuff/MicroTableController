from microstage_app.devices.stage_marlin import StageMarlin


def test_get_info_parses_new_tokens():
    response = (
        "FIRMWARE_NAME:Marlin MACHINE_TYPE:MicroStageController "
        "UUID:a3a4637a-68c4-4340-9fda-847b4fe0d3fc\nok\n"
    )
    s = StageMarlin.__new__(StageMarlin)
    s.send = lambda cmd, wait_ok=True: response
    info = StageMarlin.get_info(s)
    assert info["machine_type"] == "MicroStageController"
    assert info["uuid"] == "a3a4637a-68c4-4340-9fda-847b4fe0d3fc"


def test_get_info_handles_spaces_after_colon():
    response = (
        "FIRMWARE_NAME:Marlin MACHINE_TYPE: MicroStageController "
        "UUID: a3a4637a-68c4-4340-9fda-847b4fe0d3fc\nok\n"
    )
    s = StageMarlin.__new__(StageMarlin)
    s.send = lambda cmd, wait_ok=True: response
    info = StageMarlin.get_info(s)
    assert info["machine_type"] == "MicroStageController"
    assert info["uuid"] == "a3a4637a-68c4-4340-9fda-847b4fe0d3fc"

