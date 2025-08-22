
from microstage_app.devices.stage_marlin import parse_m115_info, is_expected_board, EXPECTED_MACHINE_NAME, EXPECTED_UUID

def test_parse_m115_and_expected_board():
    sample = f"FIRMWARE_NAME:Marlin 2.1.2 (Github) MACHINE_NAME:{EXPECTED_MACHINE_NAME} UUID:{EXPECTED_UUID}"
    info = parse_m115_info(sample)
    assert info['is_marlin'] is True
    assert info['machine_name'] == EXPECTED_MACHINE_NAME
    assert info['uuid'] == EXPECTED_UUID
    assert is_expected_board(info) is True

    generic = "FIRMWARE_NAME:Marlin 2.1.2 (Github) MACHINE_NAME:Ender3 UUID:deadbeef"
    info2 = parse_m115_info(generic)
    assert info2['is_marlin'] is True
    assert is_expected_board(info2) is False
