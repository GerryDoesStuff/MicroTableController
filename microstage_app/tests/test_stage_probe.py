def test_no_ports(monkeypatch):
    import serial.tools.list_ports
    from microstage_app.devices.stage_marlin import find_marlin_port
    monkeypatch.setattr(serial.tools.list_ports, "comports", lambda: [])
    assert find_marlin_port() is None