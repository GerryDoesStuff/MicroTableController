import serial
import serial.tools.list_ports
from microstage_app.devices.stage_marlin import find_marlin_port


def test_no_ports(monkeypatch):
    monkeypatch.setattr(serial.tools.list_ports, "comports", lambda: [])
    assert find_marlin_port() is None


def test_identifiers_match(monkeypatch):
    from microstage_app.devices import stage_marlin

    class Port:
        device = "COMX"
        vid = 0x1A86
        pid = 0x7523

    class DummySerial:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def reset_input_buffer(self):
            pass

        def write(self, data):
            pass

        def read(self, n):
            return (
                b"FIRMWARE_NAME:Marlin MACHINE_TYPE:MicroStageController UUID:a3a4637a-68c4-4340-9fda-847b4fe0d3fc\nok\n"
            )

    monkeypatch.setattr(stage_marlin.list_ports, "comports", lambda: [Port()])
    monkeypatch.setattr(stage_marlin.serial, "Serial", DummySerial)
    monkeypatch.setattr(stage_marlin.time, "sleep", lambda x: None)

    assert find_marlin_port() == "COMX"


def test_identifiers_match_with_spaces(monkeypatch):
    from microstage_app.devices import stage_marlin

    class Port:
        device = "COMS"
        vid = 0x1A86
        pid = 0x7523

    class DummySerial:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def reset_input_buffer(self):
            pass

        def write(self, data):
            pass

        def read(self, n):
            return (
                b"FIRMWARE_NAME:Marlin MACHINE_TYPE: MicroStageController UUID: a3a4637a-68c4-4340-9fda-847b4fe0d3fc\nok\n"
            )

    monkeypatch.setattr(stage_marlin.list_ports, "comports", lambda: [Port()])
    monkeypatch.setattr(stage_marlin.serial, "Serial", DummySerial)
    monkeypatch.setattr(stage_marlin.time, "sleep", lambda x: None)

    assert find_marlin_port() == "COMS"


def test_machine_name_mismatch(monkeypatch):
    from microstage_app.devices import stage_marlin

    class Port:
        device = "COMY"
        vid = 0x1A86
        pid = 0x7523

    class DummySerial:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def reset_input_buffer(self):
            pass

        def write(self, data):
            pass

        def read(self, n):
            return b"FIRMWARE_NAME:Marlin MACHINE_TYPE:Other UUID:0000\nok\n"

    monkeypatch.setattr(stage_marlin.list_ports, "comports", lambda: [Port()])
    monkeypatch.setattr(stage_marlin.serial, "Serial", DummySerial)
    monkeypatch.setattr(stage_marlin.time, "sleep", lambda x: None)

    assert find_marlin_port() is None


def test_uuid_mismatch_still_accepts(monkeypatch):
    from microstage_app.devices import stage_marlin

    class Port:
        device = "COMY"
        vid = 0x1A86
        pid = 0x7523

    class DummySerial:
        def __init__(self, *a, **k):
            pass

        # Use context manager methods to support 'with'
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def reset_input_buffer(self):
            pass

        def write(self, data):
            pass

        def read(self, n):
            return b"FIRMWARE_NAME:Marlin MACHINE_TYPE:MicroStageController UUID:0000\nok\n"

    monkeypatch.setattr(stage_marlin.list_ports, "comports", lambda: [Port()])
    monkeypatch.setattr(stage_marlin.serial, "Serial", DummySerial)
    monkeypatch.setattr(stage_marlin.time, "sleep", lambda x: None)

    assert find_marlin_port() == "COMY"


def test_prefers_uuid_when_multiple(monkeypatch):
    from microstage_app.devices import stage_marlin

    class PortA:
        device = "COMA"
        vid = 0x1A86
        pid = 0x7523

    class PortB:
        device = "COMB"
        vid = 0x1A86
        pid = 0x7523

    responses = {
        "COMA": b"FIRMWARE_NAME:Marlin MACHINE_TYPE:MicroStageController UUID:0000\nok\n",
        "COMB": b"FIRMWARE_NAME:Marlin MACHINE_TYPE:MicroStageController UUID:a3a4637a-68c4-4340-9fda-847b4fe0d3fc\nok\n",
    }

    class DummySerial:
        def __init__(self, port, *a, **k):
            self.port = port

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def reset_input_buffer(self):
            pass

        def write(self, data):
            pass

        def read(self, n):
            return responses[self.port]

    monkeypatch.setattr(stage_marlin.list_ports, "comports", lambda: [PortA(), PortB()])
    monkeypatch.setattr(stage_marlin.serial, "Serial", DummySerial)
    monkeypatch.setattr(stage_marlin.time, "sleep", lambda x: None)

    assert (
        find_marlin_port(
            machine_name="MicroStageController",
            machine_uuid="a3a4637a-68c4-4340-9fda-847b4fe0d3fc",
        )
        == "COMB"
    )
