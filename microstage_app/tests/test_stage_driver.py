import serial
from microstage_app.devices.stage_marlin import StageMarlin

class DummySerial:
    def __init__(self, *args, **kwargs):
        self.writes = []
    def write(self, data):
        self.writes.append(data.decode())
    def read(self, n):
        return b''
    def readline(self):
        return b'ok\n'
    def reset_input_buffer(self):
        pass


def test_move_commands(monkeypatch):
    dummy = DummySerial()
    monkeypatch.setattr(serial, 'Serial', lambda *a, **k: dummy)
    stage = StageMarlin('COMX')
    dummy.writes.clear()
    stage.move_relative(dx=1.0, dy=-2.0, dz=0.5, feed_mm_per_min=123, wait_ok=True)
    assert dummy.writes[0].strip() == 'G91'
    cmd = dummy.writes[1]
    assert 'X1.0000' in cmd and 'Y-2.0000' in cmd and 'Z0.5000' in cmd and 'F123.00' in cmd
    assert dummy.writes[2].strip() == 'G90'
    dummy.writes.clear()
    stage.move_absolute(x=2.0, z=-1.0, feed_mm_per_min=150, wait_ok=True)
    assert dummy.writes[0].strip() == 'G90'
    cmd = dummy.writes[1]
    assert cmd.startswith('G1') and 'X2.0000' in cmd and 'Z-1.0000' in cmd and 'F150.00' in cmd
