import time, serial
from serial.tools import list_ports
from microstage_app.devices.stage_marlin import parse_m115_info

for p in list_ports.comports():
    for b in (250000,115200):
        try:
            with serial.Serial(p.device, baudrate=b, timeout=1) as s:
                time.sleep(2.0)
                s.reset_input_buffer()
                s.write(b"M115\n"); time.sleep(0.3)
                resp = s.read(4096).decode(errors="ignore")
                info = parse_m115_info(resp)
                print(f"{p.device} @ {b}: {info} -> {resp.splitlines()[:2]}")
        except Exception as e:
            print(f"{p.device} @ {b}: ERR {e}")

for b in (250000,115200):
    try:
        with serial.Serial("COM4", baudrate=b, timeout=1) as s:
            time.sleep(2.0); s.reset_input_buffer(); s.write(b"M115\n"); time.sleep(0.3)
            print(b, "->", s.read(4096).decode(errors="ignore")[:120])
    except Exception as e:
        print(b, "ERR:", e)