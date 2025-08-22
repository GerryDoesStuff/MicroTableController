import time, serial
for b in (250000,115200):
    try:
        with serial.Serial("COM4", baudrate=b, timeout=1) as s:
            time.sleep(2.0); s.reset_input_buffer(); s.write(b"M115\n"); time.sleep(0.3)
            print(b, "->", s.read(4096).decode(errors="ignore")[:120])
    except Exception as e:
        print(b, "ERR:", e)