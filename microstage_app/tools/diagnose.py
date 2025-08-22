import sys, os, time, datetime
from pathlib import Path

def main():
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outdir = Path(os.getcwd()) / 'diagnostics'
    outdir.mkdir(exist_ok=True)
    log_path = outdir / f'diag_{ts}.txt'
    def log(*a):
        s = ' '.join(str(x) for x in a)
        print(s)
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(s + '\n')

    log('=== MicroStage Diagnostics ===', ts)
    log('Python:', sys.version)
    # Modules
    for mod in ['serial', 'toupcam', 'numpy', 'PySide6', 'opencv-python']:
        try:
            __import__(mod if mod != 'opencv-python' else 'cv2')
            log(f'[OK] module available:', mod)
        except Exception as e:
            log(f'[WARN] module missing:', mod, '|', repr(e))

    # Serial
    log('\n-- Serial / Marlin discovery --')
    try:
        import serial
        from serial.tools import list_ports
        from ..config import EXPECTED_MACHINE_NAME, EXPECTED_MACHINE_UUID
        ports = list(list_ports.comports())
        if not ports:
            log('[INFO] No COM ports found.')
        for p in ports:
            log(f'Port: {p.device} | VID:PID={p.vid}:{p.pid} | desc="{p.description}" | hwid="{p.hwid}"')
            try:
                ser = serial.Serial(p.device, baudrate=115200, timeout=0.6, write_timeout=0.6)
                ser.reset_input_buffer()
                ser.write(b'M115\n')
                time.sleep(0.3)
                resp = ser.read(4096).decode(errors='ignore')
                ser.close()
                low = resp.lower()
                if 'firmware_name:marlin' in low:
                    if EXPECTED_MACHINE_NAME and EXPECTED_MACHINE_NAME.lower() not in low:
                        log(
                            f"[WARN] {p.device} Marlin but machine name mismatch. Response snippet: {resp[:120]!r}"
                        )
                    elif EXPECTED_MACHINE_UUID and EXPECTED_MACHINE_UUID.lower() not in low:
                        log(
                            f"[WARN] {p.device} Marlin with matching machine name but UUID mismatch. Response snippet: {resp[:120]!r}"
                        )
                    else:
                        log(f'[OK] Marlin matched on {p.device}')
                else:
                    log(f'[INFO] No Marlin signature on {p.device}. Response snippet: {resp[:120]!r}')
            except Exception as e:
                log(f'[ERR] Could not open/query {p.device}: {e}')
    except Exception as e:
        log('[ERR] Serial discovery failed:', e)

    # Camera / Toupcam
    log('\n-- Camera / ToupCam SDK --')
    try:
        import toupcam
        log('[OK] toupcam module imported.')
        try:
            devs = toupcam.Toupcam.EnumV2()
            log(f'[INFO] EnumV2 found {len(devs)} devices.')
            if devs:
                cam = toupcam.Toupcam.Open(devs[0].id)
                log('[OK] Opened first device.')
                try:
                    cam.StartPullModeWithCallback(lambda evt: None)
                    log('[OK] Started pull mode.')
                    # Try to get size and pull one image
                    try:
                        w, h = cam.get_Size()
                        log(f'[INFO] Reported size: {w}x{h}')
                        try:
                            _ = cam.PullImageV3(w, h, 24)
                            log('[OK] PullImageV3 returned some data.')
                        except Exception as e:
                            log('[WARN] PullImageV3 failed:', e)
                    except Exception as e:
                        log('[WARN] Could not get size:', e)
                    cam.Stop()
                except Exception as e:
                    log('[ERR] StartPullModeWithCallback failed:', e)
                finally:
                    try:
                        cam.Close()
                    except Exception as e:
                        log('[WARN] camera close failed:', e)
                    cam = None
            else:
                log('[INFO] No ToupCam devices detected by EnumV2().')
        except Exception as e:
            log('[ERR] toupcam enumeration/open failed:', e)
    except Exception as e:
        log('[WARN] toupcam module not importable (SDK/DLL missing?):', e)

    log('\nDone. Log saved at:', str(log_path))

if __name__ == '__main__':
    main()
