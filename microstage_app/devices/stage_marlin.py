import time
import serial
from serial.tools import list_ports
from ..utils.log import log

BAUD_DEFAULT = 250000
BAUD_FALLBACK = 115200

def _ts():
    return time.perf_counter()

class ProbeError(Exception):
    pass

def _probe_port(device: str, baud: int, time_wait: float = 2.0) -> bool:
    try:
        log(f"Stage: probing {device} @ {baud}")
        with serial.Serial(device, baudrate=baud, timeout=1.0, write_timeout=1.0) as ser:
            time.sleep(time_wait)  # Arduino auto-reset on open; give it time
            ser.reset_input_buffer()
            ser.write(b"M115\n")
            time.sleep(0.3)
            resp = ser.read(4096).decode(errors="ignore")
            if "FIRMWARE_NAME:Marlin" in resp:
                log(f"Stage: Marlin detected on {device} @ {baud}")
                return True
            else:
                log(f"Stage: {device} not Marlin (got {resp[:120]!r})")
                return False
    except Exception as e:
        log(f"Stage: probe error {device}@{baud}: {e}")
        return False

def find_marlin_port(time_wait: float = 2.0):
    ports = list(list_ports.comports())

    def score(p):
        s = 0
        if str(p.device).upper() == "COM1":
            s -= 100
        if getattr(p, "vid", None) == 0x1A86 and getattr(p, "pid", None) == 0x7523:
            s += 100
        return s

    ports.sort(key=score, reverse=True)

    for p in ports:
        if not p.device:
            continue
        # Try default first, then fallback
        if _probe_port(p.device, BAUD_DEFAULT, time_wait):
            return (p.device, BAUD_DEFAULT)
        if _probe_port(p.device, BAUD_FALLBACK, time_wait):
            return (p.device, BAUD_FALLBACK)

    log("Stage: no Marlin device found")
    return None

class StageMarlin:
    def __init__(self, port: str, baud: int = BAUD_DEFAULT, timeout: float = 1.0):
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout, write_timeout=timeout)
        log(f"Stage: opened {port} @ {baud}")
        time.sleep(2.0)
        self._drain_input()
        self._send_log("M110 N0")
        self.absolute_mode()

    def _drain_input(self):
        try:
            junk = self.ser.read(8192)
            if junk:
                log(f"Stage RX (drain) {len(junk)}B")
        except Exception:
            pass

    def _send_log(self, cmd: str):
        line = (cmd.strip() + "\n").encode()
        log(f"TX >> {cmd}")
        self.ser.write(line)

    def _read_until_ok(self, overall_timeout: float = 3.0):
        start = _ts()
        buf = ""
        while (_ts() - start) < overall_timeout:
            b = self.ser.readline()
            if not b:
                continue
            s = b.decode(errors="ignore")
            buf += s
            s_stripped = s.strip()
            if s_stripped:
                log(f"RX << {s_stripped}")
            if "ok" in s.lower():
                log(f"Stage: ok in {(_ts()-start):.3f}s")
                return buf
        log(f"Stage: timeout waiting ok after {overall_timeout:.1f}s")
        return buf

    def send(self, cmd: str, wait_ok: bool = True):
        self._send_log(cmd)
        return self._read_until_ok() if wait_ok else ""

    def home_xyz(self):
        return self.send("G28")

    def absolute_mode(self):
        return self.send("G90", wait_ok=True)

    def relative_mode(self):
        return self.send("G91", wait_ok=True)

    def move_relative(self, dx=0, dy=0, dz=0, feed_mm_per_min=600, wait_ok=False):
        self.relative_mode()
        parts = ["G1"]
        if dx:
            parts.append(f"X{dx:.4f}")
        if dy:
            parts.append(f"Y{dy:.4f}")
        if dz:
            parts.append(f"Z{dz:.4f}")
        parts.append(f"F{feed_mm_per_min:.2f}")
        self.send(" ".join(parts), wait_ok=wait_ok)
        self.absolute_mode()

    def move_absolute(self, x=None, y=None, z=None, feed_mm_per_min=600, wait_ok=False):
        parts = ["G1"]
        if x is not None:
            parts.append(f"X{x:.4f}")
        if y is not None:
            parts.append(f"Y{y:.4f}")
        if z is not None:
            parts.append(f"Z{z:.4f}")
        parts.append(f"F{feed_mm_per_min:.2f}")
        self.send("G90", wait_ok=True)
        self.send(" ".join(parts), wait_ok=wait_ok)

    def wait_for_moves(self, timeout_s: float = 5.0):
        self.send("M400", wait_ok=True)
