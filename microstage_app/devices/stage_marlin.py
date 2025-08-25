import time
import re
import serial
from serial.tools import list_ports
from ..utils.log import log
from ..config import EXPECTED_MACHINE_NAME, EXPECTED_MACHINE_UUID

BAUD = 250000  # fixed baud

def find_marlin_port(
    time_wait: float = 2.0,
    machine_name: str = EXPECTED_MACHINE_NAME,
    machine_uuid: str = EXPECTED_MACHINE_UUID,
):
    """Find the serial port running the expected Marlin stage.

    The custom MACHINE_NAME is treated as the *required* identifier so the
    app only connects to boards built for this project.  If ``machine_uuid`` is
    provided it is used to disambiguate between multiple compatible boards, but
    a UUID mismatch will not prevent a connection as long as the name matches.
    """

    ports = list(list_ports.comports())
    # prefer CH340/1A86:7523; push COM1 to the end
    def score(p):
        s = 0
        if str(p.device).upper() == "COM1":
            s -= 100
        if getattr(p, "vid", None) == 0x1A86 and getattr(p, "pid", None) == 0x7523:
            s += 100
        return s

    ports.sort(key=score, reverse=True)

    fallback = None  # first port with matching name but wrong/missing UUID

    for p in ports:
        if not p.device:
            continue
        try:
            log(f"Stage: probing {p.device} @ {BAUD}")
            with serial.Serial(p.device, baudrate=BAUD, timeout=1.0, write_timeout=1.0) as ser:
                time.sleep(time_wait)  # Arduino auto-reset on open; give it time
                ser.reset_input_buffer()
                ser.write(b"M115\n")
                time.sleep(0.3)
                resp = ser.read(4096).decode(errors="ignore")
                low = resp.lower()
                if "firmware_name:marlin" not in low:
                    log(f"Stage: {p.device} not Marlin (got {resp[:120]!r})")
                    continue

                if machine_name and machine_name.lower() not in low:
                    log(
                        f"Stage: {p.device} Marlin but machine name mismatch (got {resp[:120]!r})"
                    )
                    continue

                if machine_uuid and machine_uuid.lower() in low:
                    log(
                        f"Stage: Marlin detected on {p.device} @ {BAUD} "
                        f"(name={machine_name}, uuid={machine_uuid})"
                    )
                    return p.device

                if fallback is None:
                    log(
                        f"Stage: {p.device} has matching machine name but UUID mismatch "
                        f"(got {resp[:120]!r})"
                    )
                    fallback = p.device
        except Exception as e:
            log(f"Stage: probe error {p.device}@{BAUD}: {e}")

    if fallback:
        log(
            f"Stage: using {fallback} with matching machine name despite UUID mismatch"
        )
        return fallback

    log("Stage: no Marlin device found")
    return None

class StageMarlin:
    def __init__(self, port, baud=BAUD, timeout=1.0):
        from ..utils.log import log
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout, write_timeout=timeout)
        log(f"Stage: opened {port} @ {baud}")
        time.sleep(2.0)  # give the board time after auto-reset
        self._drain_input()
        self._send_log("M110 N0")  # reset line numbers
        self.absolute_mode()
        
    def _drain_input(self):
        try:
            junk = self.ser.read(8192)
            if junk:
                log(f"Stage RX (drain) {len(junk)}B")
        except Exception:
            pass

    def _send_log(self, cmd):
        line = cmd.strip()
        log(f"TX >> {line}")
        self._write(line)

    def _write(self, cmd):
        self.ser.reset_input_buffer()
        self.ser.write((cmd.strip() + '\n').encode())

    def _read_until_ok(self):
        t0 = time.time(); buf = ''
        while time.time() - t0 < 5.0:
            chunk = self.ser.readline().decode(errors='ignore')
            if not chunk: continue
            buf += chunk
            if 'ok' in chunk.lower():
                return buf
        return buf

    def send(self, cmd, wait_ok=True):
        self._send_log(cmd)
        return self._read_until_ok() if wait_ok else ""

    def home_all(self):
        """Home Z first, then X and Y to avoid crashing into optics."""
        self.send("G28 Z")
        return self.send("G28 X Y")
    def home_x(self):       return self.send("G28 X")
    def home_y(self):       return self.send("G28 Y")
    def home_z(self):       return self.send("G28 Z")
    def absolute_mode(self):return self.send("G90", wait_ok=True)
    def relative_mode(self):return self.send("G91", wait_ok=True)

    def move_relative(self, dx=0, dy=0, dz=0, feed_mm_per_min=600, wait_ok=False):
        self.relative_mode()
        parts = ["G1"]
        if dx: parts.append(f"X{dx:.4f}")
        if dy: parts.append(f"Y{dy:.4f}")
        if dz: parts.append(f"Z{dz:.4f}")
        parts.append(f"F{feed_mm_per_min:.2f}")
        self.send(" ".join(parts), wait_ok=wait_ok)
        self.absolute_mode()

    def move_absolute(self, x=None, y=None, z=None, feed_mm_per_min=600, wait_ok=False):
        parts = ["G1"]
        if x is not None: parts.append(f"X{x:.4f}")
        if y is not None: parts.append(f"Y{y:.4f}")
        if z is not None: parts.append(f"Z{z:.4f}")
        parts.append(f"F{feed_mm_per_min:.2f}")
        self.send("G90", wait_ok=True)
        self.send(" ".join(parts), wait_ok=wait_ok)

    # --------------------------- QUERY ---------------------------

    def get_info(self):
        resp = self.send("M115")
        machine_type = None
        name = None
        uuid = None
        # tokens may contain spaces after ':' so parse with regex/semicolon split
        parts = re.findall(
            r"(MACHINE_TYPE|MACHINE_NAME|UUID|MACHINE_UUID)\s*:\s*([^\s;]+)",
            resp,
            flags=re.IGNORECASE,
        )
        for key, value in parts:
            key = key.upper()
            value = value.strip()
            if key == "MACHINE_TYPE":
                machine_type = value
            elif key == "MACHINE_NAME":
                name = value
            elif key in ("UUID", "MACHINE_UUID"):
                uuid = value
        info = {"machine_type": machine_type, "uuid": uuid, "raw": resp}
        if name is not None:
            info["name"] = name
        return info

    def get_bounds(self):
        resp = self.send("M211")
        xmin = ymin = zmin = xmax = ymax = zmax = None
        for line in resp.splitlines():
            line = line.strip()
            if line.lower().startswith("min"):
                parts = line.split()
                for p in parts:
                    if p.startswith("X:"):
                        xmin = float(p[2:])
                    elif p.startswith("Y:"):
                        ymin = float(p[2:])
                    elif p.startswith("Z:"):
                        zmin = float(p[2:])
            elif line.lower().startswith("max"):
                parts = line.split()
                for p in parts:
                    if p.startswith("X:"):
                        xmax = float(p[2:])
                    elif p.startswith("Y:"):
                        ymax = float(p[2:])
                    elif p.startswith("Z:"):
                        zmax = float(p[2:])
        return {
            "xmin": xmin, "xmax": xmax,
            "ymin": ymin, "ymax": ymax,
            "zmin": zmin, "zmax": zmax,
        }

    def get_position(self):
        resp = self.send("M114")
        # Some Marlin builds use lowercase 'count' while others use 'Count';
        # strip everything after this token case-insensitively so only the
        # machine coordinates remain.
        before_count = re.split(r"count", resp, flags=re.IGNORECASE)[0]
        x = y = z = None
        m = re.search(r"X:\s*([-+]?\d*\.?\d+)", before_count)
        if m:
            try:
                x = float(m.group(1))
            except ValueError:
                pass
        m = re.search(r"Y:\s*([-+]?\d*\.?\d+)", before_count)
        if m:
            try:
                y = float(m.group(1))
            except ValueError:
                pass
        m = re.search(r"Z:\s*([-+]?\d*\.?\d+)", before_count)
        if m:
            try:
                z = float(m.group(1))
            except ValueError:
                pass
        return (x, y, z)

    def wait_for_moves(self, timeout_s=5.0):
        # M400 blocks until the planner is empty; keep it, but it should run off the UI thread.
        self.send("M400", wait_ok=True)
