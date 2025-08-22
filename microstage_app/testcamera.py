import toupcam, time, ctypes, numpy as np
devs = toupcam.Toupcam.EnumV2() or []
cam = toupcam.Toupcam.Open(devs[0].id)
got = []
CB = ctypes.CFUNCTYPE(None, ctypes.c_uint)
def cb(evt):
    if evt == toupcam.TOUPCAM_EVENT_IMAGE:
        w,h = cam.get_Size()
        try:
            data = cam.PullImageV3(w,h,24)
            arr = np.frombuffer(data, dtype=np.uint8).reshape(h,w,3)
        except Exception:
            buf = (ctypes.c_ubyte*(w*h*3))()
            cam.PullImageV3(buf,24,w,h)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h,w,3)
        got.append(arr.shape)
cam.StartPullModeWithCallback(CB(cb))
time.sleep(1.0); cam.Stop(); cam.Close()
print("frames:", got)