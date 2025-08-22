import time

def run(stage, camera, writer, dz_mm=0.02, steps=10):
    half = (steps//2) * dz_mm
    stage.move_relative(dz=-half); stage.wait_for_moves()
    for i in range(steps):
        img = camera.snap()
        if img is not None: writer.save_single(img)
        stage.move_relative(dz=dz_mm); stage.wait_for_moves(); time.sleep(0.02)
    stage.move_relative(dz=-half); stage.wait_for_moves()
