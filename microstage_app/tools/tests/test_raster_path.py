
import types
from microstage_app.control.raster import RasterRunner, RasterConfig

class FakeStage:
    def __init__(self):
        self.moves = []
    def wait_for_moves(self, timeout_s=5.0):
        self.moves.append(('wait',))
    def move_relative(self, dx=0, dy=0, dz=0, feed_mm_per_min=600, wait_ok=False):
        self.moves.append(('move', round(dx,3), round(dy,3), round(dz,3)))

class FakeCam:
    def __init__(self):
        self.n = 0
    def snap(self):
        self.n += 1
        return None

class FakeWriter:
    def __init__(self):
        self.tiles = []
    def save_tile(self, img, r, c):
        self.tiles.append((r,c))


def test_raster_serpentine_moves_and_tiles():
    stage, cam, writer = FakeStage(), FakeCam(), FakeWriter()
    cfg = RasterConfig(rows=2, cols=3, pitch_x_mm=1.0, pitch_y_mm=2.0, serpentine=True)
    RasterRunner(stage, cam, writer, cfg).run()
    # Expect tiles in order: row0 c0,c1,c2; row1 c2,c1,c0
    assert writer.tiles == [(0,0),(0,1),(0,2),(1,2),(1,1),(1,0)]
    # Moves: between tile columns and to next row; number of X moves per row = cols-1; plus Y move between rows
    x_moves = [m for m in stage.moves if m[0]=='move' and m[1] != 0]
    y_moves = [m for m in stage.moves if m[0]=='move' and m[2] == 0 and m[1]==0]
    assert len(x_moves) == (cfg.cols-1) + (cfg.cols-1)  # two rows
    assert len(y_moves) == 1


def test_raster_non_serpentine_returns_to_start_edge():
    stage, cam, writer = FakeStage(), FakeCam(), FakeWriter()
    cfg = RasterConfig(rows=2, cols=3, pitch_x_mm=1.5, pitch_y_mm=2.0, serpentine=False)
    RasterRunner(stage, cam, writer, cfg).run()
    # Tiles: always left-to-right
    assert writer.tiles == [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2)]
    # After finishing a row, X should return by -(cols-1)*pitch to start edge
    # Look for a return move of -3.0 after first row
    moves = [m for m in stage.moves if m[0]=='move']
    # Find the first Y move to separate rows
    y_index = next(i for i,m in enumerate(moves) if m[2]==0 and m[1]==0 and m[3]==0)
    # The move right before Y should be the return move
    ret = moves[y_index-1]
    assert ret[1] == - (cfg.cols-1) * cfg.pitch_x_mm
