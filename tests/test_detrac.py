import cv2
import numpy as np

from vehicle_counter.datasets.ua_detrac import GREY, convert, parse_xml

XML = """<?xml version="1.0" encoding="utf-8"?>
<sequence name="MVI_00001">
  <sequence_attribute camera_state="unstable" sence_weather="sunny"/>
  <ignored_region><box left="0" top="0" width="50" height="40"/></ignored_region>
  <frame density="2" num="1">
    <target_list>
      <target id="1"><box left="100" top="200" width="80" height="60"/>
        <attribute speed="5" truncation_ratio="0" vehicle_type="car"/></target>
      <target id="2"><box left="900" top="500" width="100" height="80"/>
        <attribute speed="5" truncation_ratio="0" vehicle_type="bus"/></target>
    </target_list>
  </frame>
  <frame density="1" num="2">
    <target_list>
      <target id="1"><box left="100" top="210" width="80" height="60"/>
        <attribute speed="5" truncation_ratio="0" vehicle_type="car"/></target>
    </target_list>
  </frame>
</sequence>
"""


def make_dataset(root, names):
    for name in names:
        (root / "xml").mkdir(parents=True, exist_ok=True)
        (root / "xml" / f"{name}.xml").write_text(XML.replace("MVI_00001", name))
        (root / "img" / name).mkdir(parents=True)
        for num in (1, 2):
            cv2.imwrite(str(root / "img" / name / f"img{num:05d}.jpg"), np.full((540, 960, 3), 255, np.uint8))


def test_parse_xml(tmp_path):
    (tmp_path / "a.xml").write_text(XML)
    seq = parse_xml(tmp_path / "a.xml")
    assert seq.name == "MVI_00001" and seq.ignored == [(0, 0, 50, 40)]
    assert [t.vehicle_type for t in seq.frames[1]] == ["car", "bus"]


def test_convert(tmp_path):
    make_dataset(tmp_path, ["MVI_1", "MVI_2", "MVI_3"])
    counts = convert(tmp_path / "img", tmp_path / "xml", tmp_path / "out", val_fraction=0.34, frame_step=1)
    assert counts == {"train": 4, "val": 2}
    out = tmp_path / "out"
    label = next((out / "labels" / "train").glob("*_00001.txt")).read_text().splitlines()
    cls, cx, cy, w, h = label[1].split()  # the bus is clipped to the image border
    assert cls == "2" and abs(float(cx) + float(w) / 2 - 1.0) < 1e-6
    image = cv2.imread(str(next((out / "images" / "train").glob("*.jpg"))))
    assert tuple(image[10, 10]) == GREY  # ignored region greyed out
    val_seq = next((out / "sequences").iterdir())
    assert (val_seq / "video.mp4").exists() and len((val_seq / "gt.txt").read_text().splitlines()) == 3
    assert "names:" in (out / "data.yaml").read_text()
