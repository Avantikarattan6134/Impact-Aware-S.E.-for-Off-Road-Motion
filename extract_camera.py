"""
scripts/extract_camera.py
=========================
Extract camera frames from ROOAD .bag files -> NPZ archives.

Each output file: <seq>_camera.npz containing:
  timestamps : (N,)     float64 [s]
  frames     : (N,H,W)  uint8   grayscale, resized to 640x400

Usage
-----
  python scripts/extract_camera.py --seqs rt4_updown rt5_updown
"""
import sys, argparse, numpy as np
from pathlib import Path
sys.path.insert(0, '.')

SEQUENCES = {
    "rt4_gravel": "bags/rt4_gravel.bag",
    "rt4_rim":    "bags/rt4_rim.bag",
    "rt4_updown": "bags/rt4_updown.bag",
    "rt5_gravel": "bags/rt5_gravel.bag",
    "rt5_rim":    "bags/rt5_rim.bag",
    "rt5_updown": "bags/rt5_updown.bag",
}

CAM_TOPIC = "/pylon_camera_node/image_raw"
OUT_DIR   = Path("data/rooad/camera")
WORK_W, WORK_H = 640, 400

def extract(seq, bag_path):
    out = OUT_DIR / f"{seq}_camera.npz"
    if out.exists():
        d = np.load(out)
        print(f"  SKIP {seq} ({len(d['timestamps'])} frames already extracted)")
        return

    try:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_typestore
        typestore = get_typestore(Stores.ROS1_NOETIC)
        def deser(raw, mtype): return typestore.deserialize_ros1(raw, mtype)
    except ImportError:
        try:
            from rosbags.rosbag1 import Reader
            from rosbags.serde import deserialize_cdr
            def deser(raw, mtype): return deserialize_cdr(raw, mtype)
        except:
            print("  ERROR: rosbags not installed"); return

    import cv2
    bag = Path(bag_path)
    if not bag.exists():
        print(f"  WAIT  {seq} (bag not found: {bag})"); return

    print(f"  Extracting camera from {bag.name} ...")
    timestamps, frames = [], []
    count = 0

    with Reader(str(bag)) as reader:
        topics = {c.topic for c in reader.connections}
        topic  = CAM_TOPIC if CAM_TOPIC in topics else None
        if not topic:
            for alt in ["/camera/image_raw", "/image_raw"]:
                if alt in topics: topic = alt; break
        if not topic:
            print(f"  ERROR: No camera topic. Available: {sorted(topics)}"); return
        print(f"    Topic: {topic}")

        for conn, ts, raw in reader.messages():
            if conn.topic != topic: continue
            try:
                msg = deser(raw, conn.msgtype)
                t   = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

                # Decode image
                h, w = msg.height, msg.width
                enc  = msg.encoding.lower()
                data = np.frombuffer(msg.data, dtype=np.uint8)

                if 'mono' in enc or 'gray' in enc:
                    img = data.reshape(h, w)
                elif 'bayer' in enc:
                    bayer = data.reshape(h, w)
                    img = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2GRAY)
                else:
                    # bgr8 or rgb8
                    img = data.reshape(h, w, -1)
                    if 'rgb' in enc:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Resize to working resolution
                img_small = cv2.resize(img, (WORK_W, WORK_H))
                timestamps.append(t)
                frames.append(img_small)
                count += 1
                if count % 100 == 0:
                    print(f"    ... {count} frames")
            except Exception as e:
                continue

    if not timestamps:
        print(f"  ERROR: No frames extracted"); return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out,
                        timestamps=np.array(timestamps),
                        frames=np.array(frames, dtype=np.uint8))
    dur = timestamps[-1] - timestamps[0]
    hz  = len(timestamps) / dur
    print(f"  DONE  {len(timestamps)} frames  {dur:.1f}s  ~{hz:.1f} Hz  -> {out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seqs", nargs="+", default=list(SEQUENCES.keys()))
    args = parser.parse_args()
    for seq in args.seqs:
        if seq not in SEQUENCES:
            print(f"Unknown: {seq}"); continue
        extract(seq, SEQUENCES[seq])
