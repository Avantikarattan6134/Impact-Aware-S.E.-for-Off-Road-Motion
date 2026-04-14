"""
extract_imu.py - Compatible with rosbags >= 0.9.x and 0.11.x
"""
import sys, csv, numpy as np
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

IMU_TOPIC = "/vectornav/IMU"
OUT_DIR   = Path("data/rooad/imu")
OUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    from rosbags.rosbag1 import Reader
    try:
        from rosbags.typesys import Stores, get_typestore
        typestore = get_typestore(Stores.ROS1_NOETIC)
        def deserialize(rawdata, msgtype):
            return typestore.deserialize_ros1(rawdata, msgtype)
        print("rosbags API: new (>= 0.10)")
    except ImportError:
        from rosbags.serde import deserialize_cdr
        def deserialize(rawdata, msgtype):
            return deserialize_cdr(rawdata, msgtype)
        print("rosbags API: old (< 0.10)")
except ImportError:
    print("rosbags not installed. Run:  pip install rosbags")
    sys.exit(1)

for seq, bag_path in SEQUENCES.items():
    bag = Path(bag_path)
    out = OUT_DIR / f"{seq}_imu.csv"

    if out.exists():
        n = sum(1 for _ in open(out)) - 1
        print(f"  SKIP  {seq}  ({n} samples already extracted)")
        continue

    if not bag.exists():
        print(f"  WAIT  {seq}  (bag not downloaded yet)")
        continue

    print(f"\n  Extracting {seq} ... (5-15 min, please wait)")
    rows  = []
    count = 0

    with Reader(str(bag)) as reader:
        topics = {c.topic for c in reader.connections}
        topic  = IMU_TOPIC if IMU_TOPIC in topics else None
        if not topic:
            for alt in ["/imu/data", "/imu", "/imu_raw"]:
                if alt in topics:
                    topic = alt
                    break
        if not topic:
            print(f"    ERROR: No IMU topic. Available: {sorted(topics)}")
            continue
        print(f"    Topic: {topic}")

        for conn, ts, rawdata in reader.messages():
            if conn.topic != topic:
                continue
            try:
                msg = deserialize(rawdata, conn.msgtype)
                t   = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                la  = msg.linear_acceleration
                av  = msg.angular_velocity
                rows.append([t, la.x, la.y, la.z, av.x, av.y, av.z])
            except Exception:
                continue
            count += 1
            if count % 50000 == 0:
                print(f"    ... {count} messages read")

    if not rows:
        print(f"    ERROR: No IMU messages extracted")
        continue

    rows.sort(key=lambda r: r[0])
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['timestamp_s','ax','ay','az','gx','gy','gz'])
        w.writerows(rows)

    dur = rows[-1][0] - rows[0][0]
    hz  = len(rows) / dur if dur > 0 else 0
    print(f"    DONE  {len(rows)} samples  {dur:.1f}s  ~{hz:.0f} Hz  -> {out}")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)
for seq in SEQUENCES:
    out = OUT_DIR / f"{seq}_imu.csv"
    if out.exists():
        n = sum(1 for _ in open(out)) - 1
        print(f"  {seq:20s}  {n:7d} IMU samples  READY")
    else:
        bag = Path(SEQUENCES[seq])
        status = "bag not downloaded" if not bag.exists() else "not extracted yet"
        print(f"  {seq:20s}  {status}")
