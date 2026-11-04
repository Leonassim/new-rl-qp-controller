# Foot force sensors, 2026-11-04

Why: the robot falls to the right with a colleague's controller. Suspected the foot F/T sensors (ATI EtherCAT). Everything below is on the RHPS1 real robot.

## Summary

- Right foot sensor is accurate (+-1 %). Left foot sensor reads ~12 % high (scale error, not an offset).
- Workaround in the low level: per-sensor gain in the IOB (`force_gain.dat`, left foot force x0.89). Local patch, not upstream.
- With the gain applied, the robot really loads the right foot more in the default posture (60 / 40, 100 N gap). The gap shrinks with more knee flexion (52.7 / 47.3, then 51.9 / 48.1).
- Loaded readings are stable (< 0.4 % over 4 min). The zero of the left sensor drifts by ~+9 N over ~50 min in the air; the right one stays within +-3 N.
- Ankle `KI[2]` was put back to 2.0 on the 4 ankle drives and saved (it had been 0.02).

## 1. Scale check (balance under one foot, robot on one leg)

Expected = balance kg x 9.81. Sensor Fz / expected:

| Side | Points | Ratio |
|---|---|---|
| Right | 14.4 / 28 / 55 kg | 1.007 / 1.001 / 0.988 |
| Left | 15.9 / 17.5 / 22 / 34 kg | 1.115 / 1.130 / 1.094 / 1.130 |

Repeated later, same ~12 %. Fit on the left: slope ~1.14, intercept ~ -5 N (foot weight). Foot weight is a constant offset, it does not change the slope. Tilt of the leg can only lower Fz, not raise it.

Where the scale comes from: `force = raw / countPerForce`, with `countPerForce` read from each sensor by SDO `0x2040:49` (force) and `0x2040:50` (torque) at EtherCAT init (`EcManager.cpp`), default 1000 if the read fails. The IOB only adds a constant offset afterwards. The `[addr] CountPerForce = ...` lines on the `io_main` console were not read yet: if left != right (or left = 1000), that is the root cause.

## 2. IOB workaround

Local change in `hrp5p-iob` (branch `topic/EC-master3.2-rhps1`, not pushed upstream): `docs/hrp5p-iob_force_gain.patch`.

`force = raw * gain + offset`, gains read at startup from `<IOB_SHARE_DIR>/force_gain.dat`, one line per sensor id (0 RF, 1 LF, 2 RH, 3 LH), `<force gain> <torque gain>`. Missing file = 1.0.

```
1.0 1.0
0.89 1.0
1.0 1.0
1.0 1.0
```

Only the left foot force is scaled; the torque gain is unchecked. `io_main` prints `force sensor N gain: ...` at startup.

```
ninja -C build io_main
cp build/iob/io_main ~/sanbox_leo/workspace/install/bin/io_main   # backup: io_main.bak-20261104
```

Only `~/sanbox_leo/workspace/install/bin/io_main` was replaced (not `~/workspace/install/bin/io_main`). Other PCs need the patch, the build and `force_gain.dat`. Remove it if the sensor gets fixed at the source.

## 3. Load split (log, robot on the ground, gain applied)

| Posture (cmd) | Fz right | Fz left | Sum | Right |
|---|---|---|---|---|
| knee 35.6, chest 0 (t 2-18 s) | 306.9 N | 206.9 N | 513.8 N | 59.7 % |
| knee 49, ankle P -21.1, chest 39 (t 260-500 s) | 275.5 N | 247.7 N | 523.3 N | 52.7 % |
| knee 71.4, ankle P -36.7, chest 21 (t 825-860 s) | 275.2 N | 255.5 N | 530.7 N | 51.9 % |

The robot was lifted between postures (sum ~ 0). Joint tracking error: <= 0.0005 deg on hips and ankles, 0.015 deg on knees in the last posture. Servos off or on gives the same split.

Reading: the imbalance is real (not a sensor error) and decreases with knee flexion, consistent with a hyperstatic contact (small height / offset mismatch between feet) more than a fixed lateral CoM offset. One run per posture, not conclusive.

Knee current (`tauIn`, amperes, see note below) right / left: 1.9, 1.24, 1.07 for the three postures. Load ratio: 1.48, 1.11, 1.08. The knee current asymmetry follows the load asymmetry.

Ankle pitch: right foot My = 17.2 N.m vs 1.1 N.m left in posture 2 (centre of pressure ~6 cm from the ankle on the right, ~0.4 cm on the left). Not explained.

## 4. Drift

- On the ground, 4 min still (posture 2): right 276.1 -> 275.1 N, left 248.0 -> 247.5 N.
- Zero in the air (robot lifted for ~45 min after t = 880 s):

| t (s) | RF Fz | LF Fz |
|---|---|---|
| 100-120 | -2.1 | -1.2 |
| 940-1000 | -2.7 | +3.8 |
| 2040-2160 | 0.0 | +6.5 |
| 3000-3120 | +0.1 | +7.5 |
| 3480-3600 | -1.1 | +7.2 |

Left zero moves ~+9 N (~3.5 % of the load), fast in the first 15 min then flat. Re-zero in the air before each session, once warmed up.

## 5. Ankle gains

`KI[2]` of the 4 ankle drives (`LAnkleRPI`, `LAnkleRPO`, `RAnkleRPI`, `RAnkleRPO`) was set to 0.02, tested, then put back to 2.0 and stored (`0x1010:1` = `0x65766173`). `KP[3]=60`, `GS[2]=0` unchanged. Read back after the store:

```
sdo <drive> float ki 2          # read
sdo <drive> float ki 2 2.0      # write
sdo <drive> uint 0x1010 1 1702257011   # store
```

Incident: a verification loop appended `0` to the read commands and wrote `KP[3]=0`, `KI[2]=0` on `LAnkleRPI` for a few seconds (servos off, no controller). Restored to 60 / 0.02, stored again on the 4 drives. Any extra argument turns a read into a write.

## Notes

- `bodystat` prints `torque(%)` but `state.torque` is a motor current in amperes (`gearRatio` and `torqueConst` are 1 in the VRML). The local `bodystat.cpp` comment says so.
- `bodystat` smooths forces from zero at each start: wait ~2 s before reading.
- Sensor order in `bodystat` / `state.force`: RF, LF, RH, LH (`sensorId` 0..3 in `RHPS1.conf`).

## Files (`logs/tmp/`)

- `rhps1_halfsit_2026-11-04_keys_t0-1010.bin.xz`: HalfSitPose log 19:42, t = 0..1010 s, keys only (`t`, both foot force sensors, `qIn`, `qOut`, `tauIn`, accelerometer orientation). The full log is 4.3 GB / 3615 s and was not kept; the last 45 min are the robot idle in the air.
- `rhps1_halfsit_2026-11-04_summary_1Hz.csv.gz`: whole session at 1 Hz (forces, knee / ankle / chest commands, currents), including the drift record.
- `rhps1_halfsit_2026-11-04_hrpsys.log.gz`: `io_main` / rtcd output of the session.
