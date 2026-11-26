# io_main with the joint current from the torque PDO (rhps1c ~/rl_deploy, 2026-11-26)

On RHPS1 the drives' 0x6078 (current actual) reads 0, so tauIn and NewRLQPController_joint_current_A were all zero on
the robot. 0x6077 (torque actual) is valid. In ~/rl_deploy/src/hrp5p-iob (copy of ~/workspace/src/hrp5p-iob, branch
RHPS1 aed9590), iob/TransListenerImpl.cpp, both joint loops:

    current += dev->GetCurrent();   ->   current += dev->GetTorque();

GetTorque() = sign * ratedCurrent * torqueConstant_ * raw / 1e3 with torqueConstant_ = the VRML torqueConst = 1, so the
value stays in amperes: same units downstream (NewRLQPController converts with joint_torque_scale), same shm layout,
the lab's libhrpIo.so reads it unchanged. Same fix as b99f308 on the previous control PC (never pushed).

Build: cmake -S ~/rl_deploy/src/hrp5p-iob -B ~/rl_deploy/build/hrp5p-iob -G Ninja -DCMAKE_BUILD_TYPE=Release
-DROBOT_NAME=RHPS1 -DRUN_AS_RT_TASK=ON -DUSE_IOB=ON -DUSE_VEL_CNT=ON -DBUILD_BODY_SIGNAL_LIB=ON -DPYTHON_BINDING=OFF
-DCMAKE_INSTALL_PREFIX=~/rl_deploy/iob_install (sake_gripper fails: no dynamixel SDK, not needed); io_main copied to
~/rl_deploy/iob_install/bin, ~/workspace/install/share/iob copied to ~/rl_deploy/iob_install/share/iob.
Run: sudo ~/rl_deploy/RHPS1init_rl.sh instead of RHPS1init (identical except the io_main line). Check: bodystat's
current column non-zero with servo ON.
