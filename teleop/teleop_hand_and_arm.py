import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os 
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber  # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_, WirelessController_
from teleop.utils.base_velocity import (
    read_sport_snapshot,
    read_wireless_snapshot,
    fsm_mode_to_enum,
    r3_stick_to_cmd_vel,
)
from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm import G1_29_ArmController, G1_23_ArmController, H1_2_ArmController, H1_ArmController
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK, G1_23_ArmIK, H1_2_ArmIK, H1_ArmIK
from teleimager.image_client import ImageClient
from teleop.utils.episode_writer import EpisodeWriter as JsonEpisodeWriter
from teleop.utils.ipc import IPC_Server
from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper
from sshkeyboard import listen_keyboard, stop_listening

# for simulation
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
def publish_reset_category(category: int, publisher): # Scene Reset signal
    msg = String_(data=str(category))
    publisher.Write(msg)
    logger_mp.info(f"published reset category: {category}")

# state transition
START          = False  # Enable to start robot following VR user motion
STOP           = False  # Enable to begin system exit procedure
READY          = False  # Ready to (1) enter START state, (2) enter RECORD_RUNNING state
RECORD_RUNNING = False  # True if [Recording]
RECORD_TOGGLE  = False  # Toggle recording state
PAUSED              = False  # True while [p] paused — robot holds last commanded pose
RAMP_DURATION_SEC   = 1.0    # Wall-clock target for the resume ramp
RAMP_TICKS          = 30     # Recomputed from args.frequency below; placeholder = 30 ticks @ 30 Hz
#  -------        ---------                -----------                -----------            ---------
#   state          [Ready]      ==>        [Recording]     ==>         [AutoSave]     -->     [Ready]
#  -------        ---------      |         -----------      |         -----------      |     ---------
#   START           True         |manual      True          |manual      True          |        True
#   READY           True         |set         False         |set         False         |auto    True
#   RECORD_RUNNING  False        |to          True          |to          False         |        False
#                                ∨                          ∨                          ∨
#   RECORD_TOGGLE   False       True          False        True          False                  False
#  -------        ---------                -----------                 -----------            ---------
#  ==> manual: when READY is True, set RECORD_TOGGLE=True to transition.
#  --> auto  : Auto-transition after saving data.
#
#  [p] Pause branch (only from [Ready], not from [Recording]):
#     [Ready]  --p-->  [Paused]  --p-->  [Resuming (ramp, RAMP_TICKS ticks)]  -->  [Ready]
#  During [Paused]: arm re-sends frozen target, hand writes skipped, [s] rejected.
#  During [Resuming]: alpha sweeps 1/RAMP_TICKS → 1.0, blending frozen → fresh.

def on_press(key):
    global STOP, START, RECORD_TOGGLE, PAUSED
    if key == 'r':
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START and not PAUSED:
        RECORD_TOGGLE = True
    elif key == 'p' and START and not RECORD_RUNNING and not RECORD_TOGGLE:
        # Also gate on RECORD_TOGGLE: closes the sub-tick race where `s` flips
        # RECORD_TOGGLE=True but the main loop has not yet promoted it to
        # RECORD_RUNNING=True — otherwise a `p` pressed between those two
        # events would sneak paused frames into the new episode.
        PAUSED = not PAUSED
        # Synchronous feedback so a fast double-press (where both flips happen
        # between two main-loop ticks → no edge detected → no ⏸/▶ log) still
        # tells the operator that each press registered.
        logger_mp.info(f"[on_press] PAUSED → {PAUSED}")
    elif key in ('s', 'p'):
        # Key is known but its guard rejected it (not tracking, wrong state, etc.)
        logger_mp.warning(f"[on_press] {key} rejected — guard not satisfied for current state.")
    else:
        # Key is not bound to any action in this FSM.
        logger_mp.warning(f"[on_press] {key} — no action defined for this key.")

def get_state() -> dict:
    """Return current heartbeat state"""
    global START, STOP, RECORD_RUNNING, READY, PAUSED
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
        "PAUSED": PAUSED,
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # basic control parameters
    parser.add_argument('--frequency', type = float, default = 30.0, help = 'control and record \'s frequency')
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand', help='Select XR device input tracking source')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive', help='Select XR device display mode')
    parser.add_argument('--arm', type=str, choices=['G1_29', 'G1_23', 'H1_2', 'H1'], default='G1_29', help='Select arm controller')
    parser.add_argument('--ee', type=str, choices=['dex1', 'dex3', 'inspire_ftp', 'inspire_dfx', 'brainco'], help='Select end effector controller')
    parser.add_argument('--img-server-ip', type=str, default='192.168.123.164', help='IP address of image server, used by teleimager and televuer')
    parser.add_argument('--network-interface', type=str, default=None, help='Network interface for dds communication, e.g., eth0, wlan0. If None, use default interface.')
    # mode flags
    g_lower = parser.add_mutually_exclusive_group()
    g_lower.add_argument('--motion', action='store_true',
        help='R1+X gamepad path (legacy). Operator must put G1 in Regular mode first.')
    g_lower.add_argument('--homie', action='store_true',
        help='HOMIE-managed lower body. run_homie.sh must be running on PC2 before this script.')

    parser.add_argument('--homie-host', default='127.0.0.1')
    parser.add_argument('--homie-port', type=int, default=7701)
    parser.add_argument('--homie-height', type=float, default=None,
        help="If set, request HOMIE to stand at this height (meters) via "
             "set_velocity height field after calibration. Default: HOMIE's "
             "stock nominal height (0.74m). Useful for table clearance.")
    parser.add_argument('--headless', action='store_true', help='Enable headless mode (no display)')
    parser.add_argument('--sim', action = 'store_true', help = 'Enable isaac simulation mode')
    parser.add_argument('--ipc', action = 'store_true', help = 'Enable IPC server to handle input; otherwise enable sshkeyboard')
    parser.add_argument('--affinity', action = 'store_true', help = 'Enable high priority and set CPU affinity mode')
    # record mode and task info
    parser.add_argument('--record', action = 'store_true', help = 'Enable data recording mode')
    parser.add_argument('--writer', type = str, default = 'lerobot', choices = ['json', 'lerobot'],
                        help = "Episode writer backend. 'lerobot' (default) emits LeRobot v3 "
                               "staging artifacts (parquet + MP4 + meta.json) directly; run "
                               "unitree_lerobot.utils.finalize_lerobot_dataset after the session "
                               "to consolidate into the v3 chunked layout. Schema is locked on "
                               "Unitree_G1_Inspire_HeadOnly_Mono_BaseVel_v1 (30D state / 29D action, 1-cam head mono 640x480 "
                               "@ 30fps). 'json' is the legacy path — data.json + per-frame "
                               "image files for offline conversion; kept for rollback during "
                               "the post-flip window.")
    parser.add_argument('--task-dir', type = str, default = './utils/data/', help = 'path to save data')
    parser.add_argument('--task-name', type = str, default = 'pick cube', help = 'task file name for recording')
    parser.add_argument('--task-goal', type = str, default = 'pick up cube.', help = 'task goal for recording at json file')
    parser.add_argument('--task-desc', type = str, default = 'task description', help = 'task description for recording at json file')
    parser.add_argument('--task-steps', type = str, default = 'step1: do this; step2: do that;', help = 'task steps for recording at json file')

    args = parser.parse_args()
    logger_mp.info(f"args: {args}")

    # Recompute RAMP_TICKS so the resume blend is RAMP_DURATION_SEC of wall-clock
    # at whatever --frequency the operator picked. Without this, RAMP_TICKS=30
    # silently means 0.5 s at 60 Hz or 2.0 s at 15 Hz.
    RAMP_TICKS = max(1, int(round(args.frequency * RAMP_DURATION_SEC)))

    # Pre-declare HOMIE-related state at __main__ scope so the finally block can
    # reference them even if the try: body bails out before the lower-body branch.
    gate = None
    homie_abort = threading.Event()

    try:
        # setup dds communication domains id
        if args.sim:
            ChannelFactoryInitialize(1, networkInterface=args.network_interface)
        else:
            ChannelFactoryInitialize(0, networkInterface=args.network_interface)

        # Latest-sample DDS subscribers for base-velocity recording (spec §6).
        # Atomic single-element-list assignment is GIL-atomic in CPython, so the
        # record loop gets a consistent tuple snapshot without a lock.
        # Queue depth 1: we only ever read the latest; queued backlog is waste.
        latest_sport = [None]      # (mode:int, vx, vy, vz, yaw_speed)
        latest_wireless = [None]   # (lx, ly, rx)

        def _sport_cb(msg):
            # CycloneDDS IDL uint8/float32 fields deliver native Python int/float;
            # no coercion needed.
            latest_sport[0] = (
                msg.mode,
                msg.velocity[0],
                msg.velocity[1],
                msg.velocity[2],
                msg.yaw_speed,
            )

        def _wireless_cb(msg):
            latest_wireless[0] = (msg.lx, msg.ly, msg.rx)

        sport_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        sport_sub.Init(_sport_cb, 1)
        wireless_sub = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
        wireless_sub.Init(_wireless_cb, 1)

        # ipc communication mode. client usage: see utils/ipc.py
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press,get_state=get_state)
            ipc_server.start()
        # sshkeyboard communication mode
        else:
            listen_keyboard_thread = threading.Thread(target=listen_keyboard, 
                                                      kwargs={"on_press": on_press, "until": None, "sequential": False,}, 
                                                      daemon=True)
            listen_keyboard_thread.start()

        # image client
        img_client = ImageClient(host=args.img_server_ip, request_bgr=True)
        camera_config = img_client.get_cam_config()
        logger_mp.debug(f"Camera config: {camera_config}")
        xr_need_local_img = not (args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

        # televuer_wrapper: obtain hand pose data from the XR device and transmit the robot's head camera image to the XR device.
        tv_wrapper = TeleVuerWrapper(use_hand_tracking=args.input_mode == "hand", 
                                     binocular=camera_config['head_camera']['binocular'],
                                     img_shape=camera_config['head_camera']['image_shape'],
                                     # maybe should decrease fps for better performance?
                                     # https://github.com/unitreerobotics/xr_teleoperate/issues/172
                                     # display_fps=camera_config['head_camera']['fps'] ? args.frequency? 30.0?
                                     display_mode=args.display_mode,
                                     zmq=camera_config['head_camera']['enable_zmq'],
                                     webrtc=camera_config['head_camera']['enable_webrtc'],
                                     webrtc_url=f"https://{args.img_server_ip}:{camera_config['head_camera']['webrtc_port']}/offer",
                                     )
        
        # Lower-body controller selection: HOMIE / R1+X gamepad / debug mode
        # (gate and homie_abort are pre-declared at __main__ scope above.)
        if args.homie:
            from teleop.utils.homie_client import HomieGate, HomieGateError
            try:
                gate = HomieGate(args.homie_host, args.homie_port,
                                 connect_deadline_s=30.0, connect_interval_s=1.0,
                                 socket_timeout_s=0.5)
                gate.connect_with_retry()
                state = gate.probe_state()
                gate.calibrate_if_needed(state)
                if args.homie_height is not None:
                    logger_mp.info(f"Setting HOMIE standing height to "
                                    f"{args.homie_height:.3f}m via set_velocity")
                    gate.set_height(args.homie_height)
                gate.start_watchdog(
                    period_s=1.0, fail_threshold=3,
                    on_abort=lambda: (
                        logger_mp.error("🛑 HOMIE WATCHDOG TRIPPED — stopping teleop"),
                        homie_abort.set()))
                logger_mp.info(f"HOMIE ready (calibrated={state.calibrated}, "
                                f"controller_running={state.controller_running})")
            except HomieGateError as e:
                logger_mp.error(f"HOMIE pre-flight failed: {e}")
                if gate is not None:
                    gate.close()
                sys.exit(4)
            # NOTE: skip Enter_Debug_Mode (HOMIE owns lower body)
            # and LocoClientWrapper (no Unitree AI/loco controller running).
        elif args.motion:
            if args.input_mode == "controller":
                loco_wrapper = LocoClientWrapper()
        else:
            motion_switcher = MotionSwitcher()
            status, result = motion_switcher.Enter_Debug_Mode()
            logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

        # arm
        # Under --motion (R1+X gamepad) we publish to DDS rt/arm_sdk with
        # weight=1 (motion_mode=True) so our writes win over the AI controller.
        # Under --homie we instead route through LCM arm_action — g1_control
        # merges leg torques (from HOMIE) and arm positions (from us) into a
        # single rt/lowcmd output. Writing rt/arm_sdk under HOMIE-release-mode
        # has NO arbiter (no AI controller running), so it grinds against
        # HOMIE's lowcmd writes at the lowcmd level. This mirrors the
        # IL_PhysicalAI eval-side fix (commit 1fda045b on
        # feat/pi05-pickhold-v03).
        _use_arm_sdk = args.motion  # --homie no longer touches rt/arm_sdk
        homie_lcm_handle = None     # kept alive for the lifetime of the process
        if args.homie:
            import lcm as _lcm
            from physical_ai_expo.motion.capture_playback import LCM_URL
            from teleop.utils.homie_io import HomieArmController
            homie_lcm_handle = _lcm.LCM(LCM_URL)
            if args.arm == "G1_29":
                arm_ik = G1_29_ArmIK()
                arm_ctrl = HomieArmController(lc=homie_lcm_handle, lerp_sec=0.15)
                logger_mp.info(
                    "HOMIE release mode: arm_ctrl rerouted via LCM arm_action "
                    "(was DDS rt/arm_sdk via G1_29_ArmController)")
            else:
                # HomieArmController is G1_29-only (matches eval-side path).
                # Other arms under --homie are unsupported here; fail loudly
                # rather than silently grinding via the legacy DDS path.
                raise NotImplementedError(
                    f"--homie currently supports --arm=G1_29 only; got {args.arm!r}. "
                    "Add a HomieArmController variant before enabling other arms."
                )
        else:
            if args.arm == "G1_29":
                arm_ik = G1_29_ArmIK()
                arm_ctrl = G1_29_ArmController(motion_mode=_use_arm_sdk, simulation_mode=args.sim)
            elif args.arm == "G1_23":
                arm_ik = G1_23_ArmIK()
                arm_ctrl = G1_23_ArmController(motion_mode=_use_arm_sdk, simulation_mode=args.sim)
            elif args.arm == "H1_2":
                arm_ik = H1_2_ArmIK()
                arm_ctrl = H1_2_ArmController(motion_mode=_use_arm_sdk, simulation_mode=args.sim)
            elif args.arm == "H1":
                arm_ik = H1_ArmIK()
                arm_ctrl = H1_ArmController(simulation_mode=args.sim)

        # end-effector
        if args.ee == "dex3":
            from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock = False)   # [output] current left, right hand state(14) data.
            dual_hand_action_array = Array('d', 14, lock = False)  # [output] current left, right hand action(14) data.
            hand_ctrl = Dex3_1_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                          dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "dex1":
            from teleop.robot_control.robot_hand_unitree import Dex1_1_Gripper_Controller
            left_gripper_value = Value('d', 0.0, lock=True)        # [input]
            right_gripper_value = Value('d', 0.0, lock=True)       # [input]
            dual_gripper_data_lock = Lock()
            dual_gripper_state_array = Array('d', 2, lock=False)   # current left, right gripper state(2) data.
            dual_gripper_action_array = Array('d', 2, lock=False)  # current left, right gripper action(2) data.
            gripper_ctrl = Dex1_1_Gripper_Controller(left_gripper_value, right_gripper_value, dual_gripper_data_lock, 
                                                     dual_gripper_state_array, dual_gripper_action_array, simulation_mode=args.sim)
        elif args.ee == "inspire_dfx" or args.ee == "inspire_ftp":
            if args.input_mode == "controller":
                # 1DOF mode: controller trigger → single grip value
                left_gripper_value = Value('d', 10.0, lock=True)         # [input] init OPEN (safety)
                right_gripper_value = Value('d', 10.0, lock=True)        # [input] init OPEN (safety)
                dual_gripper_data_lock = Lock()
                dual_gripper_state_array = Array('d', 2, lock=False)     # [output] [left_state, right_state]
                dual_gripper_action_array = Array('d', 2, lock=False)    # [output] [left_action, right_action]
                if args.homie:
                    # Under HOMIE release mode, route hand commands via DDS
                    # rt/inspire_hand/ctrl/{l,r}; Headless_driver_double
                    # subscribes and owns Modbus. Direct Modbus from
                    # Inspire_1DOF_Controller would collide with that driver
                    # (two RS485 masters → grinding). Mirrors eval-side fix.
                    from teleop.utils.homie_io import HomieHandController
                    gripper_ctrl = HomieHandController(
                        left_gripper_value, right_gripper_value, dual_gripper_data_lock,
                        dual_gripper_state_array, dual_gripper_action_array,
                        simulation_mode=args.sim, fps=30, mode="1dof")
                    logger_mp.info(
                        "HOMIE release mode: gripper_ctrl rerouted via DDS "
                        "rt/inspire_hand/ctrl (was direct Modbus via "
                        "Inspire_1DOF_Controller)")
                else:
                    from teleop.robot_control.robot_hand_inspire import Inspire_1DOF_Controller
                    gripper_ctrl = Inspire_1DOF_Controller(
                        left_gripper_value, right_gripper_value, dual_gripper_data_lock,
                        dual_gripper_state_array, dual_gripper_action_array,
                        protocol="dfx" if args.ee == "inspire_dfx" else "ftp",
                        simulation_mode=args.sim)
            else:
                # Full 6DOF hand tracking mode (existing code)
                if args.homie:
                    # 75D keypoint → 6D motor mapping happens inside
                    # Inspire_Controller_{DFX,FTP}'s control loop and is
                    # tightly coupled to direct-Modbus output. HomieHandController
                    # accepts only 6D motor arrays (or 1D scalar) on its inputs,
                    # so the keypoint adapter is out of scope for this commit.
                    # Fail loudly rather than silently grinding via the legacy
                    # direct-Modbus path under HOMIE.
                    raise NotImplementedError(
                        "--homie + --input-mode=hand + Inspire EE not supported yet. "
                        "Use --input-mode=controller (1DOF grip) or implement a HOMIE "
                        "keypoint→6D motor adapter that publishes via DDS "
                        "rt/inspire_hand/ctrl/{l,r}."
                    )
                if args.ee == "inspire_dfx":
                    from teleop.robot_control.robot_hand_inspire import Inspire_Controller_DFX
                    left_hand_pos_array = Array('d', 75, lock = True)
                    right_hand_pos_array = Array('d', 75, lock = True)
                    dual_hand_data_lock = Lock()
                    dual_hand_state_array = Array('d', 12, lock = False)
                    dual_hand_action_array = Array('d', 12, lock = False)
                    hand_ctrl = Inspire_Controller_DFX(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
                else:  # inspire_ftp
                    from teleop.robot_control.robot_hand_inspire import Inspire_Controller_FTP
                    left_hand_pos_array = Array('d', 75, lock = True)
                    right_hand_pos_array = Array('d', 75, lock = True)
                    dual_hand_data_lock = Lock()
                    dual_hand_state_array = Array('d', 12, lock = False)
                    dual_hand_action_array = Array('d', 12, lock = False)
                    hand_ctrl = Inspire_Controller_FTP(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        elif args.ee == "brainco":
            from teleop.robot_control.robot_hand_brainco import Brainco_Controller
            left_hand_pos_array = Array('d', 75, lock = True)      # [input]
            right_hand_pos_array = Array('d', 75, lock = True)     # [input]
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 12, lock = False)   # [output] current left, right hand state(12) data.
            dual_hand_action_array = Array('d', 12, lock = False)  # [output] current left, right hand action(12) data.
            hand_ctrl = Brainco_Controller(left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock, 
                                           dual_hand_state_array, dual_hand_action_array, simulation_mode=args.sim)
        else:
            pass
        
        # affinity mode (if you dont know what it is, then you probably don't need it)
        if args.affinity:
            import psutil
            p = psutil.Process(os.getpid())
            p.cpu_affinity([0,1,2,3]) # Set CPU affinity to cores 0-3
            try:
                p.nice(-20)           # Set highest priority
                logger_mp.info("Set high priority successfully.")
            except psutil.AccessDenied:
                logger_mp.warning("Failed to set high priority. Please run as root.")
                
            for child in p.children(recursive=True):
                try:
                    logger_mp.info(f"Child process {child.pid} name: {child.name()}")
                    child.cpu_affinity([5,6])
                    child.nice(-20)
                except psutil.AccessDenied:
                    pass

        # simulation mode
        if args.sim:
            reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
            reset_pose_publisher.Init()
            from teleop.utils.sim_state_topic import start_sim_state_subscribe
            sim_state_subscriber = start_sim_state_subscribe()

        # record + headless / non-headless mode
        if args.record:
            _writer_task_dir = os.path.join(args.task_dir, args.task_name)
            _writer_kwargs = dict(
                task_dir = _writer_task_dir,
                task_goal = args.task_goal,
                task_desc = args.task_desc,
                task_steps = args.task_steps,
                frequency = args.frequency,
                rerun_log = not args.headless,
            )
            if args.writer == 'lerobot':
                # Startup-time schema guard — the writer is locked on
                # Unitree_G1_Inspire_HeadOnly_Mono (640x480 mono head camera,
                # G1_29 arm, 6-DOF Inspire end-effector). Catching
                # incompatibilities here prevents the mid-episode BEL abort
                # that would otherwise burn a VR session.
                _hc = camera_config.get('head_camera', {}) if isinstance(camera_config, dict) else {}
                if _hc.get('binocular', False):
                    raise SystemExit(
                        "--writer=lerobot requires a mono head camera; current "
                        "head camera is binocular. Pass --no-binocular, or use "
                        "--writer=json."
                    )
                _shape = tuple(_hc.get('image_shape', ()) or ())
                if _shape and _shape != (640, 480):
                    raise SystemExit(
                        f"--writer=lerobot expects head camera image_shape "
                        f"(640, 480); got {_shape}. Reconfigure the head camera "
                        f"or use --writer=json."
                    )
                if args.arm != 'G1_29':
                    raise SystemExit(
                        f"--writer=lerobot is locked on arm=G1_29; got "
                        f"arm={args.arm}. Use --writer=json for other arms."
                    )
                if args.ee not in ('inspire_dfx', 'inspire_ftp'):
                    raise SystemExit(
                        f"--writer=lerobot is locked on ee=inspire_dfx|inspire_ftp "
                        f"(6-DOF Inspire); got ee={args.ee}. Use --writer=json "
                        f"for other end effectors."
                    )
                # Lazy import so the legacy json path doesn't require
                # pyarrow / the full LeRobot test dependency chain.
                from teleop.utils.lerobot_episode_writer import LeRobotEpisodeWriter
                logger_mp.info(
                    "📦 Writer: LeRobotEpisodeWriter — direct-to-LeRobot v3 "
                    f"staging at {_writer_task_dir}/_staging/. Run "
                    "`python -m unitree_lerobot.utils.finalize_lerobot_dataset "
                    "--task-dir <path> --repo-id <id>` after this session to "
                    "consolidate."
                )
                recorder = LeRobotEpisodeWriter(**_writer_kwargs)
            else:
                logger_mp.info(
                    "📝 Writer: JsonEpisodeWriter (legacy) — data.json + "
                    f"per-frame images at {_writer_task_dir}/. Pass "
                    "`--writer lerobot` to skip the offline conversion step."
                )
                recorder = JsonEpisodeWriter(**_writer_kwargs)

        logger_mp.info("----------------------------------------------------------------")
        logger_mp.info("🟢  Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("🟡  Press [s] to START or SAVE recording (toggle cycle).")
        else:
            logger_mp.info("🔵  Recording is DISABLED (run with --record to enable).")
        logger_mp.info("🔴  Press [q] to stop and exit the program.")
        logger_mp.info("⚠️  IMPORTANT: Please keep your distance and stay safe.")
        READY = True                  # now ready to (1) enter START state
        while not START and not STOP: # wait for start or stop signal.
            time.sleep(0.033)
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = img_client.get_head_frame()
                tv_wrapper.render_to_xr(head_img)

        logger_mp.info("---------------------🚀start Tracking🚀-------------------------")
        arm_ctrl.speed_gradual_max()
        # Pause/ramp tick-local state (initialized once; mutated each tick below)
        prev_paused             = False
        ramp_remaining          = 0
        frozen_sol_q            = None   # arm q captured at pause entry (blend-from)
        frozen_sol_tauff        = None
        last_commanded_q        = None   # arm q commanded at end of the previous tick
        last_commanded_tauff    = None
        frozen_hand_snapshot    = {}     # keyed 'left'/'right' — value type matches EE branch
        last_commanded_hand     = {}     # keyed 'left'/'right' — last value we wrote to shared mem
        # main loop. robot start to follow VR user's motion
        while not STOP:
            if homie_abort.is_set():
                logger_mp.error("HOMIE-abort flagged; exiting teleop main loop")
                break
            start_time = time.time()

            # --- Pause/ramp top-of-tick block -----------------------------------------
            # Stable per-tick snapshot of the async-mutated pause flag — use the local
            # for the rest of the tick so all downstream branches see one consistent
            # value even if the keyboard thread flips PAUSED mid-tick.
            # (RECORD_RUNNING is deliberately NOT snapshotted
            # here: the downstream recording append at line ~450 needs the freshest
            # value, which may change during this tick when RECORD_TOGGLE is handled.)
            current_tick_paused = PAUSED

            entering_pause = current_tick_paused and not prev_paused
            leaving_pause  = (not current_tick_paused) and prev_paused

            if entering_pause:
                logger_mp.info("⏸ PAUSED — robot holding last pose. Press [p] to resume.")
            if leaving_pause:
                ramp_remaining = RAMP_TICKS
                logger_mp.info(f"▶ Resumed — ramping to current VR pose over {RAMP_TICKS/args.frequency:.1f}s.")

            prev_paused = current_tick_paused

            # Blend factor: 0.0 = fully frozen target, 1.0 = fully tracking fresh pose.
            if current_tick_paused:
                alpha = 0.0
            elif ramp_remaining > 0:
                ramp_remaining -= 1
                alpha = (RAMP_TICKS - ramp_remaining) / RAMP_TICKS   # 1/30, 2/30, ..., 30/30
            else:
                alpha = 1.0
            # --- end pause/ramp top-of-tick block ------------------------------------

            # get image
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img = img_client.get_head_frame()
                if xr_need_local_img:
                    tv_wrapper.render_to_xr(head_img)
            if camera_config['left_wrist_camera']['enable_zmq']:
                if args.record:
                    left_wrist_img = img_client.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq']:
                if args.record:
                    right_wrist_img = img_client.get_right_wrist_frame()

            # record mode
            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()
                    if args.sim:
                        publish_reset_category(1, reset_pose_publisher)

            # get xr's tele data
            tele_data = tv_wrapper.get_tele_data()
            # Pause/ramp dispatch across EEs — the five non-else branches below share
            # an identical structure (snapshot → skip-if-paused → blend → write →
            # update last_commanded), differing only in data type (array vs scalar)
            # and which shared-memory target they write to. Scheduled for extraction
            # into a single helper per genonai/IL_PhysicalAI#30.
            if (args.ee == "dex3" or args.ee == "brainco") and args.input_mode == "hand":
                new_left  = tele_data.left_hand_pos.flatten()
                new_right = tele_data.right_hand_pos.flatten()
                if entering_pause:
                    frozen_hand_snapshot['left']  = last_commanded_hand.get('left',  new_left).copy()
                    frozen_hand_snapshot['right'] = last_commanded_hand.get('right', new_right).copy()
                if current_tick_paused:
                    pass   # Do not overwrite — hand controller reads last values
                else:
                    if alpha < 1.0:
                        f_left  = frozen_hand_snapshot['left']
                        f_right = frozen_hand_snapshot['right']
                        out_left  = f_left  + alpha * (new_left  - f_left)
                        out_right = f_right + alpha * (new_right - f_right)
                    else:
                        out_left, out_right = new_left, new_right
                    with left_hand_pos_array.get_lock():
                        left_hand_pos_array[:] = out_left
                    with right_hand_pos_array.get_lock():
                        right_hand_pos_array[:] = out_right
                    last_commanded_hand['left']  = out_left.copy()
                    last_commanded_hand['right'] = out_right.copy()
            elif args.ee == "dex1" and args.input_mode == "controller":
                new_left  = tele_data.left_ctrl_triggerValue
                new_right = tele_data.right_ctrl_triggerValue
                if entering_pause:
                    frozen_hand_snapshot['left']  = last_commanded_hand.get('left',  new_left)
                    frozen_hand_snapshot['right'] = last_commanded_hand.get('right', new_right)
                if current_tick_paused:
                    pass   # Do not overwrite — hand controller reads last values
                else:
                    if alpha < 1.0:
                        f_left  = frozen_hand_snapshot['left']
                        f_right = frozen_hand_snapshot['right']
                        out_left  = f_left  + alpha * (new_left  - f_left)
                        out_right = f_right + alpha * (new_right - f_right)
                    else:
                        out_left, out_right = new_left, new_right
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = out_left
                    with right_gripper_value.get_lock():
                        right_gripper_value.value = out_right
                    last_commanded_hand['left']  = out_left
                    last_commanded_hand['right'] = out_right
            elif args.ee == "dex1" and args.input_mode == "hand":
                new_left  = tele_data.left_hand_pinchValue
                new_right = tele_data.right_hand_pinchValue
                if entering_pause:
                    frozen_hand_snapshot['left']  = last_commanded_hand.get('left',  new_left)
                    frozen_hand_snapshot['right'] = last_commanded_hand.get('right', new_right)
                if current_tick_paused:
                    pass   # Do not overwrite — hand controller reads last values
                else:
                    if alpha < 1.0:
                        f_left  = frozen_hand_snapshot['left']
                        f_right = frozen_hand_snapshot['right']
                        out_left  = f_left  + alpha * (new_left  - f_left)
                        out_right = f_right + alpha * (new_right - f_right)
                    else:
                        out_left, out_right = new_left, new_right
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = out_left
                    with right_gripper_value.get_lock():
                        right_gripper_value.value = out_right
                    last_commanded_hand['left']  = out_left
                    last_commanded_hand['right'] = out_right
            elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp") and args.input_mode == "hand":
                new_left  = tele_data.left_hand_pos.flatten()
                new_right = tele_data.right_hand_pos.flatten()
                if entering_pause:
                    frozen_hand_snapshot['left']  = last_commanded_hand.get('left',  new_left).copy()
                    frozen_hand_snapshot['right'] = last_commanded_hand.get('right', new_right).copy()
                if current_tick_paused:
                    pass   # Do not overwrite — hand controller reads last values
                else:
                    if alpha < 1.0:
                        f_left  = frozen_hand_snapshot['left']
                        f_right = frozen_hand_snapshot['right']
                        out_left  = f_left  + alpha * (new_left  - f_left)
                        out_right = f_right + alpha * (new_right - f_right)
                    else:
                        out_left, out_right = new_left, new_right
                    with left_hand_pos_array.get_lock():
                        left_hand_pos_array[:] = out_left
                    with right_hand_pos_array.get_lock():
                        right_hand_pos_array[:] = out_right
                    last_commanded_hand['left']  = out_left.copy()
                    last_commanded_hand['right'] = out_right.copy()
            elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp") and args.input_mode == "controller":
                new_left  = tele_data.left_ctrl_triggerValue
                new_right = tele_data.right_ctrl_triggerValue
                if entering_pause:
                    frozen_hand_snapshot['left']  = last_commanded_hand.get('left',  new_left)
                    frozen_hand_snapshot['right'] = last_commanded_hand.get('right', new_right)
                if current_tick_paused:
                    pass   # Do not overwrite — hand controller reads last values
                else:
                    if alpha < 1.0:
                        f_left  = frozen_hand_snapshot['left']
                        f_right = frozen_hand_snapshot['right']
                        out_left  = f_left  + alpha * (new_left  - f_left)
                        out_right = f_right + alpha * (new_right - f_right)
                    else:
                        out_left, out_right = new_left, new_right
                    with left_gripper_value.get_lock():
                        left_gripper_value.value = out_left
                    with right_gripper_value.get_lock():
                        right_gripper_value.value = out_right
                    last_commanded_hand['left']  = out_left
                    last_commanded_hand['right'] = out_right
            else:
                pass
            
            # high level control
            if args.input_mode == "controller" and args.motion:
                # quit teleoperate
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                # command robot to enter damping mode. soft emergency stop function
                if tele_data.left_ctrl_thumbstick and tele_data.right_ctrl_thumbstick:
                    loco_wrapper.Damp()
                # https://github.com/unitreerobotics/xr_teleoperate/issues/135, control, limit velocity to within 0.3
                loco_wrapper.Move(-tele_data.left_ctrl_thumbstickValue[1] * 0.3,
                                  -tele_data.left_ctrl_thumbstickValue[0] * 0.3,
                                  -tele_data.right_ctrl_thumbstickValue[0]* 0.3)

            # get current robot state data.
            current_lr_arm_q  = arm_ctrl.get_current_dual_arm_q()
            current_lr_arm_dq = arm_ctrl.get_current_dual_arm_dq()

            # solve ik using motor data and wrist pose, then use ik results to control arms.
            # IK is solved every tick even during pause so the solver's warm-start state
            # (self.init_data seed + smooth_filter history) stays current for resume.
            time_ik_start = time.time()
            sol_q, sol_tauff  = arm_ik.solve_ik(tele_data.left_wrist_pose, tele_data.right_wrist_pose, current_lr_arm_q, current_lr_arm_dq)
            time_ik_end = time.time()
            logger_mp.debug(f"ik:\t{round(time_ik_end - time_ik_start, 6)}")

            if entering_pause:
                # Freeze at the pose we actually commanded LAST tick (C0-continuous even
                # if we were mid-ramp). Bootstrap with sol_q on the very first tick.
                #
                # Note: arm bootstrap falls back to fresh IK output (sol_q), while hand
                # bootstraps fall back to fresh VR input (new_left/new_right). The asymmetry
                # is benign because in both cases the bootstrap value is unused while paused
                # — paused arm re-sends frozen, paused hand skips writes — and is only the
                # blend-from anchor on resume, which gets overwritten by `last_commanded_*`
                # at every non-paused tick. So the first-tick fallback can never produce
                # observable behavior.
                frozen_sol_q     = (last_commanded_q     if last_commanded_q     is not None else sol_q).copy()
                frozen_sol_tauff = (last_commanded_tauff if last_commanded_tauff is not None else sol_tauff).copy()

            if current_tick_paused:
                target_q, target_tauff = frozen_sol_q, frozen_sol_tauff
            elif alpha < 1.0:
                target_q     = frozen_sol_q     + alpha * (sol_q     - frozen_sol_q)
                # Linearly blending tauff is an approximation: tauff is feedforward
                # dynamics (pose- and velocity-dependent), not a pose state, so
                # interpolating it across the frozen→fresh boundary is not
                # physically rigorous. Acceptable here because the arm controller's
                # PD term dominates during the 1 s ramp window — tauff is a hint,
                # not the primary command — and magnitudes are small at the
                # near-quasistatic teleop rates we run.
                target_tauff = frozen_sol_tauff + alpha * (sol_tauff - frozen_sol_tauff)
            else:
                target_q, target_tauff = sol_q, sol_tauff

            arm_ctrl.ctrl_dual_arm(target_q, target_tauff)

            last_commanded_q     = target_q.copy()
            last_commanded_tauff = target_tauff.copy()

            # record data
            if args.record:
                READY = recorder.is_ready() # now ready to (2) enter RECORD_RUNNING state
                # dex hand or gripper
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "hand":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                elif args.ee == "dex1" and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = arm_ctrl.get_current_motor_q().tolist()
                        current_body_action = [-tele_data.left_ctrl_thumbstickValue[1]  * 0.3,
                                               -tele_data.left_ctrl_thumbstickValue[0]  * 0.3,
                                               -tele_data.right_ctrl_thumbstickValue[0] * 0.3]
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp" or args.ee == "brainco") and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:6]
                        right_ee_state = dual_hand_state_array[-6:]
                        left_hand_action = dual_hand_action_array[:6]
                        right_hand_action = dual_hand_action_array[-6:]
                        current_body_state = []
                        current_body_action = []
                elif (args.ee == "inspire_dfx" or args.ee == "inspire_ftp") and args.input_mode == "controller":
                    with dual_gripper_data_lock:
                        left_ee_state = [dual_gripper_state_array[0]]
                        right_ee_state = [dual_gripper_state_array[1]]
                        left_hand_action = [dual_gripper_action_array[0]]
                        right_hand_action = [dual_gripper_action_array[1]]
                        current_body_state = []
                        current_body_action = []
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_state = []
                    current_body_action = []

                # arm state and action
                left_arm_state  = current_lr_arm_q[:7]
                right_arm_state = current_lr_arm_q[-7:]
                left_arm_action = sol_q[:7]
                right_arm_action = sol_q[-7:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img.bgr[:, :camera_config['head_camera']['image_shape'][1]//2]
                            colors[f"color_{1}"] = head_img.bgr[:, camera_config['head_camera']['image_shape'][1]//2:]
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{2}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{3}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    else:
                        if head_img is not None:
                            colors[f"color_{0}"] = head_img
                        else:
                            logger_mp.warning("Head image is None!")
                        if camera_config['left_wrist_camera']['enable_zmq']:
                            if left_wrist_img is not None:
                                colors[f"color_{1}"] = left_wrist_img.bgr
                            else:
                                logger_mp.warning("Left wrist image is None!")
                        if camera_config['right_wrist_camera']['enable_zmq']:
                            if right_wrist_img is not None:
                                colors[f"color_{2}"] = right_wrist_img.bgr
                            else:
                                logger_mp.warning("Right wrist image is None!")
                    # Base-velocity capture (spec §6). Atomic single-read of the
                    # latest DDS snapshot; the box[0] load is GIL-atomic in
                    # CPython so no lock is needed. None-fallbacks emit zeros
                    # so the row is always well-formed (frozen-stale invariant).
                    base_achieved, fsm_enum = read_sport_snapshot(latest_sport)
                    base_cmd = read_wireless_snapshot(latest_wireless)

                    states = {
                        "left_arm": {
                            "qpos":   left_arm_state.tolist(),    # numpy.array -> list
                            "qvel":   [],
                            "torque": [],
                        },
                        "right_arm": {
                            "qpos":   right_arm_state.tolist(),
                            "qvel":   [],
                            "torque": [],
                        },
                        "left_ee": {
                            "qpos":   left_ee_state,
                            "qvel":   [],
                            "torque": [],
                        },
                        "right_ee": {
                            "qpos":   right_ee_state,
                            "qvel":   [],
                            "torque": [],
                        },
                        "body": {
                            "qpos": current_body_state,
                        },
                        "base_achieved": {
                            "qpos": base_achieved,
                            "qvel": [],
                            "torque": [],
                        },
                    }
                    actions = {
                        "left_arm": {
                            "qpos":   left_arm_action.tolist(),
                            "qvel":   [],
                            "torque": [],
                        },
                        "right_arm": {
                            "qpos":   right_arm_action.tolist(),
                            "qvel":   [],
                            "torque": [],
                        },
                        "left_ee": {
                            "qpos":   left_hand_action,
                            "qvel":   [],
                            "torque": [],
                        },
                        "right_ee": {
                            "qpos":   right_hand_action,
                            "qvel":   [],
                            "torque": [],
                        },
                        "body": {
                            "qpos": current_body_action,
                        },
                        "base_cmd": {
                            "qpos": base_cmd,
                            "qvel": [],
                            "torque": [],
                        },
                    }
                    if args.sim:
                        sim_state = sim_state_subscriber.read_data()
                        recorder.add_item(
                            colors=colors, depths=depths,
                            states=states, actions=actions,
                            fsm_mode=fsm_enum,
                            sim_state=sim_state,
                        )
                    else:
                        recorder.add_item(
                            colors=colors, depths=depths,
                            states=states, actions=actions,
                            fsm_mode=fsm_enum,
                        )

                    # LeRobot writer can abort the episode mid-recording (queue
                    # overflow, max-duration, malformed input) — transition the
                    # FSM back to idle so the operator can re-record as the
                    # same episode ID. Json writer has no such attribute;
                    # getattr() keeps this branch-safe for both backends.
                    if getattr(recorder, 'episode_corrupted', False):
                        logger_mp.error(
                            "❌ Episode aborted by writer — resetting "
                            "RECORD_RUNNING. Press [s] to start a fresh take."
                        )
                        RECORD_RUNNING = False

            current_time = time.time()
            time_elapsed = current_time - start_time
            sleep_time = max(0, (1 / args.frequency) - time_elapsed)
            time.sleep(sleep_time)
            logger_mp.debug(f"main process sleep: {sleep_time}")

    except KeyboardInterrupt:
        logger_mp.info("⛔ KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        if not homie_abort.is_set():
            try:
                arm_ctrl.ctrl_dual_arm_go_home()
            except Exception as e:
                logger_mp.error(f"Failed to ctrl_dual_arm_go_home: {e}")
        else:
            logger_mp.warning(
                "Skipping ctrl_dual_arm_go_home — HOMIE-abort active. "
                "Arms hold last commanded pose; do NOT publish into a falling robot.")
        
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        
        try:
            img_client.close()
        except Exception as e:
            logger_mp.error(f"Failed to close image client: {e}")

        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")

        try:
            if not args.motion:
                pass
                # status, result = motion_switcher.Exit_Debug_Mode()
                # logger_mp.info(f"Exit debug mode: {'Success' if status == 3104 else 'Failed'}")
        except Exception as e:
            logger_mp.error(f"Failed to exit debug mode: {e}")

        try:
            if args.sim:
                sim_state_subscriber.stop_subscribe()
        except Exception as e:
            logger_mp.error(f"Failed to stop sim state subscriber: {e}")

        try:
            sport_sub.Close()
            wireless_sub.Close()
        except Exception as e:
            logger_mp.error(f"Failed to close base-vel subscribers: {e}")

        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")

        if gate is not None:
            try:
                gate.stop()       # zero velocity to HOMIE — best effort
            finally:
                gate.close()
        logger_mp.info("✅ Finally, exiting program.")
        exit(0)