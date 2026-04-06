#!/usr/bin/python3

import os
import sys
import time
import traceback

import av
import cv2
import numpy as np
import rospy
import tellopy
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import Bool, Float32MultiArray

from typing import Tuple

from ros_data import ROSData
from topic_names import IMAGE_TOPIC, REACHED_GOAL_TOPIC, WAYPOINT_TOPIC
from utils import clip_angle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "../config/robot.yaml")

with open(CONFIG_PATH, "r") as f:
    robot_config = yaml.safe_load(f)

MAX_V = robot_config["max_v"]
MAX_W = robot_config["max_w"]
DT = 1 / robot_config["frame_rate"]
EPS = 1e-8
WAYPOINT_TIMEOUT = 1

reached_goal = False
reverse_mode = False
waypoint = ROSData(WAYPOINT_TIMEOUT, name="waypoint")
frame_count = 0


def reset(drone):
    drone.set_roll(0.0)
    drone.set_pitch(0.0)
    drone.set_yaw(0.0)


def cb_cmd_vw(drone, v, w):
    drone.set_pitch(v)
    drone.set_yaw(-w)


def handler(event, sender, data, **args):
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        pass


def pd_controller(current_waypoint: np.ndarray) -> Tuple[float, float]:
    assert len(current_waypoint) in (2, 4), "waypoint must be a 2D or 4D vector"
    if len(current_waypoint) == 2:
        dx, dy = current_waypoint
    else:
        dx, dy, hx, hy = current_waypoint
    if len(current_waypoint) == 4 and abs(dx) < EPS and abs(dy) < EPS:
        v = 0
        w = clip_angle(np.arctan2(hy, hx)) / DT
    elif abs(dx) < EPS:
        v = 0
        w = np.sign(dy) * np.pi / (2 * DT)
    else:
        v = dx / DT
        w = np.arctan(dy / dx) / DT
    v = np.clip(v, 0, MAX_V)
    w = np.clip(w, -MAX_W, MAX_W)
    return v, w


def callback_drive(waypoint_msg: Float32MultiArray):
    waypoint.set(waypoint_msg.data)


def callback_reached_goal(reached_goal_msg: Bool):
    global reached_goal
    reached_goal = reached_goal_msg.data


def main():
    global frame_count
    frame_times = []
    bridge = CvBridge()
    drone = tellopy.Tello()
    out = None

    try:
        drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)
        drone.connect()
        drone.wait_for_connection(10.0)
        drone.takeoff()
        time.sleep(1)

        retry = 3
        container = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        while container is None and retry > 0:
            retry -= 1
            try:
                container = av.open(drone.get_video_stream())
            except av.AVError as err:
                print(err)
                print("retry...")

        if container is None:
            raise RuntimeError(
                "Unable to open the Tello video stream after 3 attempts. "
                "Check the drone connection and confirm video streaming is available."
            )

        frame_skip = 300
        while True:
            for frame in container.decode(video=0):
                if frame_skip > 0:
                    frame_skip -= 1
                    continue

                start_time = time.time()
                if frame_count == 0:
                    start_time_all = time.time()
                    image = np.array(frame.to_image())
                    height, width = image.shape[:2]
                    out = cv2.VideoWriter("vlfly_fpv.mp4", fourcc, 20.0, (width, height))

                frame_times.append(time.time())
                if len(frame_times) >= 30:
                    fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                    print(f"[Image FPS] ~{fps:.2f} fps")
                    frame_times = []

                frame_count += 1
                image = np.array(frame.to_image())
                cv2.imshow("VLFly FPV", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

                bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                out.write(bgr_image)

                img_msg = bridge.cv2_to_imgmsg(image, encoding="rgb8")
                image_curr_msg.publish(img_msg)

                if reached_goal:
                    reset(drone)
                    cb_cmd_vw(drone, 0, 0)
                    drone.land()
                    print("Reached goal! Stopping...")
                    print("End Frame:", frame_count)
                    print("End Time:", time.time() - start_time_all)
                    time.sleep(5)
                    drone.quit()
                    cv2.destroyAllWindows()
                    return
                elif waypoint.is_valid(verbose=True):
                    v, w = pd_controller(waypoint.get())
                    print(f"Publishing control: v={v:.3f}, w={w:.3f}")
                    if reverse_mode:
                        v *= -1
                    cb_cmd_vw(drone, v, w)
                    time.sleep(0.01)

                time_base = 1.0 / 60 if frame.time_base < 1.0 / 60 else frame.time_base
                frame_skip = int((time.time() - start_time) / time_base)
    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        drone.quit()
        if out:
            out.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    rospy.init_node("vlfly_tello_bridge", anonymous=True)
    image_curr_msg = rospy.Publisher(IMAGE_TOPIC, RosImage, queue_size=1)
    rospy.Subscriber(WAYPOINT_TOPIC, Float32MultiArray, callback_drive, queue_size=1)
    rospy.Subscriber(REACHED_GOAL_TOPIC, Bool, callback_reached_goal, queue_size=1)
    main()
