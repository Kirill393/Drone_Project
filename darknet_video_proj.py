from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
from datetime import datetime
import signal
import sys
from statistics import mean
import math
import numpy as np

class HistoryElement:
    """
    Class representing the element in history

    Attributes
    ----------
    detection : darknet detection
        the detection that was made
    num_fake : str
        number of fake drones predictions that were added so far for specific drone track
    search_radius : str
        search radius around detection to search for the next detection
    noise_counter : int
        number of noise detection that were made so far

    Methods
    -------
    set_noise(count)
        set noise_counter parameter to count
    add_fake()
        add 1 to num_fake
    """
    detection = 0
    num_fake = 0
    search_radius = 0
    noise_counter = 0
    
    def __init__(self, detection, num_fake, search_radius, noise_counter):
        self.detection = detection
        self.num_fake = num_fake
        self.search_radius = search_radius
        self.noise_counter = noise_counter
    
    def set_noise(self, count):
        self.noise_counter = count
    
    def add_fake(self):
        self.num_fake += 1
    

class HistoryLine:
    """
    Class representing the history track of a specific drone

    Attributes
    ----------
    history_elements : array of HistoryElement
        all the detections that were made of a specific drone

    Methods
    -------
    get_last()
        get last HistoryElement of the drone
    new_element(element)
        add new HistoryElement to history_elements
    crop_old(history_timer)
        remove old HistoryElements that are further than history count in the history_elements
        
    """
    history_elements = 0
    
    def __init__(self):
        self.history_elements = []
    
    def __init__(self, element):
        self.history_elements = []
        self.history_elements.append(element)
    
    def get_last(self):
        return self.history_elements[len(self.history_elements) - 1]
    
    def new_element(self, element):
        self.history_elements.append(element)
    
    def crop_old(self, history_timer):
        self.history_elements =  self.history_elements[len(self.history_elements)-history_timer:len(self.history_elements)]
    

class DetectionHistory:
    """
    Class representing the total drone tracking history of the system

    Attributes
    ----------
    history_lines : array of HistoryLine
        array of all the drones the system is traking 

    Methods
    -------
    new_line()
        add new drone track (HistoryLine) to the system
    new_line(element)
        add new drone track (HistoryLine) to the system with new HistoryElement
    remove_lines(list(int))
        remove history lines in the list
        
    """
    history_lines = 0
    
    def __init__(self):
        self.history_lines = []
    
    def new_line(self):
        self.history_lines.append(HistoryLine())
        
    def new_line(self, element):
        self.history_lines.append(HistoryLine(element))
        
    def remove_lines(self, lines):
        for idx in reversed(lines):
            self.history_lines.pop(idx)

def signal_handler(self, sig, frame):
    """
    Close video (so the video and the results.txt will be saved) when given CTAL+C.
    Relevant mainly for webcam scenario (infinite loop).
    """
    print("You pressed CTAL+C!, exiting while save log & output video (if there is) properly")
    cap.release()
    video.release()
    cv2.destroyAllWindows()
    time.sleep(3)
    exit()

 # CTRL+C
signal.signal(signal.SIGINT, signal_handler)
# stop on debugger
signal.signal(signal.SIGTERM, signal_handler)

def avg(x, width):
    """
    Claculate average of "width" last elements in "x" array
    args:
        x : array of float numbers
        width (int): over how many elements in x to average over
    returns:
        (float): the average of the elements, None if "x" has fewer elements than width
    """
    if len(x) >= width:
        return sum(x[len(x)-width:len(x)])/width
    else:
        return None
def mag(x): 
    """
    Claculate magnitude of vector x
    args:
        x : array of float numbers
    returns:
        (float): magnitude
    """
    return math.sqrt(sum(i**2 for i in x))

def mag_steps(x, a, width):
    """
    calculate weighted average magnitude over all "x" elements
    args:
        x : array of float numbers
        a (float): weight
        width (int): amount of elements to average over
    returns:
        (float): weighted magnitude
    """
    if len(x) == 1:
        return mag(x[0])
    if len(x) < width:
        return mag(x[len(x)-1])
    
    value_last = avg(x, width)
    x_cut = x[0:len(x)-1]
    return a*mag(value_last) + (1-a)*mag_steps(x_cut, a, width)

def angle_steps(x, a, width):
    """
    calculate weighted average Angle over all "x" elements
    args:
        x : array of float numbers
        a (float): weight
        width (int): amount of elements to average over
    returns:
        (float): weighted angle
    """
    if len(x) == 1:
        return x[0]
    if len(x) < width:
        return x[len(x)-1]
    
    value_last = avg(x, width)
    x_cut = x[0:len(x)-1]
    #print(a*value_last)
    return np.add(a*value_last, (1-a)*angle_steps(x_cut, a, width))

def isInside_zone(detect, zone_x, zone_y, zone_radius):
    """
    Check if detect is insize the zone
    args:
        detect : Detection
        zone_x (float): zone's x axis value
        zone_y (float): zone's y axis value
        zone_radius (float): zone radius 
    returns:
        (Bool): True if detect iside zone, False otherwise
    """
    x = detect[2][0]
    y = detect[2][1]
    if pow(zone_x-x, 2) + pow(zone_y-y, 2) <= pow(zone_radius, 2):
        return True
    else:
        return False
    
def parameter_M(detection_history, detect):
    """
    Calculate parameter M of detect for specific history line
    args:
        detection_history : history line
        detect : detection
    returns:
        (float): M parameter of current detections relative to the history line
    """
    # Return if the drone is a fake one
    if float(detect[1]) < 0.01:
        return 0
        
    # Get predicted future location of the drone
    new_x, new_y = calculate_new_spot(detection_history)
    
    # Calculate distance between predicted spot and actual detection
    new_x = new_x/width
    new_y = new_y/height
    x = detect[2][0]/width
    y = detect[2][1]/height
    
    distance_param = ((math.sqrt(pow(x-new_x, 2) + pow(y-new_y, 2)))*(-1)) + 1 # 0 if the are very far away, 1 if they are at the same spot
    
    # Calculate the actual history size
    frame_history_count = args.hf
    
    range_max = len(detection_history.history_elements)
    frame_history_count_actual = min(range_max, frame_history_count)
    
    # Calculate size over certain amount of frame history
    width_avg = 0
    height_avg = 0
    
    for i in range(frame_history_count_actual):
        width_avg += detection_history.history_elements[range_max - 1 - i].detection[2][2]/width
        height_avg += detection_history.history_elements[range_max - 1 - i].detection[2][3]/height
    
    
    width_avg = width_avg/(frame_history_count_actual)
    height_avg = height_avg/(frame_history_count_actual)
    
    detect_width = detect[2][2]
    detect_height = detect[2][3]
    
    size_param = ((min(detect_width, width_avg)/max(detect_width, width_avg))+(min(detect_height, height_avg)/max(detect_height, height_avg)))/2
    
    if frame_history_count_actual == 1:
        return 10*distance_param + 2*size_param
    
    # Calculate velocity and size over certain amount of frame history
    velocity_x = (detection_history.history_elements[range_max - 1].detection[2][0] - detection_history.history_elements[range_max - frame_history_count_actual].detection[2][0])/(frame_history_count_actual -1)# - sum_x
    velocity_y = (detection_history.history_elements[range_max - 1].detection[2][1] - detection_history.history_elements[range_max - frame_history_count_actual].detection[2][1])/(frame_history_count_actual -1)# - sum_y
    
    
    # Calculate velocity difference and angle
    velocity_x_new = x - detection_history.history_elements[range_max - 1].detection[2][0]/width
    velocity_y_new = y - detection_history.history_elements[range_max - 1].detection[2][1]/height
    
    velocity_mag_new = math.sqrt(pow(velocity_x_new, 2) + pow(velocity_y_new, 2))
    velocity_mag = math.sqrt(pow(velocity_x, 2) + pow(velocity_y, 2))
    
    if velocity_mag == 0 and velocity_mag_new == 0:
        velocity_angle_param = 1
        velocity_mag_param = 1
    elif (velocity_mag == 0 and velocity_mag_new != 0) or (velocity_mag != 0 and velocity_mag_new == 0):
        velocity_angle_param = 0
        velocity_mag_param = 0
    else:
        velocity_angle_param = ((((velocity_y * velocity_y_new) + (velocity_x * velocity_x_new))/(velocity_mag_new * velocity_mag)) + 1)/2
        velocity_mag_param = abs(min(velocity_mag, velocity_mag_new)/max(velocity_mag, velocity_mag_new))

    return 10*distance_param + 1*size_param + 20*velocity_angle_param + 1*velocity_mag_param
    
    
def calculate_new_spot(detection_history):
    """
    Predict next drone location based on history
    args:
        detection_history : history line
    returns:
        [new_x, new_y]: Predicted x and y coordinates of new location
    """
    # Calculate the actual history size
    frame_history_count = args.hf
    
    range_max = len(detection_history.history_elements)
    frame_history_count_actual = min(range_max, frame_history_count)
    
    if frame_history_count_actual == 1:
        new_x = detection_history.history_elements[range_max - 1].detection[2][0]
        new_y = detection_history.history_elements[range_max - 1].detection[2][1]
        return [new_x, new_y]
    
    # Calculate difference between every couple of detections
    data = [[detection_history.history_elements[range_max - j].detection[2][0], detection_history.history_elements[range_max - j].detection[2][1]] for j in range(frame_history_count_actual, 0, -1)]
    
    diff = np.array([[data[i][0]-data[i-1][0], data[i][1]-data[i-1][1]] for i in range(1, frame_history_count_actual)])
    
    # Calculate the weighted angle and velocity
    V = mag_steps(diff, args.hv, args.hw)
    A = angle_steps(diff, args.ha, args.hw)
    if mag(A) == 0:
        
        x = 0
        y = 0
    else:
        x = (A[0] / mag(A))*V
        y = (A[1] / mag(A))*V
    diff_new = [x, y]
    
    point_last = data[len(data)-1]
    new_x = point_last[0] + diff_new[0]
    new_y = point_last[1] + diff_new[1]
    return [new_x, new_y]

def zoomin(drone_id, window_size, detect, original_image, darknet_image_zoomin, network_zoomin, class_names_zoomin, thresh_zoom):
    """
    Use zoomin network on croped image around the drone
    args:
        drone_id (int): drone ID
        window_size (int): croped image size
        detect : drone detection by the default network
        original_image : original image
        darknet_image_zoomin : darknet image
        network_zoomin : network
        class_names_zoomin : network labels
        thresh_zoom : network thresh
    returns:
        detection if the network detected a drone or None otherwise
    """

    original_width = len(original_image[0])
    original_height = len(original_image)
    width_ratio = original_width / width
    height_ratio = original_height / height
    
    # Crop the image to fit into zoomin network
    x = float(detect[2][0]) * width_ratio
    y = float(detect[2][1]) * height_ratio
    center_x = int(x)
    center_y = int(y)
    edge_low_x = center_x - int(window_size/2) 
    edge_low_y = center_y - int(window_size/2) 
    if edge_low_x < 0:
        edge_low_x = 0
    if edge_low_y < 0:
        edge_low_y = 0
    edge_high_x = edge_low_x + window_size
    edge_high_y = edge_low_y + window_size
    if edge_high_x > original_width:
        edge_high_x = original_width
        edge_low_x = edge_high_x - window_size
    if edge_high_y > original_height:
        edge_high_y = original_height
        edge_low_y = edge_high_y - window_size
    sub_image = original_image[edge_low_y:edge_high_y, edge_low_x:edge_high_x]
    
    darknet.copy_image_from_bytes(darknet_image_zoomin, sub_image.tobytes())
    detections_zoomin = darknet.detect_image(network_zoomin, class_names_zoomin, darknet_image_zoomin, thresh=float(thresh_zoom))
    if len(detections_zoomin) > 0:
        return (detections_zoomin[0][0] + "_"+str(drone_id), detect[1], detect[2])
    else:
        return None

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=0,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--export_logname", type=str, default="",
                        help="out log name. Not saved if empty")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--weights_zoomin", default="yolov4.weights",
                        help="yolo zoomin weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="window inference display. For headless systems")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--config_file_zoomin", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--data_file_zoomin", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    parser.add_argument("--capture_frame_width", type=int, default=3840,
                        help="define the camera frame width")
    parser.add_argument("--capture_frame_height", type=int, default=2160,
                        help="define the camera frame height")
    parser.add_argument("--history", default=1, type=int,
                        help="define the history capacity in frames")
    parser.add_argument("--z", default=0, type=float,
                        help="what threshold to use for zoomin")
    parser.add_argument("--hv", default=0.05, type=float,
                        help="speed magnitude weight")
    parser.add_argument("--ha", default=0.2, type=float,
                        help="speed angle weight")
    parser.add_argument("--hf", default=20, type=int,
                        help="how many frames to look back in history")
    parser.add_argument("--hw", default=3, type=int,
                        help="average size")
    parser.add_argument("--z_sub", default=0.8, type=float,
                        help="what threshold to use for zoomin sub")
    parser.add_argument("--thresh_sub", type=float, default=.98,
                        help="consider detections in the sub test only with thresh above this")
    parser.add_argument("--ignore_noise", type=float, default=2,
                        help="how many confirmed detection frames must be before target considered real")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed.
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    """
    Arguments checker, raises error for false arguments.
    """
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("\nInvalid config path {}".format(
            os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("\nInvalid weight path {}".format(
            os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("\nInvalid data file path {}".format(
            os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise(ValueError("\nInvalid video path {}".format(
            os.path.abspath(args.input))))
    if not args.export_logname:
        raise(ValueError("\nNeed to set results log-name"))
    if args.out_filename and not args.out_filename.endswith('.mp4'):
        raise(ValueError("\nOut file name need to end with '.mp4'"))


def set_saved_video(input_video, output_video, size):
    """
    creating the result video obj.
    """
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def video_capture(frame_queue, frame_queue_4k, darknet_image_queue):
    """
    reading frames from the caputre (webcam\video) and the time of caputre,
    and push them into queues for farther use.
    """
    width_input = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height_input = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    print("Input resolution is: {}x{} (if 0x0, then the camera is occupied with something else)".format(
        int(width_input), int(height_input)))
    
    first = True
    image_list=[]
    idx = 1
    
    while cap.isOpened():
        ret, frame_main = cap.read()
        if not ret:
            break
        # capture_time
        capture_time_queue.put(time.time())


        image_list.append(cv2.cvtColor(frame_main, cv2.COLOR_BGR2GRAY))
        if idx > 1:
            for frame in image_list:
                if first:
                    previous_frame_gray = frame
                    first = False
                    final_img = cv2.absdiff(frame,frame)
                    continue
                frame_diff = cv2.absdiff(frame,previous_frame_gray)
                
                previous_frame_gray = frame
                
                contrast = 3
                brightness = 0
                frame_diff = cv2.addWeighted( frame_diff, contrast, frame_diff, 0, brightness)
                
                final_img = final_img + frame_diff

            
            img_4channel = cv2.cvtColor(frame_main, cv2.COLOR_BGR2RGBA)
            img_4channel[:,:,3] = final_img
            first = True
            image_list.pop(0)
        else:
            frame_diff = cv2.absdiff(cv2.cvtColor(frame_main, cv2.COLOR_BGR2GRAY),cv2.cvtColor(frame_main, cv2.COLOR_BGR2GRAY))
            img_4channel = cv2.cvtColor(frame_main, cv2.COLOR_BGR2RGBA)
            img_4channel[:,:,3] = frame_diff
        idx += 1
        
        frame_main_rgb = cv2.cvtColor(frame_main, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(
            img_4channel, (width, height), interpolation=cv2.INTER_LINEAR)
        frame_resized_normal = cv2.resize(frame_main_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame_resized_normal)
        frame_queue_4k.put(frame_main_rgb)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        darknet_image_queue.put(darknet_image)
    cap.release()


def inference(frame_queue_4k, darknet_image_queue, detections_queue, fps_queue, history_timer):
    """
    inference the captures into the darknet.
    function is also in charge of the printing/writing (fps, caputre time, detections, Tracking).
    """
    # results log
    logname = args.export_logname
    
    """ INITILIZE PARAMETERS: """
    detect_history = DetectionHistory()
    drone_id = 0
    window_size = 160
    prev_time = 0
    r1_basic = 0.05
    r1_rate = 0.00015
    if history_timer < 2:
    	history_timer = 0
    frame_counter = 1
    test = 0
    """ INITILIZE PARAMETERS: END """
    """ OPTION: """
    
    # each time will open a new txt file
    logname_split = args.export_logname.rsplit(".", 1)
    index = 0
    while 1:
        # name_<index>.txt
        logname = logname_split[0] + '_' + str(index) + '.' + logname_split[1]
        # file not exists
        if not os.path.isfile(logname):
            break
        # trying next index
        index += 1
    """ OPTION: END """
    f = open(logname, "w")
    enter_time_queue = [0, 0, 0]
    exit_time_queue = [1, 1, 1]
    while cap.isOpened():
        # get new image from queue
        darknet_image = darknet_image_queue.get()
        original_image = frame_queue_4k.get()
        # sample entering time
        prev_time = time.time()
        enter_time_queue.pop(0)
        enter_time_queue.append(prev_time)
        # detect image (inference image in neural network)
        detections = darknet.detect_image(
            network, class_names, darknet_image, thresh=args.thresh)
        
        """ TRACKING SYSTEM: """
        
        # Check if using tracking system
        if history_timer > 1:
            M_table = []
            idx = 0
            
            # =================================================================
            # -- Calculate realtions between old detections and new detections
            #==================================================================
            for line in detect_history.history_lines:
                idx_detect = 0
                for detect in detections:
                    if idx == 0:
                        M_table.append([]) 
                    
                    # Get search parameters
                    zone_radius = line.get_last().search_radius
                    zone_x, zone_y = calculate_new_spot(line)
                    
                    #Check if drone inside zone
                    if isInside_zone(detect, zone_x, zone_y, zone_radius):
                        param_m = parameter_M(line, detect)
                        M_table[idx_detect].append(param_m)
                    else:
                        M_table[idx_detect].append(0)
                    idx_detect+=1
                idx+=1
            
            # =================================================================
            # -- Adding the new detections to proper previous detections
            #==================================================================
            unconnected_detections = list(range(0, len(detections)))
            if M_table != []:
                not_checked = list(range(len(M_table[0])))
                order = []
                
                # Determine the order in which new detection should be connected to history lines
                while not_checked:
                    max_col = 0
                    row = 0
                    
                    # Find the maximum M parameter for specific history line
                    for j in not_checked:
                        col = [c[j] for c in M_table]
                        tmp_max = max(col)
                        
                        if tmp_max >= max_col:
                            max_col = tmp_max
                            index_max = j
                            row = col.index(tmp_max)
                    # Add detection if found one
                    if max_col > 0:
                        order.append((index_max, row))
                    # Nullify all other detections for this specific history line
                    for i in range(len(M_table[0])):
                        M_table[row][i] = 0
                    not_checked.remove(index_max)
            
                # Create all the new Hisotry elements and add them to corresponding history lines
                for j, idx_min in order:
                    # Create detection
                    detect = (detect_history.history_lines[j].get_last().detection[0], detections[idx_min][1], detections[idx_min][2])
                    # Create HistoryElements
                    new_detection = HistoryElement(detection = detect, num_fake = 0, search_radius = r1_basic * width, noise_counter = detect_history.history_lines[j].get_last().noise_counter)
                    # Add the HistoryElements to history line
                    detect_history.history_lines[j].new_element(new_detection)
                    unconnected_detections.remove(idx_min)
                
            
            # =================================================================
            # -- For all previous detections add fake drone if necessary, remove if too old
            #==================================================================
            tmp_list = []
            idx_remove = -1
            for line in detect_history.history_lines:
                idx_remove += 1
                # Remove history lines if the fake drone cound over the limit or the fake drone outside the screen
                if line.get_last().num_fake > history_timer or line.get_last().detection[2][0]/width > 1.01 or line.get_last().detection[2][1]/height > 1.01 or line.get_last().detection[2][0]/width < -0.01 or line.get_last().detection[2][1]/height < -0.01:
                    tmp_list.append(idx_remove)
                    continue
                # Add fake drone
                if line.get_last().num_fake > 0 and len(line.history_elements) > 1:
                    # Calculate new fake drone parameters
                    x_new, y_new = calculate_new_spot(line)
                    r1 = (r1_basic + r1_rate*line.get_last().num_fake) * width
                    labl = line.get_last().detection[0]
                    if labl.split("_")[0] == 'UNKNOWN':
                        detect_new = None
                        if args.z != 0:
                            detect_new = zoomin(drone_id, window_size, detect, original_image, darknet_image_zoomin, network_zoomin, class_names_zoomin, args.z_sub)
                        if detect_new != None:
                            labl = detect_new[0]
                    # Create new element and add it to the history line
                    new_detect = ((labl),(0),((x_new),(y_new),(line.get_last().detection[2][2]),(line.get_last().detection[2][3])))
                    new_element = HistoryElement(detection = new_detect, num_fake = line.get_last().num_fake + 1, search_radius = r1, noise_counter = line.get_last().noise_counter)
                    line.new_element(new_element)
            detect_history.remove_lines(tmp_list)  
            
            # =================================================================
            # -- Add fake drone counter to all lines in detect_history
            #==================================================================
            for line in detect_history.history_lines:
                line.get_last().add_fake()
            
            # =================================================================
            # -- All new unconnected detection make into new lines
            #==================================================================
            i = 0
            for detect in detections:
                if i in unconnected_detections:
                    if args.z != 0:
                        detect_new = zoomin(drone_id, window_size, detect, original_image, darknet_image_zoomin, network_zoomin, class_names_zoomin, args.z)
                    else:
                        detect_new = (str(detect[0]) + "_"+str(drone_id), detect[1], detect[2])
                    
                    if detect_new == None and float(detect[1]) > 100*float(args.thresh_sub):
                        detect_new = ('UNKNOWN' + "_"+str(drone_id), detect[1], detect[2])
                    if detect_new != None:
                        new_element = HistoryElement(detection = detect_new, num_fake = 1, search_radius = r1_basic * width, noise_counter = 0)
                        detect_history.new_line(new_element)
                        drone_id += 1
                i+=1
            detections_v2 = []
            
            # =================================================================
            # -- Update all noise counters in detection history
            #==================================================================
            
            for line in detect_history.history_lines:
                if len(line.history_elements) > history_timer:
                    line.crop_old(history_timer)
                # If noise already passed the limit no need to calculate
                if line.get_last().noise_counter > args.ignore_noise:
                    detections_v2.append(line.get_last().detection)
                # Calculate the noise by counting all real detections in drones history
                else:
                    count = 0
                    for frm in line.history_elements:
                        if float(frm.detection[1]) > 0.1:
                            count += 1
                            # Quit if counted enough real drones
                            if count > args.ignore_noise:
                                line.get_last().set_noise(count)
                                detections_v2.append(line.get_last().detection)
                                break
        # Not using the tracking system but using the Zommin network
        else:
            if args.z != 0:
                for detect in detections:
                    detect_new = zoomin(drone_id, window_size, detect, original_image, darknet_image_zoomin, network_zoomin, class_names_zoomin, args.z)
                    if detect_new != None:
                        detections_tmp.append((detect_new[0], detect[1], detect[2]))
                detections = detections_tmp
        
        """ TRACKING SYSTEM: END """
        
        # store result in queue
        if history_timer > 1:
            detections_queue.put(detections_v2)
        else:
            detections_queue.put(detections)
        # calculate fps of passing image
        fps = float(1 / (time.time() - prev_time))
        exit_time_queue.pop(0)
        exit_time_queue.append(time.time())
        # store fps in queue
        fps_queue.put(int(fps))
        # calculate the average fps of 3 last frame (just to follow up)
        fps_list = [1./(m - n)
                    for m, n in zip(exit_time_queue, enter_time_queue)]
        print("Average FPS over last 3 frames is: {:.2f}".format(
            mean(fps_list)))
        # store capture time to file (in ms, for ground station)
        f.write("time: {} , frame: {}\n".format(str(round(capture_time_queue.get()*1000)), frame_counter))
        frame_counter += 1
        # store bbox to file
        #height_ratio = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)/height
        #width_ratio = cap.get(cv2.CAP_PROP_FRAME_WIDTH)/width
        height_ratio = 1/height
        width_ratio = 1/width
        if history_timer > 1:
            darknet.print_detections(detections_v2, height_ratio, width_ratio, f)
        else:
            darknet.print_detections(detections, height_ratio, width_ratio, f)
        
    cap.release()
    f.close()
    print("\nFinished successfully, results: {}".format(logname))


def drawing(frame_queue, detections_queue, fps_queue):
    """
    drawing bbox on the image and writing results video file or show video image.
    """
    # so we could release it if a signal is given
    global video
    # deterministic bbox colors
    random.seed(3)
    # results video file
    filename = args.out_filename
    # each time will open a new out file
    if args.out_filename:
        filename_split = args.out_filename.rsplit(".", 1)
        index = 0
        while 1:
            # save file: name_<index>.mp4
            filename = filename_split[0] + '_' + \
                str(index) + '.' + filename_split[1]
            # file not exists
            if not os.path.isfile(filename):
                break
            # trying next index
            index += 1
    # result video obj
    video = set_saved_video(
        cap, filename, (args.capture_frame_width, args.capture_frame_height))
    while cap.isOpened():
        frame_resized = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        if frame_resized is not None:
            # draw detection bounding boxs on image
            image = darknet.draw_boxes(detections, frame_resized, class_colors)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (args.capture_frame_width,
                                       args.capture_frame_height), interpolation=cv2.INTER_LINEAR)
            # writing video image
            if args.out_filename is not None:
                video.write(image)
            # show video image
            if not args.dont_show:
                cv2.imshow('Inference', image)
            # Esc key to stop GUI
            if cv2.waitKey(fps) == 27:
                break
    # Closes video file or capturing device
    cap.release()
    # Closes video write file
    video.release()
    # destroys all of the opened HighGUI windows
    cv2.destroyAllWindows()
    if args.out_filename:
        print("\nOut file: {}".format(filename))


if __name__ == '__main__':
    frame_queue = Queue()
    frame_queue_4k = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)
    capture_time_queue = Queue(maxsize=1)

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
        args.config_file, args.data_file, args.weights, batch_size=1)
    if args.z != 0:
        network_zoomin, class_names_zoomin, class_colors_zoomin = darknet.load_network(
            args.config_file_zoomin, args.data_file_zoomin, args.weights_zoomin, batch_size=1)
    if args.z != 0:
        class_colors_zoomin['UNKNOWN']=(255,0,0)
        class_colors_zoomin['DJI']=(0,0,255)
        class_colors_zoomin['Storm']=(0,0,255)
        class_colors_zoomin['Inspire']=(0,0,255)
        class_colors_zoomin['Matrice']=(0,0,255)
        class_colors_zoomin['Mavic']=(0,0,255)
        class_colors_zoomin['Tello']=(0,0,255)
        class_colors['drone']=(255,0,0)
        class_colors = class_colors_zoomin
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 4)
    if args.z != 0:
        darknet_image_zoomin = darknet.make_image(160, 160, 3)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_frame_height)
    Thread(target=video_capture, args=(
        frame_queue, frame_queue_4k, darknet_image_queue)).start()
    Thread(target=inference, args=(frame_queue_4k, darknet_image_queue,
                                   detections_queue, fps_queue, int(args.history))).start()
    Thread(target=drawing, args=(frame_queue,
                                 detections_queue, fps_queue)).start()
