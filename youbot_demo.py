from vrep import VRep
from vrep.const import *
from youbot import YouBot
from time import sleep
import numpy as np
import cv2
import atexit
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
from mpl_toolkits.mplot3d import Axes3D     # Implicitely needed for 3D projections
from time import perf_counter as timer
from youbot.transforms import angdiff
import multiprocessing

# Illustrates the V-REP bindings.

# (C) Copyright Renaud Detry 2013, Thibaut Cuvelier 2017, Mathieu Baijot 2017.
# Distributed under the GNU General Public License.
# (See http://www.gnu.org/copyleft/gpl.html)


# Open house.ttt in vrep
# Run simRemoteApi.start(19998) in the vrep LUA console
# Run this script

def rel2abs(X_rel, Y_rel, youbot_pos, angle):
    X_rel = np.array(X_rel)
    Y_rel = np.array(Y_rel)
    X_abs = X_rel * np.cos(angle) - Y_rel * np.sin(angle)
    Y_abs = X_rel * np.sin(angle) + Y_rel * np.cos(angle)
    X_abs += youbot_pos[0]
    Y_abs += youbot_pos[1]
    return X_abs, Y_abs

class YoubotHandler:
    def __init__():
        pass

if __name__ == '__main__':
#    cv2.imshow('test', (np.ones((100, 100))*220).astype("uint8"))
#    cv2.waitKey(0)
#        pass

    ## Initiate the connection to the simulator. 
    print('Program started')
    VRep.simxFinish(-1)
    vrep = VRep('127.0.0.1', 19997, True, True, 2000, 5)
    print('Connection %d to the remote API server open.\n' % vrep.clientID)

    # This will only work in "continuous remote API server service". 
    # See http://www.v-rep.eu/helpFiles/en/remoteApiServerSide.htm
    vrep.simxStartSimulation(simx_opmode_oneshot_wait)
    
    # Stop the simulation whenever exiting (e.g. ctrl-C)
    @atexit.register
    def stop_simulation():
        vrep.simxStopSimulation(simx_opmode_oneshot_wait)
        vrep.simxFinish(vrep.clientID)

    # Retrieve all handles, and stream arm and wheel joints, the robot's pose, the Hokuyo, and the 
    # arm tip pose. The tip corresponds to the point between the two tongs of the gripper (for 
    # more details, see later or in the file focused/youbot_arm.m). 
    youbot = YouBot(vrep)
    youbot.examples(vrep)
    youbot.hokuyo_init(vrep)
    
    # Let a few cycles pass to make sure there's a value waiting for us next time we try to get a 
    # joint angle or the robot pose with the simx_opmode_buffer option.
    sleep(.2)

    ## Youbot constants
    # The time step the simulator is using (your code should run close to it). 
    timestep = .1

    # Minimum and maximum angles for all joints. Only useful to implement custom IK. 
    arm_joint_ranges = [-2.9496064186096, 2.9496064186096
                        -1.5707963705063, 1.308996796608
                        -2.2863812446594, 2.2863812446594
                        -1.7802357673645, 1.7802357673645
                        -1.5707963705063, 1.5707963705063]

    # Definition of the starting pose of the arm (the angle to impose at each joint to be in the 
    # rest position).
    starting_joints = np.deg2rad(np.array([0, 30.91, 52.42, 72.68, 0]))

    ## Preset values for the demo. 
    print('Starting robot')
    
    # Define the preset pickup pose for this demo. 
    pickup_joints = np.deg2rad(np.array([90, 19.6, 113, -41, 0]))

    # Parameters for controlling the youBot's wheels: at each iteration, those values will be set 
    # for the wheels.
    forw_back_vel = 0       # Move straight ahead. 
    right_vel = 0           # Go sideways. 
    rotate_right_vel = 0    # Rotate. 
    prev_orientation = 0    # Previous angle to goal (easy way to have a condition on the robot's 
                            # angular speed). 
    prev_position = 0       # Previous distance to goal (easy way to have a condition on the 
                            # robot's speed). 

    # Set the arm to its starting configuration. 
    vrep.simxPauseCommunication(True) 
    for arm_joint, starting_joint in zip(youbot.arm_joints, starting_joints):
        vrep.simxSetJointTargetPosition(arm_joint, starting_joint)
    vrep.simxPauseCommunication(False) 

    # Initialise the plot. 
    plot_data = True
    if plot_data:
        # Prepare the plot area to receive three plots: what the Hokuyo sees at the top (2D map), 
        # the point cloud and the image of what is in front of the robot at the bottom.
        plt.ion()   # Enable interactive mode
        sensor_ax = plt.subplot(211)    # type: plt.Axes
        camera_ax = plt.subplot(224)    # type: plt.Axes
        pt_cloud_ax = plt.subplot(223, projection='3d') # type: plt.Axes
        canvas = plt.gcf().canvas
        plt.draw()

    # Make sure everything is settled before we start. (is this necessary?)
    # sleep(2)

    # Retrieve the position of the gripper. 
    home_gripper_position = np.asarray(vrep.simxGetObjectPosition(youbot.ptip, youbot.arm_ref, 
                                                                  simx_opmode_buffer))

    # Initialise the map.
    map_len = 10 # meters
    map_ppm = 8 # map_points_per_meters = 1
    array_side_len =  map_len * map_ppm
    youbot_pos = vrep.simxGetObjectPosition(youbot.ref, -1, simx_opmode_buffer)
    # Compute the offset between the starting position of the robot and the center of the map array 
    map_offset = np.array(youbot_pos[:2]) * map_ppm - np.asarray([2*array_side_len, 2*array_side_len])
    # 4*len+1 because we don't know where we start in the env, so the robot will start in the middle of the array
    map_arr = np.ones((4*array_side_len+1, 4*array_side_len+1), dtype="uint8") * 127
    cv2.imshow('RIP35', map_arr)

    # List of exploring movements
    moves = [('r', np.pi), ('f', 2, -5.25), ('r', np.pi/2), ('f', 2, 0), ('r', 0), ('f', 5, 0)]
    explo_count = 0

    # Initialise the state machine. 
    fsm = 'rotate'
    fsm = 'test-move-init'

    ## Start the demo. 
    while True:
        start_time = timer()
        print()
        print()
        print()
        
        if vrep.simxGetConnectionId() == -1:
            raise Exception('Lost connection to remote API.')

        # Get the position and the orientation of the robot. 
        youbot_pos = vrep.simxGetObjectPosition(youbot.ref, -1, simx_opmode_buffer)
        youbot_euler = vrep.simxGetObjectOrientation(youbot.ref, -1, simx_opmode_buffer)
        # The angle from euler is referenced from y axis so substracting pi/2 make it referenced from x axis
        youbot_angle = youbot_euler[2] - np.pi/2
        angle = -np.pi / 2

        ## Plot something if required. 
        if plot_data:
            # Read data from the depth sensor, more often called the Hokuyo (if you want to be more
            # precise about the way you control the sensor, see later for the details about this 
            # line or the file focused/youbot_3dpointcloud.m).
            # This function returns the set of points the Hokuyo saw in pts. contacts indicates, 
            # for each point, if it corresponds to an obstacle (the ray the Hokuyo sent was 
            # interrupted by an obstacle, and was not allowed to go to infinity without being 
            # stopped). 
            pts, contacts = youbot.hokuyo_read(vrep, simx_opmode_buffer)
            X_rel = [youbot.hokuyo1_pos[0], *pts[0, :], youbot.hokuyo2_pos[0]]
            Y_rel = [youbot.hokuyo1_pos[1], *pts[1, :], youbot.hokuyo2_pos[1]]
            def print_empty_space_on_map(pts, contacts, X_rel, Y_rel, map_arr):
                # Create a 2D mesh of points, stored in the vectors X and Y. This will be used to 
                # display 
                # the area the robot can see, by selecting the points within this mesh that are within 
                # the visibility range.
                mesh_time = timer()
                x, y = np.meshgrid(np.arange(-5, 5, 1.0/map_ppm), np.arange(-5, 5, 1.0/map_ppm))
                #print("Meshgrid time:", timer() - mesh_time, timer()-start_time)
                x, y = x.reshape(-1), y.reshape(-1)

                # Select the points in the mesh [X, Y] that are visible, as returned by the Hokuyo (it 
                # returns the area that is visible, but the visualisation draws a series of points 
                # that are within this visible area). 
 
                poly_time = timer()
                path = PolygonPath(list(zip(X_rel,Y_rel)))
                inside = path.contains_points(list(zip(x, y)))

                # Plot those points. Green dots: the visible area for the Hokuyo. Red starts: the 
                # obstacles. Red lines: the visibility range from the Hokuyo sensor. The youBot is 
                # indicated with two dots: the blue one corresponds to the rear, the red one to the 
                # Hokuyo sensor position. 
                # the points inside the laser range and that are not obstacles
                XY_inside = rel2abs(x[inside], y[inside], youbot_pos, youbot_euler[2])
                #sensor_ax.plot(*XY_inside, '.g')
                inside_timer = timer()
                for i in range(len(XY_inside[0])):
                    x = int((XY_inside[0][i]) * map_ppm - map_offset[0])
                    y = int((XY_inside[1][i]) * map_ppm - map_offset[1])
                    map_arr[x][y] = 255
                #print("Inside timer", timer() - inside_timer, start_time - timer())
            #process = multiprocessing.Process(target=print_empty_space_on_map, args=(pts, contacts, X_rel, Y_rel, map_arr))
            #process.start()
            print_empty_space_on_map(pts, contacts, X_rel, Y_rel, map_arr)
            sensor_ax.clear()
            sensor_ax.plot(*rel2abs(X_rel, Y_rel, youbot_pos, youbot_euler[2]), 'r')
            # The absolut position of the different contacts points 
            XY_contacts = rel2abs(pts[0, contacts], pts[1, contacts], youbot_pos, youbot_euler[2])
            contacts_timer = timer()
            for i in range(len(XY_contacts[0])):
                x = int((XY_contacts[0][i]) * map_ppm - map_offset[0])
                y = int((XY_contacts[1][i]) * map_ppm - map_offset[1])
                map_arr[x][y] = 0
            #print("Contacts timer", timer() - contacts_timer)
            #np.set_printoptions(threshold=np.nan)
            sensor_ax.plot(*XY_contacts, '*m')

            # The absolute position of the bot
            sensor_ax.plot(*rel2abs(0, 0, youbot_pos, youbot_euler[2]), 'ob')
            # The absolute position of the bot's sensors
            sensor_ax.plot(*rel2abs(*youbot.hokuyo1_pos[:2], youbot_pos, youbot_euler[2]), 'or')
            sensor_ax.plot(*rel2abs(*youbot.hokuyo2_pos[:2], youbot_pos, youbot_euler[2]), 'or')
            sensor_ax.set_xlim(-10.5, 10.5)
            sensor_ax.set_ylim(-10.5, 10.5)
            sensor_ax.set_aspect('equal')
            canvas.flush_events()
        
        # Test the exploration by using a hardcoded path 
        if fsm == 'test-move-init':
            #youbot0 = youbot_pos[0]
            #euler2 = youbot_euler[2]
            fsm = 'test-explo'

        if fsm == 'test-explo':
            cv2.imshow('RIP35', map_arr)
            cv2.waitKey(10)
            
            move = moves[explo_count]
            print('move is', move)
            move_type = move[0]
            if move_type is 'r':
                move_val = move[1]
                rotate_right_vel = angdiff(move_val, youbot_angle)
                if (abs(angdiff(move_val, youbot_angle)) < np.deg2rad(.1)) and \
                        (abs(angdiff(prev_orientation, youbot_angle)) < np.deg2rad(.01)):
                    rotate_right_vel = 0
                    explo_count += 1
                    fsm = 'test-move-init'
                prev_orientation = youbot_angle
            elif move_type is 'f':
                print("Youbot pos", youbot_pos)
                move_x = move[1]
                move_y = move[2]
                delta_x = youbot_pos[0] - move_x
                delta_y = youbot_pos[1] - move_y
                x_rel = delta_x*np.cos(youbot_angle) + delta_y*np.sin(youbot_angle)
                y_rel = -delta_x*np.sin(youbot_angle) + delta_y*np.cos(youbot_angle)
                den_norm = np.sqrt(x_rel**2+y_rel**2)
                x_rel_norm = x_rel / den_norm
                y_rel_norm = y_rel / den_norm
                theta_target = np.sign(y_rel_norm) * np.arccos(x_rel_norm)
                
                print("delta pos:", delta_x, delta_y)
                print("Bot angle:", youbot_angle)
                print("pos rel:\t", x_rel, y_rel)
                print("pos rel norm:\t", x_rel_norm, y_rel_norm)
                print("thetas", theta_target)
                
                print(delta_x, delta_y, x_rel, youbot_angle)
                print(np.cos(youbot_angle), np.sin(youbot_angle))
                """
                tan_theta = delta_x/delta_y
                theta = np.arctan(tan_theta)
                print('theta', theta, youbot_euler[2])
                delta_theta = (theta - youbot_euler[2])%(2*np.pi)
                print("delta theta", delta_theta)
                if delta_theta < np.pi/2 or delta_theta > 3*np.pi/2:
                    forw_sign = 1
                else:
                    forw_sign = -1
                """
                forw_sign = np.sign(x_rel)

                print(forw_sign)
                dist_from_target = np.sqrt((youbot_pos[0] - move_x)**2 + (youbot_pos[1] - move_y)**2)
                forw_back_vel = forw_sign * (dist_from_target)
                #print(forw_back_vel, youbot_pos[0])
                if (dist_from_target < .1) and (abs(youbot_pos[0] - prev_position) < .001):
                    forw_back_vel = 0
                    explo_count += 1
                    fsm = 'test-move-init'
                prev_position = youbot_pos[0]
            if explo_count >= len(moves):
                while True:
                    pass
        ## Apply the state machine. 
        elif fsm == 'rotate':
            cv2.imshow('RIP35', map_arr)
            cv2.waitKey(10)
            ## First, rotate the robot to go to one table.             
            # The rotation velocity depends on the difference between the current angle and the 
            # target. 
            rotate_right_vel = angdiff(angle, youbot_euler[2])

            # When the rotation is done (with a sufficiently high precision), move on to the 
            # next state. 
            if (abs(angdiff(angle, youbot_euler[2])) < np.deg2rad(.1)) and \
                    (abs(angdiff(prev_orientation, youbot_euler[2])) < np.deg2rad(.01)):
                rotate_right_vel = 0
                fsm = 'drive'
                print("Entering 'Drive' State")

            prev_orientation = youbot_euler[2]
        elif fsm == 'drive':
            cv2.imshow('RIP35', map_arr)
            cv2.waitKey(10)
            ## Then, make it move straight ahead until it reaches the table (x = 3.167 m).
            # The further the robot, the faster it drives. (Only check for the first dimension.)
            # For the project, you should not use a predefined value, but rather compute it from 
            # your map.
            forw_back_vel = - (youbot_pos[0] + 3.167)

            # If the robot is sufficiently close and its speed is sufficiently low, stop it and 
            # move its arm to a specific location before moving on to the next state.
            if (youbot_pos[0] + 3.167 < .001) and (abs(youbot_pos[0] - prev_position) < .001):
                forw_back_vel = 0

                # Change the orientation of the camera to focus on the table (preparation for 
                # the next state).
                vrep.simxSetObjectOrientation(youbot.rgbd_casing, youbot.ref, [0, 0, np.pi / 4], 
                                              simx_opmode_oneshot)

                # Move the arm to the preset pose pickupJoints (only useful for this demo you 
                # should compute it based on the object to grasp).
                for arm_joint, pickup_joint in zip(youbot.arm_joints, pickup_joints):
                    vrep.simxSetJointTargetPosition(arm_joint, pickup_joint, simx_opmode_oneshot)

                fsm = 'snapshot'
            prev_position = youbot_pos[0]
        elif fsm == 'snapshot':
            ## Read data from the depth camera (Hokuyo)
            # Reading a 3D image costs a lot to VREP (it has to simulate the image). It also 
            # requires a lot of bandwidth, and processing a 3D point cloud (for instance, 
            # to find one of the boxes or cylinders that the robot has to grasp) will take a long
            # time in MATLAB. In general, you will only want to capture a 3D image at specific 
            # times, for instance when you believe you're facing one of the tables.

            # Reduce the view angle to pi/8 in order to better see the objects. Do it only once.
            # ^^^^^^     ^^^^^^^^^^    ^^^^                                     ^^^^^^^^^^^^^^^
            # simxSetFloatSignal                                                simx_opmode_oneshot_wait
            #            |
            #            rgbd_sensor_scan_angle
            # The depth camera has a limited number of rays that gather information. If this number 
            # is concentrated on a smaller angle, the resolution is better. pi/8 has been 
            # determined by experimentation.
            vrep.simxSetFloatSignal('rgbd_sensor_scan_angle', np.pi / 8, simx_opmode_oneshot_wait)

            # Ask the sensor to turn itself on, take A SINGLE POINT CLOUD, and turn itself off again. 
            # ^^^     ^^^^^^                ^^       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # simxSetIntegerSignal          1        simx_opmode_oneshot_wait
            #         |
            #         handle_xyz_sensor
            vrep.simxSetIntegerSignal('handle_xyz_sensor', 1, simx_opmode_oneshot_wait)

            # Then retrieve the last point cloud the depth sensor took.
            # If you were to try to capture multiple images in a row, try other values than 
            # vrep.simx_opmode_oneshot_wait. 
            print('Capturing a point cloud...')
            pts = youbot.xyz_read(vrep, simx_opmode_oneshot_wait)
            # Each column of pts has [xyzdistancetosensor]. However, plot3 does not have the same 
            # frame of reference as the output data. To get a correct plot, you should invert the
            # y and z dimensions. 

            # Here, we only keep points within 1 meter, to focus on the table. 
            pts = pts[:3, pts[3, :] < 1]
            if plot_data:
                pt_cloud_ax.scatter(pts[0, :], pts[2, :], pts[1, :], '*')
                pt_cloud_ax.set_aspect('equal')
                canvas.flush_events()
                # view([-169 -46])  -> can't remember what's the plt equivalent for this

            # Save the point cloud to pc.xyz. (This file can be displayed with 
            # http://www.meshlab.net/.)
            with open('pc.xyz','w') as f:
                for pt in pts.T:
                    f.write("%f %f %f\n" % tuple(pt))
            print('Read %d 3D points, saved to pc.xyz.' % pts.shape[1])

            ## Read data from the RGB camera
            # This starts the robot's camera to take a 2D picture of what the robot can see. 
            # Reading an image costs a lot to VREP (it has to simulate the image). It also requires 
            # a lot of bandwidth, and processing an image will take a long time in MATLAB. In 
            # general, you will only want to capture an image at specific times, for instance 
            # when you believe you're facing one of the tables or a basket.

            # Ask the sensor to turn itself on, take A SINGLE IMAGE, and turn itself off again. 
            # ^^^     ^^^^^^                ^^       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            # simxSetIntegerSignal          1        simx_opmode_oneshot_wait
            #         |
            #         handle_rgb_sensor
            vrep.simxSetIntegerSignal('handle_rgb_sensor', 1, simx_opmode_oneshot_wait)

            # Then retrieve the last picture the camera took. The image must be in RGB 
            # (not grayscale). 
            #     ^^^^^^^^^^^^^^^^^^^^^^^^^     ^^^^^^                            ^^^
            #     simxGetVisionSensorImage2     h.rgbSensor                       0
            # If you were to try to capture multiple images in a row, try other values than 
            # simx_opmode_oneshot_wait. 
            print('Capturing image...')
            image = vrep.simxGetVisionSensorImage(youbot.rgb_sensor, 0, simx_opmode_oneshot_wait)
            print("Captured %dx%dx%dx image." % image.shape)

            # Finally, show the image. 
            if plot_data:
                camera_ax.imshow(image, origin='lower')
                canvas.flush_events()

            # Next state. 
            fsm = 'extend'
            print("Entering 'Extend' State")

        elif fsm == 'extend':
            ## Move the arm to face the object.
            # Get the arm position.
            tpos = np.asarray(vrep.simxGetObjectPosition(youbot.ptip, youbot.arm_ref, simx_opmode_buffer))
            # If the arm has reached the wanted position, move on to the next state.
            # Once again, your code should compute this based on the object to grasp.
            if np.linalg.norm(tpos - np.asarray([0.3259, -0.0010, 0.2951])) < .002:
                # Set the inverse kinematics (IK) mode to position AND orientation.
                vrep.simxSetIntegerSignal('km_mode', 2, simx_opmode_oneshot_wait)
                fsm = 'reachout'
                print("Entering 'Reachout' State")
        elif fsm == 'reachout':
            ## Move the gripper tip along a line so that it faces the object with the right angle.
            # Get the arm tip position. The arm is driven only by the position of the tip, not by 
            # the angles of the joints, except if IK is disabled.
            # Following the line ensures the arm attacks the object with the right angle.
            tpos = vrep.simxGetObjectPosition(youbot.ptip, youbot.arm_ref, simx_opmode_buffer)

            # If the tip is at the right position, go on to the next state. Again, this value 
            # should be computed based on the object to grasp and on the robot's position.
            if tpos[0] > .39:
                fsm = 'grasp'
                print("Entering 'Grasp' State")

            # Move the tip to the next position along the line.
            tpos[0] = tpos[0] + .01
            vrep.simxSetObjectPosition(youbot.ptarget, youbot.arm_ref, tpos, simx_opmode_oneshot)
        elif fsm == 'grasp':
            ## Grasp the object by closing the gripper on it.
            # Close the gripper. Please pay attention that it is not possible to adjust the force 
            # to apply: the object will sometimes slip from the gripper!
            vrep.simxSetIntegerSignal('gripper_open', 0, simx_opmode_oneshot_wait)

            # Make the program wait for the gripper to be closed. This value was determined by 
            # experiments.
            sleep(2)

            # Disable IK this is used at the next state to move the joints manually.
            vrep.simxSetIntegerSignal('km_mode', 0, simx_opmode_oneshot_wait)
            fsm = 'backoff'
            print("Entering 'Backoff' State")
        elif fsm == 'backoff':
            ## Go back to rest position.
            # Set each joint to their original angle, as given by startingJoints. Please note that 
            # this operation is not instantaneous, it takes a few iterations of the code for the 
            # arm to reach the requested pose.
            for arm_joint, starting_joint in zip(youbot.arm_joints, starting_joints):
                vrep.simxSetJointTargetPosition(arm_joint, starting_joint, simx_opmode_oneshot)

            # Get the gripper position and check whether it is at destination (the original 
            # position).
            tpos = np.asarray(vrep.simxGetObjectPosition(youbot.ptip, youbot.arm_ref, 
                                                         simx_opmode_buffer))
            if np.linalg.norm(tpos - home_gripper_position) < .02:
                # Open the gripper when the arm is above its base.
                vrep.simxSetIntegerSignal('gripper_open', 1, simx_opmode_oneshot_wait)

            if np.linalg.norm(tpos - home_gripper_position) < .002:
                fsm = 'finished'
                print("Entering 'Finished' State")

        elif fsm == 'finished':
            ## Demo done: exit the function.
            sleep(3)
            break
        else:
            raise Exception('Unknown state %s' % fsm)

        # Update wheel velocities using the global values (whatever the state is). 
        youbot.drive(vrep, forw_back_vel, right_vel, rotate_right_vel)
        # Make sure that we do not go faster than the physics simulation (each iteration must take 
        # roughly 50 ms).
        elapsed = timer() - start_time
        timeleft = timestep - elapsed
        print("Elapsed time;", elapsed)
        print("Time left;", timeleft)
        if timeleft > 0:
            # Note: use plt.pause when running an interactive plot, otherwise time.sleep() is fine.
            #plt.pause(min(timeleft, .01))   
            plt.pause(timeleft)   

