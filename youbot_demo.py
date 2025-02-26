from vrep import VRep
from vrep.const import *
from youbot import YouBot
from time import sleep
import numpy as np
import atexit
import matplotlib.pyplot as plt
from matplotlib.path import Path as PolygonPath
from time import perf_counter as timer
from youbot.transforms import angdiff

# Illustrates the V-REP bindings.

# (C) Copyright Renaud Detry 2013, Thibaut Cuvelier 2017, Mathieu Baijot 2017.
# Distributed under the GNU General Public License.
# (See http://www.gnu.org/copyleft/gpl.html)


# Open house.ttt in vrep
# Run simRemoteApi.start(19998) in the vrep LUA console
# Run this script

if __name__ == '__main__':
    ## Initiate the connection to the simulator. 
    print('Program started')
    VRep.simxFinish(-1)
    vrep = VRep('127.0.0.1', 19998, True, True, 2000, 5)
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
    timestep = .05

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
        canvas = plt.gcf().canvas
        plt.draw()

    # Make sure everything is settled before we start. (is this necessary?)
    # sleep(2)

    # Retrieve the position of the gripper. 
    home_gripper_position = np.asarray(vrep.simxGetObjectPosition(youbot.ptip, youbot.arm_ref, simx_opmode_buffer))

    # Initialise the state machine. 
    fsm = 'rotate'

    ## Start the demo. 
    while True:
        start_time = timer()
        
        if vrep.simxGetConnectionId() == -1:
            raise Exception('Lost connection to remote API.')

        # Get the position and the orientation of the robot. 
        youbot_pos = vrep.simxGetObjectPosition(youbot.ref, -1, simx_opmode_buffer)
        youbot_euler = vrep.simxGetObjectOrientation(youbot.ref, -1, simx_opmode_buffer)
        angle = -np.pi / 2

        ## Plot something if required. 
        if plot_data:
            # Read data from the depth sensor, more often called the Hokuyo (if you want to be more
            # precise about the way you control the sensor, see later for the details about this 
            # line or the file focused/youbot_3dpointcloud.m).
            # This def returns the set of points the Hokuyo saw in pts. contacts indicates, 
            # for each point, if it corresponds to an obstacle (the ray the Hokuyo sent was 
            # interrupted by an obstacle, and was not allowed to go to infinity without being 
            # stopped). 
            pts, contacts = youbot.hokuyo_read(vrep, simx_opmode_buffer)

            # Create a 2D mesh of points, stored in the vectors X and Y. This will be used to 
            # display 
            # the area the robot can see, by selecting the points within this mesh that are within 
            # the visibility range.
            x, y = np.meshgrid(np.arange(-5, 5, .25), np.arange(-5, 5, .25))
            x, y = x.reshape(-1), y.reshape(-1)

            # Select the points in the mesh [X, Y] that are visible, as returned by the Hokuyo (it 
            # returns the area that is visible, but the visualisation draws a series of points 
            # that are within this visible area). 
            path = PolygonPath(list(zip(
                [youbot.hokuyo1_pos[0], *pts[0, :], youbot.hokuyo2_pos[0]],
                [youbot.hokuyo1_pos[1], *pts[1, :], youbot.hokuyo2_pos[1]]
            )))
            inside = path.contains_points(list(zip(x, y)))

            # Plot those points. Green dots: the visible area for the Hokuyo. Red starts: the 
            # obstacles. Red lines: the visibility range from the Hokuyo sensor. The youBot is 
            # indicated with two dots: the blue one corresponds to the rear, the red one to the 
            # Hokuyo sensor position. 
            sensor_ax.clear()
            sensor_ax.plot(x[inside], y[inside], '.g')
            sensor_ax.plot([youbot.hokuyo1_pos[0], *pts[0, :], youbot.hokuyo2_pos[0]],
                           [youbot.hokuyo1_pos[1], *pts[1, :], youbot.hokuyo2_pos[1]], 'r')
            sensor_ax.plot(pts[0, contacts], pts[1, contacts], '*m')
            sensor_ax.plot(0, 0, 'ob')
            sensor_ax.plot(*youbot.hokuyo1_pos[:2], 'or')
            sensor_ax.plot(*youbot.hokuyo2_pos[:2], 'or')
            sensor_ax.set_xlim(-5.5, 5.5)
            sensor_ax.set_ylim(-5.5, 2.5)
            sensor_ax.set_aspect('equal')
            canvas.flush_events()
        
        ## Apply the state machine. 
        if fsm == 'rotate':
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
            ## Then, make it move straight ahead until it reaches the table (x = 3.167 m).
            # The further the robot, the faster it drives. (Only check for the first dimension.)
            # For the project, you should not use a predefined value, but rather compute it from your map.
            forw_back_vel = - (youbot_pos[0] + 3.167)

            # If the robot is sufficiently close and its speed is sufficiently low, stop it and move its arm to
            # a specific location before moving on to the next state.
            if (youbot_pos[0] + 3.167 < .001) and (abs(youbot_pos[0] - prev_position) < .001):
                forwBackVel = 0

                # Change the orientation of the camera to focus on the table (preparation for next state).
                vrep.simxSetObjectOrientation(youbot.rgbd_casing, youbot.ref, [0, 0, np.pi / 4], simx_opmode_oneshot)

                # Move the arm to the preset pose pickupJoints (only useful for this demo you should compute it based
                # on the object to grasp).
                for i in range(5):
                    res = vrep.simxSetJointTargetPosition(youbot.arm_joints[i], pickup_joints[i], simx_opmode_oneshot)
    #                 vrchk(vrep, res, True)

                fsm = 'snapshot'
            prev_position = youbot_pos[0]
        elif fsm == 'snapshot':
            ## Read data from the depth camera (Hokuyo)
            # Reading a 3D image costs a lot to VREP (it has to simulate the image). It also requires a lot of
            # bandwidth, and processing a 3D point cloud (for instance, to find one of the boxes or cylinders that
            # the robot has to grasp) will take a long time in MATLAB. In general, you will only want to capture a 3D
            # image at specific times, for instance when you believe you're facing one of the tables.

            # Reduce the view angle to pi/8 in order to better see the objects. Do it only once.
            # ^^^^^^     ^^^^^^^^^^    ^^^^                                     ^^^^^^^^^^^^^^^
            # simxSetFloatSignal                                                simx_opmode_oneshot_wait
            #            |
            #            rgbd_sensor_scan_angle
            # The depth camera has a limited number of rays that gather information. If this number is concentrated
            # on a smaller angle, the resolution is better. pi/8 has been determined by experimentation.
    #         res = vrep.simxSetFloatSignal(id, 'rgbd_sensor_scan_angle', pi / 8, vrep.simx_opmode_oneshot_wait)
    #         vrchk(vrep, res)
    # 
    #         # Ask the sensor to turn itself on, take A SINGLE POINT CLOUD, and turn itself off again. 
    #         # ^^^     ^^^^^^                ^^       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #         # simxSetIntegerSignal          1        simx_opmode_oneshot_wait
    #         #         |
    #         #         handle_xyz_sensor
    #         res = vrep.simxSetIntegerSignal(id, 'handle_xyz_sensor', 1, vrep.simx_opmode_oneshot_wait)
    #         vrchk(vrep, res)
    # 
    #         # Then retrieve the last point cloud the depth sensor took.
    #         # If you were to try to capture multiple images in a row, try other values than 
    #         # vrep.simx_opmode_oneshot_wait. 
    #         fprintf('Capturing a point cloud...\n')
    #         pts = youbot_xyz_sensor(vrep, handles, vrep.simx_opmode_oneshot_wait)
    #         # Each column of pts has [xyzdistancetosensor]. However, plot3 does not have the same frame of reference as 
    #         # the output data. To get a correct plot, you should invert the y and z dimensions. 
    # 
    #         # Here, we only keep points within 1 meter, to focus on the table. 
    #         pts = pts(1:3, pts(4, :) < 1)
    # 
    #         if plotData:
    #             subplot(223)
    #             plot3(pts(1, :), pts(3, :), pts(2, :), '*')
    #             axis equal
    #             view([-169 -46])
    # 
    #         # Save the point cloud to pc.xyz. (This file can be displayed with http://www.meshlab.net/.)
    #         fileID = fopen('pc.xyz','w')
    #         fprintf(fileID,'#f #f #f\n', pts)
    #         fclose(fileID)
    #         fprintf('Read #i 3D points, saved to pc.xyz.\n', max(size(pts)))
    # 
    #         ## Read data from the RGB camera
    #         # This starts the robot's camera to take a 2D picture of what the robot can see. 
    #         # Reading an image costs a lot to VREP (it has to simulate the image). It also requires a lot of bandwidth, 
    #         # and processing an image will take a long time in MATLAB. In general, you will only want to capture 
    #         # an image at specific times, for instance when you believe you're facing one of the tables or a basket.
    # 
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

            #Finally, show the image. 
            if plot_data:
                image = image[::-1, :, :]
                camera_ax.imshow(image)
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
                res = vrep.simxSetIntegerSignal('km_mode', 2, simx_opmode_oneshot_wait)
    #             vrchk(vrep, res, True)
                fsm = 'reachout'
                print("Entering 'Reachout' State")
        elif fsm == 'reachout':
            ## Move the gripper tip along a line so that it faces the object with the right angle.
            # Get the arm tip position. The arm is driven only by the position of the tip, not by the angles of
            # the joints, except if IK is disabled.
            # Following the line ensures the arm attacks the object with the right angle.
            tpos = vrep.simxGetObjectPosition(youbot.ptip, youbot.arm_ref, simx_opmode_buffer)

            # If the tip is at the right position, go on to the next state. Again, this value should be computed based
            # on the object to grasp and on the robot's position.
            if tpos[0] > .39:
                fsm = 'grasp'
                print("Entering 'Grasp' State")

            # Move the tip to the next position along the line.
            tpos[0] = tpos[0] + .01
            res = vrep.simxSetObjectPosition(youbot.ptarget, youbot.arm_ref, tpos, simx_opmode_oneshot)
    #         vrchk(vrep, res, True)
        elif fsm == 'grasp':
            ## Grasp the object by closing the gripper on it.
            # Close the gripper. Please pay attention that it is not possible to adjust the force to apply:
            # the object will sometimes slip from the gripper!
            res = vrep.simxSetIntegerSignal('gripper_open', 0, simx_opmode_oneshot_wait)
    #         vrchk(vrep, res)

            # Make the program wait for the gripper to be closed. This value was determined by experiments.
            sleep(2)

            # Disable IK this is used at the next state to move the joints manually.
            res = vrep.simxSetIntegerSignal('km_mode', 0, simx_opmode_oneshot_wait)
    #         vrchk(vrep, res)
            fsm = 'backoff'
            print("Entering 'Backoff' State")
        elif fsm == 'backoff':
            ## Go back to rest position.
            # Set each joint to their original angle, as given by startingJoints. Please note that this operation is not
            # instantaneous, it takes a few iterations of the code for the arm to reach the requested pose.
            for i in range(5):
                res = vrep.simxSetJointTargetPosition(youbot.arm_joints[i], starting_joints[i], simx_opmode_oneshot)
    #             vrchk(vrep, res, True)

            # Get the gripper position and check whether it is at destination (the original position).
            tpos = np.asarray(vrep.simxGetObjectPosition(youbot.ptip, youbot.arm_ref, simx_opmode_buffer))
    #         vrchk(vrep, res, True)
            if np.linalg.norm(tpos - home_gripper_position) < .02:
                # Open the gripper when the arm is above its base.
                res = vrep.simxSetIntegerSignal('gripper_open', 1, simx_opmode_oneshot_wait)
    #             vrchk(vrep, res)

            if np.linalg.norm(tpos - home_gripper_position) < .002:
                fsm = 'finished'
                print("Entering 'Finished' State")

        elif fsm == 'finished':
            ## Demo done: exit the function.
            sleep(3)
            break
        else:
            raise Exception(f'Unknown state {fsm}.')

        # Update wheel velocities using the global values (whatever the state is). 
        youbot.drive(vrep, forw_back_vel, right_vel, rotate_right_vel)
        # Make sure that we do not go faster than the physics simulation (each iteration must take 
        # roughly 50 ms).
        elapsed = timer() - start_time
        timeleft = timestep - elapsed
        if timeleft > 0:
            # Note: use plt.pause when running an interactive plot, otherwise time.sleep() is fine.
            plt.pause(min(timeleft, .01))   
