#!/usr/bin/python3
from pytocl.driver import Driver
from pytocl.car import State, Command
import torch
from torch.autograd import Variable
from torch_model import TwoLayerNet
import numpy as np
import math
import argparse
import os

# Boolean indicating if we use the driver for training or swarm
# If both False, we use a neural net, either given by initialisation or by
# loading our model for the none-blocking car.
TRAIN = False
SWARM = False

class MyDriver(Driver):
    '''
    MyDriver extends Driver with our driver.
    '''
    def __init__(self, model=None, logdata=False):
        Driver.__init__(self, logdata)
        parser = argparse.ArgumentParser(
            description='Client for TORCS racing car simulation with SCR network'
                        ' server.'
        )
        parser.add_argument(
            '-p',
            '--port',
            help='Port to connect, 3001 - 3010 for clients 1 - 10.',
            type=int,
            default=3001
        )
        args = parser.parse_args()
        # Driver port used as id
        self.port = args.port
        self.score = 0

        if (TRAIN):
            # Set output file to save sensor data to
            # TODO: filename from user input
            self.output_file = open('./data/aalborg.csv','a')
        elif (model):
            self.model = model
        elif (SWARM):
            # Load "winner" model
            self.model_winner = TwoLayerNet(22, 15, 2, False)
            self.model_winner.load_state_dict(torch.load("./models/model_without_blocking_22.pt", map_location=lambda storage, loc: storage))

            # Load "blocker" model
            self.model_helper = TwoLayerNet(59, 30, 2, False)
            self.model_helper.load_state_dict(torch.load("./models/model_with_blocking.pt", map_location=lambda storage, loc: storage))

            # Position files
            self.own_pos_filename = "./positions/pos" + str(self.port)
            # Write test line
            own_pos_file = open(self.own_pos_filename, "w")
            own_pos_file.write("100 \n")
            own_pos_file.close()
            self.helper_pos_filename = "./positions/pos"

            # Boolean indicating game is gonna start
            self.start = True
        else:
            self.model = TwoLayerNet(22, 15, 2, False)
            self.model.load_state_dict(torch.load("./models/model_without_blocking_22.pt", map_location=lambda storage, loc: storage))


    def drive(self, carstate):
        """
        Produces driving command in response to newly received car state.

        If we are training, see trainDrive.
        If we are using swarm:
        Depending on the position of our own car and our teammate,
        determine if we are a winner or a helper and use that model
        to predict the output based on the car state as input.

        If we are just using a network:
        Use the model to predict the output based on the car state as input.
        """
        if (TRAIN):
            command = self.trainDrive(carstate)
        elif (SWARM):
            # If first run, set helper file name
            if (self.start):
                for f in os.listdir("./positions/"):
                    if str(self.port) not in f:
                        self.helper_pos_filename = "./positions/" + f
                        break
                self.start = False

            # Write own position to file
            own_pos_file = open(self.own_pos_filename, "w")
            own_pos_file.write(str(carstate.race_position) + " \n")
            own_pos_file.close()

            # Always assume winner strategy
            model = self.model_winner
            np_car = self.carstateToNumpy(carstate, "winner")

            # Check if teammate is ahead of behind
            helper_pos_file = open(self.helper_pos_filename, "r")
            line = helper_pos_file.read()
            helper_pos_file.close()
            # If teammate is ahead with at least 3 positions, become a helper
            if (line and int(line) < carstate.race_position - 3):
                model = self.model_helper
                np_car = self.carstateToNumpy(carstate, "helper")

            command = self.modelDrive(model, carstate, np_car)
        else:
            np_car = self.carstateToNumpy(carstate, "winner")
            command = self.modelDrive(self.model, carstate, np_car)


        # Set logging data
        if self.data_logger:
            self.data_logger.log(carstate, command)
        return command

    def trainDrive(self, carstate):
        '''
        Drive method used when recording trainingdata.
        Use the opponent sensors to determine where the car needs to drive
        on the road and the edge sensors to adjust its speed.
        '''
        command = Command()
        # Convert carstate to numpy car
        np_car = self.carstateToNumpy(carstate, "helper")

        # The back opponent sensors
        left_1_opponent_sensors = carstate.opponents[:3]
        left_2_opponent_sensors = carstate.opponents[3:6]
        left_3_opponent_sensors = carstate.opponents[6:9]
        right_1_opponent_sensors = carstate.opponents[33:36]
        right_2_opponent_sensors = carstate.opponents[30:33]
        right_3_opponent_sensors = carstate.opponents[27:30]

        # Assume we are trying to drive in the center of the track
        target_track_pos = 0.0
        closest_opponent = 200
        # Set track position based on the opponents behind us
        for left_opponent in left_1_opponent_sensors:
            if left_opponent < 70 and left_opponent < closest_opponent:
                target_track_pos = 0.5
                closest_opponent = left_opponent
        for left_opponent in left_2_opponent_sensors:
            if left_opponent < 70 and left_opponent < closest_opponent:
                target_track_pos = 0.7
                closest_opponent = left_opponent
        for left_opponent in left_3_opponent_sensors:
            if left_opponent < 100 and left_opponent < closest_opponent:
                target_track_pos = 0.7
                closest_opponent = left_opponent
        for right_opponent in right_1_opponent_sensors:
            if right_opponent < 50 and right_opponent < closest_opponent:
                target_track_pos = -0.5
                closest_opponent = right_opponent
        for right_opponent in right_2_opponent_sensors:
            if right_opponent < 70 and right_opponent < closest_opponent:
                target_track_pos = -0.7
                closest_opponent = right_opponent
        for right_opponent in right_3_opponent_sensors:
            if right_opponent < 100 and right_opponent < closest_opponent:
                target_track_pos = -0.7
                closest_opponent = right_opponent

        # Set steering command using the target_track_pos
        self.steer(carstate, target_track_pos, command)

        # The five front edge sensors used to see if the drivers is approaching
        # an edge
        front_sensors = carstate.distances_from_edge[7:12]
        v_x = 0
        # Set speed based on edge sensors in front of the car
        if (max(front_sensors) < 70):
            if (carstate.speed_x * 3.6 > 90):
                command.brake = 0.8
            else:
                v_x = 60
        elif (max(front_sensors) < 100):
            if (carstate.speed_x * 3.6 > 100):
                command.brake = 0.8
            else:
                v_x = 80
        elif (max(front_sensors) < 150):
            v_x = 150
        elif (max(front_sensors) <= 200):
            v_x = 170

        # Set acceleration based on target speed
        self.accelerate(carstate, v_x, command)

        # Write carstate and output command to a file
        self.writeToFile(np_car, command, v_x)
        return command

    def modelDrive(self, model, carstate, np_car):
        '''
        Construct driving command based on a model.
        '''
        # Get output from model
        x = torch.from_numpy(np_car)
        y = model(Variable(x.float())).data.numpy()
        # Prevent from driving really slow
        if (y[1] * 306 < 10): y[1] = 40 / 306

        # Set command
        command = Command()
        command.steering = y[0]
        self.accelerate(carstate, y[1] * 306, command)

        # Compute score based on current lap time
        if (self.score > 10000):
            self.score = carstate.current_lap_time + carstate.damage / 100
        else:
            self.score = max(self.score, carstate.current_lap_time + carstate.damage / 100)
        if (carstate.distances_from_edge[0] == -1 and self.score < 10000): self.score += 10000

        return command

    def carstateToNumpy(self, carstate, model):
        '''
        Convert the carstate to a numpy array with the data used by the model.
        '''
        # If winner model, the input is only 22
        if (model == "winner"):
            car = np.ones(22)*-3
        else:
            car = np.ones(59)*-3
        # Add sensors and normalise
        car[0] = carstate.speed_x / 85 # +- 300km/h as max
        car[1] = carstate.distance_from_center
        car[2] = carstate.angle / math.pi

        for i in range(0, 19):
            car[i+3] = carstate.distances_from_edge[i] / 200

        # If helper model, add opponents
        if (model == "helper"):
            for i in range(len(carstate.opponents)):
                car[i+22] = carstate.opponents[i] / 200
            car[58] = carstate.race_position

        return car

    def writeToFile(self, car, command, speed):
        '''
        Write sensor and output data to a file.
        car: the numpy car state.
        command: the command executed by the car.
        speed: the target speed of the car.
        '''
        output = np.zeros(64)
        output[:59] = car
        output[60] = command.accelerator
        output[61] = command.brake
        output[62] = command.steering
        output[63] = speed / 306 # normalise using max speed (306km/h = 85m/s)
        if car[3] != -1:
            self.output_file.write(", ".join(map(str, output.tolist())))
            self.output_file.write("\n")


    def on_shutdown(self):
        '''
        On shutdown of the driver, call the shutdown of Driver and remove
        positions files.
        '''
        Driver.on_shutdown(self)
        if (SWARM):
            os.remove(self.own_pos_filename, dir_fd=None)
