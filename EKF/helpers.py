from datapoint import DataPoint
from fusionekf import FusionEKF


def parse_data(file_path):
    """
      Args:
      file_path
         radar has three measurements (rho, phi, rhodot), lidar has two measurements (x, y).



          For a row containing radar data, the columns are:
          sensor_type, rho_measured, phi_measured, rhodot_measured, timestamp, x_groundtruth, y_groundtruth, vx_groundtruth, vy_groundtruth

          For a row containing lidar data, the columns are:
          sensor_type, x_measured, y_measured, timestamp, x_groundtruth, y_groundtruth, vx_groundtruth, vy_groundtruth


    """

    all_sensor_data = []
    all_ground_truths = []

    with open(file_path) as f:

        for line in f:
            data = line.split()

            if data[0] == 'L':

                sensor_data = DataPoint({
                    'timestamp': int(data[3]),
                    'name': 'lidar',
                    'x': float(data[1]),
                    'y': float(data[2])
                })

                g = {'timestamp': int(data[3]),
                     'name': 'state',
                     'x': float(data[4]),
                     'y': float(data[5]),
                     'vx': float(data[6]),
                     'vy': float(data[7])
                     }

                ground_truth = DataPoint(g)

            elif data[0] == 'R':

                sensor_data = DataPoint({
                    'timestamp': int(data[4]),
                    'name': 'radar',
                    'rho': float(data[1]),
                    'phi': float(data[2]),
                    'drho': float(data[3])
                })

                g = {'timestamp': int(data[4]),
                     'name': 'state',
                     'x': float(data[5]),
                     'y': float(data[6]),
                     'vx': float(data[7]),
                     'vy': float(data[8])
                     }
                ground_truth = DataPoint(g)

            all_sensor_data.append(sensor_data)
            all_ground_truths.append(ground_truth)

    return all_sensor_data, all_ground_truths


def get_state_estimations(EKF, all_sensor_data):
    """
    Calculates all state estimations given a FusionEKF instance() and sensor measurements

    Args:
      EKF - an instance of a FusionEKF() class
      all_sensor_data - a list of sensor measurements as a DataPoint() instance

    Returns:
      all_state_estimations
        - a list of all state estimations as predicted by the EKF instance
        - each state estimation is wrapped in  DataPoint() instance
    """

    all_state_estimations = []

    for data in all_sensor_data:
        EKF.process(data)

        x = EKF.get()
        px, py, vx, vy = x[0, 0], x[1, 0], x[2, 0], x[3, 0]

        g = {'timestamp': data.get_timestamp(),
             'name': 'state',
             'x': px,
             'y': py,
             'vx': vx,
             'vy': vy}

        state_estimation = DataPoint(g)
        all_state_estimations.append(state_estimation)

    return all_state_estimations
