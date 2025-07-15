"""
Run autonomous exploration and occupancy mapping in Carla Sim.

This script uses Carla's autodrive functionality to navigate a vehicle
through the environment while building an occupancy map, similar to
the structure of run_vlm_exp.py but adapted for Carla.
"""

import os
import time
import logging
import carla
import numpy as np
import pickle
import signal
import sys
import carla
import csv
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import json
from omegaconf import OmegaConf

from src.carla_interface import CarlaSimulator
# from src.carla_tsdf import CarlaTSDFPlanner

# --- Constants and Configurations ---
SENSOR_CONFIGS = {
    'rgb': {
        'blueprint': 'sensor.camera.rgb',
        'attributes': {'image_size_x': '640', 'image_size_y': '480', 'fov': '90'},
        'location': {'x': 2.0, 'y': 0.0, 'z': 1.5},
        'rotation': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
    },
    'depth': {
        'blueprint': 'sensor.camera.depth',
        'attributes': {'image_size_x': '640', 'image_size_y': '480', 'fov': '90'},
        'location': {'x': 2.0, 'y': 0.0, 'z': 1.5},
        'rotation': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
    },
    # 'lidar': {
    #     'blueprint': 'sensor.lidar.ray_cast',
    #     'attributes': {'channels': '32', 'range': '50', 'points_per_second': '56000', 'rotation_frequency': '10'},
    #     'location': {'x': 0.0, 'y': 0.0, 'z': 2.0},
    #     'rotation': {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0}
    # },
    'birdseye': {
        'blueprint': 'sensor.camera.rgb',
        'attributes': {'image_size_x': '640', 'image_size_y': '480', 'fov': '90'},
        'location': {'x': 0.0, 'y': 0.0, 'z': 45.0},
        'rotation': {'pitch': -90.0, 'yaw': 0.0, 'roll': 0.0}
    }
}

# --- Helper Functions ---

def setup_sensors(simulator: CarlaSimulator, vehicle: carla.Actor) -> dict:
    """Set up sensors for occupancy mapping."""
    attached_sensors = {}
    for sensor_name, config in SENSOR_CONFIGS.items():
        # Birdseye camera is not attached to the vehicle
        attach_to = vehicle if sensor_name != 'birdseye' else None
        sensor = simulator.attach_sensor(sensor_name, config, attach_to=attach_to)

        if sensor:
            attached_sensors[sensor_name] = sensor
            logging.info(f"Successfully attached {sensor_name} sensor")
        else:
            logging.error(f"Failed to attach {sensor_name} sensor")
    return attached_sensors

def calculate_tsdf_volume_bounds(spawn_location: carla.Location, map_size: float = 200.0) -> np.ndarray:
    """Calculate TSDF volume bounds around spawn location."""

    # Stub out for now - return dummy bounds
    return np.array([[0, 100], [0, 100], [0, 20]])

def collect_synchronized_sensor_data(simulator: CarlaSimulator, expected_sensors: set, timeout: float = 1.0) -> dict:
    """Collect synchronized sensor data from expected sensors."""
    sensor_data = {}
    start_time = time.time()

    while len(sensor_data) < len(expected_sensors) and (time.time() - start_time) < timeout:
        try:
            data = simulator.get_sensor_data(timeout=0.1)
            if data and data.sensor_type not in sensor_data:
                sensor_data[data.sensor_type] = data
        except Exception as e:
            logging.warning(f"Error collecting sensor data: {e}")
            break
    return sensor_data


def create_combined_visualization(output_dir: str, birdseye_image_path: str, occupancy_map_path: str, step: int):
    """Create a combined visualization by merging RGB and occupancy map images."""
    try:
        if not os.path.exists(occupancy_map_path):
            logging.warning(f"Occupancy map not found at {occupancy_map_path}")
            return None

        if not os.path.exists(birdseye_image_path):
            logging.warning(f"Birdseye img not found at {birdseye_image_path}")
            return None

        occupancy_img = plt.imread(occupancy_map_path)
        birdseye_img = plt.imread(birdseye_image_path)


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        ax1.imshow(birdseye_img)
        ax1.set_title(f'RGB Camera View - Step {step}', fontsize=14)
        ax1.axis('off')

        ax2.imshow(occupancy_img)
        ax2.set_title(f'Occupancy Map - Step {step}', fontsize=14)
        ax2.axis('off')

        plt.tight_layout()
        return fig

    except Exception as e:
        logging.error(f"Failed to create combined visualization: {e}")
        return None

def record_step_data(results: dict, step: int, step_start_time: float, vehicle_state: dict, sensor_data: dict):
    """Records relevant data for the current simulation step."""
    step_data = {
        'step': step,
        'timestamp': time.time(),
        'vehicle_location': [vehicle_state['location'].x, vehicle_state['location'].y, vehicle_state['location'].z],
        'vehicle_rotation': [vehicle_state['rotation'].pitch, vehicle_state['rotation'].yaw, vehicle_state['rotation'].roll],
        'velocity': [vehicle_state['velocity'].x, vehicle_state['velocity'].y, vehicle_state['velocity'].z],
        'sensors_active': list(sensor_data.keys()),
        'processing_time': time.time() - step_start_time
    }
    results['steps'].append(step_data)

def convert_to_json_serializable(obj):
    """Recursively convert objects to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

# --- Main Experiment Class ---

class CarlaExplorationExperiment:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.output_dir = self._setup_output_directory()
        self.results = self._initialize_results()
        self.simulator = None
        self.vehicle = None
        self.sensors = {}
        self.tsdf_planner = None
        self.spectator = None

    def _setup_output_directory(self) -> str:
        """Sets up the output directory for the experiment."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{self.cfg.exp_name}_{timestamp}"
        output_dir = os.path.join(self.cfg.output_parent_dir, exp_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def _initialize_results(self) -> dict:
        """Initializes the results dictionary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return {
            'experiment_name': os.path.basename(self.output_dir),
            'config': self.cfg,
            'start_time': timestamp,
            'steps': [],
            'exploration_summary': {}
        }

    def _setup_carla_environment(self) -> bool:
        """Connects to Carla, loads world, spawns vehicle, and sets up sensors."""
        self.simulator = CarlaSimulator(host=self.cfg.carla.host, port=self.cfg.carla.port)

        if not self.simulator.connect():
            logging.error("Failed to connect to Carla server")
            return False

        if not self.simulator.load_world(self.cfg.carla.town):
            logging.error(f"Failed to load world {self.cfg.carla.town}")
            return False

        self.vehicle = self.simulator.spawn_vehicle(self.cfg.carla.vehicle_type)
        if not self.vehicle:
            logging.error("Failed to spawn vehicle")
            return False

        self.sensors = setup_sensors(self.simulator, self.vehicle)
        if not self.sensors:
            logging.error("Failed to set up sensors")
            return False

        if not self.simulator.enable_autopilot(self.cfg.carla.desired_speed):
            logging.error("Failed to enable autopilot")
            return False

        actual_vehicle_state = self.simulator.get_vehicle_state()
        if not actual_vehicle_state:
            logging.error("Failed to get actual vehicle state")
            return False

        actual_location = actual_vehicle_state['location']
        logging.info(f"Vehicle actual location: ({actual_location.x:.1f}, {actual_location.y:.1f}, {actual_location.z:.1f})")

        # Set up spectator camera to follow vehicle
        self.spectator = self.simulator.world.get_spectator()


        self._update_spectator_camera()
        # Wait for vehicle to settle and get actual location
        for i in range(10):
            self.simulator.tick()
            self._update_spectator_camera()

        # Initialize TSDF planner
        # vol_bounds = calculate_tsdf_volume_bounds(actual_location, self.cfg.mapping.map_size)
        # spawn_transform = np.eye(4)
        # spawn_transform[:3, 3] = [actual_location.x, actual_location.y, actual_location.z]
        # tsdf_spawn_transform = carla_to_tsdf_coordinates(spawn_transform)
        # tsdf_spawn_pos = tsdf_spawn_transform[:3, 3]

        # self.tsdf_planner = CarlaTSDFPlanner(
        #     vol_bnds=vol_bounds,
        #     voxel_size=self.cfg.mapping.voxel_size,
        #     vehicle_init_pos=tsdf_spawn_pos,
        #     init_clearance=self.cfg.mapping.init_clearance
        # )
        # logging.info(f"Initialized TSDF planner with bounds: {vol_bounds}")
        self.tsdf_planner = None  # Stub out for now
        return True

    def _update_spectator_camera(self):
        """Updates spectator camera to follow vehicle."""
        if self.vehicle and self.spectator and 'birdseye' in self.sensors:
            vehicle_transform = self.vehicle.get_transform()
            birdseye_view_transform = carla.Transform(
                carla.Location(
                    x=vehicle_transform.location.x,
                    y=vehicle_transform.location.y,
                    z=vehicle_transform.location.z + 15
                ),
                carla.Rotation(
                    pitch=-90,
                    yaw=0,
                    roll=0
                )
            )
            self.spectator.set_transform(birdseye_view_transform)
            # The birdseye sensor should also follow the spectator's view for consistency
            self.sensors['birdseye'].set_transform(birdseye_view_transform)

    def _process_step_data(self, step: int, sensor_data: dict, vehicle_state: dict):
        """Processes sensor data and integrates into TSDF."""
        if len(sensor_data) < 2:  # Need at least 2 sensors
            logging.warning(f"Insufficient sensor data at step {step}: {list(sensor_data.keys())}")
            return False

        # try:
        #     self.tsdf_planner.integrate_carla_data(sensor_data, vehicle_state['transform'])
        #     return True
        # except Exception as e:
        #     logging.error(f"Failed to integrate sensor data at step {step}: {e}")
        #     return False
        
        # Stub out TSDF processing for now
        logging.info(f"Step {step}: Sensor data collected (TSDF processing disabled)")
        return True

    def _save_periodic_results(self, step: int, sensor_data: dict):
        """Saves periodic visualizations and logs progress."""
        # Save occupancy map visualization
        # fig = self.tsdf_planner.visualize_occupancy_map()
        # occupancy_map_path = os.path.join(self.output_dir, f"occupancy_map_step_{step:04d}.png")
        # fig.savefig(occupancy_map_path, dpi=150, bbox_inches='tight')
        # plt.close(fig)

        # Log progress
        # exploration_summary = self.tsdf_planner.get_exploration_summary()
        # logging.info(f"Step {step}: Explored {exploration_summary['exploration_percentage']:.1f}% "
        #              f"({exploration_summary['vehicle_distance_traveled']:.1f}m traveled)")
        
        # Stub out TSDF visualization and logging for now
        logging.info(f"Step {step}: Sensor data saved (TSDF visualization disabled)")

        # Save RGB camera data and combined visualization
        if 'rgb' in sensor_data:
            rgb_data = sensor_data['rgb'].data
            rgb_image_path = os.path.join(self.output_dir, f"rgb_step_{step:04d}.png")
            rgb_data.save_to_disk(rgb_image_path)

        else:
            logging.warning(f"No RGB data available at step {step}")

        # Save birdseye camera data
        if 'birdseye' in sensor_data:
            birdseye_data = sensor_data['birdseye'].data
            birdseye_image_path = os.path.join(self.output_dir, f"birdseye_step_{step:04d}.png")
            birdseye_data.save_to_disk(birdseye_image_path)
        else:
            logging.warning(f"No birdseye data available at step {step}")
        
        # Save RGB depth camera data
        if 'depth' in sensor_data:
            depth_data = sensor_data['depth'].data
            depth_image_path = os.path.join(self.output_dir, f"depth_step_{step:04d}.png")
            depth_data.save_to_disk(depth_image_path)
        else:
            logging.warning(f"No depth data available at step {step}")

        # Save Location
        # loc = self.simulator.vehicle.get_location()

        # # Append to CSV file (create if doesn't exist)
        # path_csv_path = os.path.join(self.output_dir, "path.csv")
        # file_exists = os.path.exists(path_csv_path)
        
        # with open(path_csv_path, "a", newline='') as f:
        #     writer = csv.writer(f, delimiter='\t')
        #     # Write header if file doesn't exist
        #     if not file_exists:
        #         writer.writerow(['step', 'x', 'y', 'z'])
        #     writer.writerow([step, loc.x, loc.y, loc.z])

        trans = self.simulator.vehicle.get_transform().get_matrix()
        #trans = self.sensors['depth'].get_transform().get_matrix()
        #loc = self.sensors['depth'].get_camera_intrinsics()

        with open(os.path.join(self.output_dir, f"matrix_step_{step:04d}.txt"), "w+") as f:
            # Write each row of the 4x4 matrix on a separate line
            for i in range(len(trans)):
                row = trans[i]
                f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f}\n")





        # try:
        #     combined_fig = create_combined_visualization(self.output_dir, birdseye_image_path, occupancy_map_path, step)
        #     if combined_fig:
        #         combined_path = os.path.join(self.output_dir, f"combined_step_{step:04d}.png")
        #         combined_fig.savefig(combined_path, dpi=150, bbox_inches='tight')
        #         plt.close(combined_fig)
        #         logging.debug(f"Saved combined visualization for step {step}")
        # except Exception as e:
        #     logging.error(f"Failed to create combined visualization: {e}")
        
        # Stub out combined visualization for now
        logging.debug(f"Combined visualization disabled for step {step}")
        
    def run(self):
        """Main exploration loop."""
        if not self._setup_carla_environment():
            return

        logging.info(f"Starting exploration for {self.cfg.num_steps} steps")

        try:
            for step in tqdm(range(self.cfg.num_steps), desc="Exploration"):
                step_start_time = time.time()

                self.simulator.tick()
                self._update_spectator_camera()

                vehicle_state = self.simulator.get_vehicle_state()
                if not vehicle_state:
                    logging.warning(f"Failed to get vehicle state at step {step}")
                    continue

                sensor_data = collect_synchronized_sensor_data(self.simulator, set(SENSOR_CONFIGS.keys()))

                if not self._process_step_data(step, sensor_data, vehicle_state):
                    continue

                record_step_data(self.results, step, step_start_time, vehicle_state, sensor_data)

                if step % 1 == 0 or step == self.cfg.num_steps: # Save on last step too
                    print("step: ", step)
                    self._save_periodic_results(step, sensor_data)
                    #self.tsdf_planner.save_raw_voxel_data(f"tsdf_map_step{step:04d}.npz")

            self._finalize_experiment()

        except Exception as e:
            logging.error(f"Experiment failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()

    def _finalize_experiment(self):
        """Performs final processing and saves results."""
        logging.info("Exploration completed, generating final results...")

        # self.results['exploration_summary'] = self.tsdf_planner.get_exploration_summary()

        # final_map_path = os.path.join(self.output_dir, "occupancy_map.png")
        # fig = self.tsdf_planner.visualize_occupancy_map() # visualize_occupancy_map no longer takes path argument
        # fig.savefig(final_map_path, dpi=150, bbox_inches='tight')
        # plt.close(fig)

        # occupancy_data_path = os.path.join(self.output_dir, "occupancy_grid.npz")
        # self.tsdf_planner.save_occupancy_data(occupancy_data_path)
        
        # Stub out TSDF finalization for now
        self.results['exploration_summary'] = {
            'exploration_percentage': 0.0,
            'vehicle_distance_traveled': 0.0,
            'free_voxels': 0,
            'occupied_voxels': 0
        }

        self.results['end_time'] = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results['total_duration'] = time.time() - time.mktime(time.strptime(self.results['start_time'], "%Y%m%d_%H%M%S"))

        # Convert to JSON and save
        json_results_path = os.path.join(self.output_dir, "results.json")
        with open(json_results_path, 'w') as f:
            json_serializable_results = {}
            for key, value in self.results.items():
                if key == 'config':
                    json_serializable_results[key] = OmegaConf.to_container(value, resolve=True)
                else:
                    json_serializable_results[key] = convert_to_json_serializable(value)
            json.dump(json_serializable_results, f, indent=2)

        # Save as pickle for full compatibility
        pickle_results_path = os.path.join(self.output_dir, "results.pkl")
        with open(pickle_results_path, 'wb') as f:
            pickle.dump(self.results, f)

        self._print_summary()

    def _print_summary(self):
        """Prints a summary of the experiment."""
        summary = self.results['exploration_summary']
        logging.info(f"\n==== EXPERIMENT SUMMARY ====")
        logging.info(f"Experiment: {self.results['experiment_name']}")
        logging.info(f"Duration: {self.results['total_duration']:.1f} seconds")
        logging.info(f"Steps completed: {len(self.results['steps'])}")
        logging.info(f"Distance traveled: {summary['vehicle_distance_traveled']:.1f} meters")
        logging.info(f"Area explored: {summary['exploration_percentage']:.1f}%")
        logging.info(f"Free voxels: {summary['free_voxels']}")
        logging.info(f"Occupied voxels: {summary['occupied_voxels']}")
        logging.info(f"Results saved to: {self.output_dir}")

    def _cleanup(self):
        """Cleans up Carla simulator resources."""
        if self.simulator:
            self.simulator.cleanup()

# --- Main execution block ---

def setup_logging(output_dir: str):
    """Configures logging for the experiment."""
    logging_path = os.path.join(output_dir, "log.log")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler()
        ]
    )
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)



def parse_arguments_and_config():
    """Parses command-line arguments and loads/creates configuration."""
    parser = argparse.ArgumentParser(description="Run Carla exploration experiment")
    parser.add_argument("-cf", "--cfg_file", help="Config file path",
                       default="cfg/carla_exp.yaml", type=str)
    parser.add_argument("--host", help="Carla server host", default="10.0.0.101", type=str)
    parser.add_argument("--port", help="Carla server port", default=2000, type=int)
    parser.add_argument("--town", help="Carla town to load", default="Town02", type=str)
    parser.add_argument("--steps", help="Number of exploration steps", default=200, type=int)
    args = parser.parse_args()

    if os.path.exists(args.cfg_file):
        cfg = OmegaConf.load(args.cfg_file)

    # Override config with command line arguments
    cfg.carla.host = args.host
    cfg.carla.port = args.port
    cfg.num_steps = args.steps
    cfg.carla.town = args.town

    OmegaConf.resolve(cfg)
    return cfg

def main():
    """Main entry point for the Carla exploration script."""
    cfg = parse_arguments_and_config()

    # Determine output_dir for logging setup *before* creating the experiment object
    # This is a bit of a chicken-and-egg problem, but necessary for early logging.
    # The experiment object will recalculate its own output_dir, but they should match.
    exp_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_output_dir_for_logging = os.path.join(cfg.output_parent_dir, f"{cfg.exp_name}_{exp_timestamp}")
    os.makedirs(temp_output_dir_for_logging, exist_ok=True) # Ensure dir exists for log file

    setup_logging(temp_output_dir_for_logging)

    logging.info(f"***** Starting Carla Exploration Experiment *****")
    logging.info(f"Config: {OmegaConf.to_yaml(cfg)}")

    experiment = CarlaExplorationExperiment(cfg)

    # Set up signal handler for graceful shutdown
    def signal_handler(signum, frame):
        logging.info("Received interrupt signal, cleaning up...")
        experiment._cleanup() # Call the cleanup method of the experiment instance
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    experiment.run()

if __name__ == "__main__":
    import argparse
    main()