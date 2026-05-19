import numpy as np

class CableDrivenForce:
    def __init__(self, control_input: callable, hole_offset: np.ndarray):
        self.control_input = control_input # control_input = [force_1, force_2, force_3] forces from each cable
        self.hole_offset = hole_offset # (3,3) [[x_hole_1,y_hole_2,0],...] The position of the hole described in the x-y plan

    def compute_force_collection(self, slender_robot, time):

        n = len(slender_robot.robots)
        force_collection = np.zeros((n, 6))  # Bug 1 fixed: np.zeros needs a tuple shape

        for i in range(n):
            force_on_single_robot = np.zeros(6)

            for cable_index in range(3):

                position = slender_robot.robots[i].posture[:3, 3] + self.hole_offset[cable_index]
                orientation_Q = slender_robot.robots[i].posture[:3, :3]


                anchor_point_before = slender_robot.robots[i-1].posture[:3, 3] + self.hole_offset[cable_index] if i != 0 else self.hole_offset[cable_index]          
                anchor_point_after  = slender_robot.robots[i+1].posture[:3, 3] + self.hole_offset[cable_index] if i != n-1 else None

                #-------------------- Compute the Forces in local frame -------------------
                anchor_point_local_before = np.linalg.inv(orientation_Q) @ (anchor_point_before - position)                
                if np.linalg.norm(anchor_point_local_before) > 1e-6:
                    unit_anchor_vector_local_before = anchor_point_local_before / np.linalg.norm(anchor_point_local_before)
                else:
                    
                    unit_anchor_vector_local_before = np.zeros(3)

                if anchor_point_after is not None:
                    anchor_point_local_after = np.linalg.inv(orientation_Q) @ (anchor_point_after - position)
                    # Bug 5 fixed: use local-frame vector, not world-frame anchor_point_after
                    if np.linalg.norm(anchor_point_local_after) > 1e-6:
                        unit_anchor_vector_local_after = anchor_point_local_after / np.linalg.norm(anchor_point_local_after)
                    else:
                        unit_anchor_vector_local_after = np.zeros(3)
                else:
                    unit_anchor_vector_local_after = np.zeros(3)

                linear_force_before = self.control_input(time)[cable_index] * unit_anchor_vector_local_before
                linear_force_after  = self.control_input(time)[cable_index] * unit_anchor_vector_local_after
                linear_total_on_single_robot = linear_force_before + linear_force_after

                # --------------------- Compute Torque in Local Frame ----------------------
                force_arm_before = self.hole_offset[cable_index] + np.array([0.0, 0.0,  slender_robot.robots[i].thickness])
                force_arm_after  = self.hole_offset[cable_index] + np.array([0.0, 0.0, -slender_robot.robots[i].thickness])
                torque_before = np.cross(force_arm_before, linear_force_before)
                torque_after  = np.cross(force_arm_after,  linear_force_after)
                torque_total_on_single_robot = torque_before + torque_after

                # --------------------- Total Force by one Cable ----------------------------
                force_on_single_robot_by_one_cable = np.hstack((linear_total_on_single_robot, torque_total_on_single_robot))
                force_on_single_robot += force_on_single_robot_by_one_cable

            force_collection[i] = force_on_single_robot

        return force_collection
